"""
API nhận diện nhạc & tra cứu bản quyền.

main.py chỉ còn lo phần vòng đời ứng dụng và /health; toàn bộ endpoint nghiệp vụ
nằm ở `backend/api/routes.py` theo đặc tả §11.

Những gì đã sửa ở Giai đoạn 0:
  - Không còn nạp FAISS ở tầng module. Trước đây `faiss.read_index()` chạy ngay
    lúc import trên một file index hỏng -> cả uvicorn lẫn pytest đều sập.
  - Bỏ hoàn toàn nhánh "dummy": bản cũ sinh vector ngẫu nhiên khi thiếu index,
    tức là TRẢ VỀ KẾT QUẢ BỊA. Nay thiếu index thì báo MODEL_FAILURE.
  - Lỗi được map sang mã chuẩn hoá (§12) thay vì ném traceback ra client.
"""
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.api.routes import router
from backend.database.session import SessionLocal, check_connection
from backend.services import (
    audio_service,
    cover_service,
    embedding_service,
    fingerprint_service,
)
from backend.services.decision_service import load_rules
from backend.services.retrieval_service import load_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("music_rights_ai")


def _warm_fingerprints() -> dict:
    db = SessionLocal()
    try:
        return fingerprint_service.warm_up(db)
    finally:
        db.close()


# lambda: tra hàm lúc CHẠY, không phải lúc import (để test thay được từng bước)
WARMUP_STEPS = (
    ("fingerprint", lambda: _warm_fingerprints()),
    ("mert", lambda: embedding_service.warm_up()),
    ("cover", lambda: cover_service.warm_up()),
)


def warm_up(report: dict, plan: dict) -> None:
    """
    Nạp trước các bộ đệm lười để truy vấn ĐẦU TIÊN không phải gánh chúng. Đo
    2026-09-18 (24.375 bản ghi, GTX 1650): Chromaprint ~21 s, MERT ~4,4 s, Cover
    ~0,7 s ở lần đầu, so với ~40 ms / ~0,8 s / ~0,2–1,4 s ở các lần sau.

    Chạy trong luồng nền nên server trả lời ngay; request đến giữa chừng chờ trên
    khoá của bộ đệm đang nạp chứ không nạp lần hai. Một bước hỏng chỉ ghi log — bộ
    đệm đó vẫn nạp lười ở truy vấn đầu tiên như trước.
    """
    for name, step in WARMUP_STEPS:
        if not plan.get(name):
            report["steps"][name] = {"status": "SKIPPED"}
            continue
        started = time.perf_counter()
        try:
            step()
            status = "READY"
        except Exception:
            # Nội dung lỗi có thể chứa đường dẫn máy chủ; /health là công khai -> log
            logger.exception("Lam nong '%s' that bai", name)
            status = "FAILED"
        report["steps"][name] = {"status": status,
                                 "seconds": round(time.perf_counter() - started, 1)}
    report["status"] = "DONE"
    logger.info("Lam nong xong: %s", report["steps"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Nạp tài nguyên lúc khởi động; hỏng thì đánh dấu DEGRADED chứ không sập."""
    app.state.vector_index = None
    app.state.index_error = None
    app.state.rules_error = None
    app.state.rules_version = None

    try:
        app.state.vector_index = load_index(
            config.FAISS_INDEX_PATH, config.FAISS_ID_MAP_PATH
        )
        logger.info(
            "FAISS san sang: %s vector / %s recording (%s)",
            app.state.vector_index.ntotal,
            app.state.vector_index.n_recordings,
            config.FAISS_INDEX_PATH,
        )
    except Exception as e:
        app.state.index_error = f"{type(e).__name__}: {e}"
        logger.error(
            "Khong nap duoc FAISS index -> tang 2 (MERT) se bao loi. %s",
            app.state.index_error,
        )

    try:
        rules = load_rules()
        app.state.rules_version = rules.get("name")
        logger.info("Rule Engine: %s (%s nhom)", rules.get("name"), len(rules["groups"]))
    except Exception as e:
        app.state.rules_error = f"{type(e).__name__}: {e}"
        logger.error("Khong nap duoc rules: %s", app.state.rules_error)

    db_ok, db_msg = check_connection()
    logger.info("Database: %s", db_msg)
    if not os.getenv("DATABASE_URL"):
        logger.error(
            "DATABASE_URL chua duoc dat (xem .env.example). Ma nguon KHONG con mat "
            "khau mac dinh viet cung, nen may chu se o trang thai DEGRADED."
        )

    if not config.API_KEY:
        logger.warning(
            "API_KEY chua duoc dat -> cac endpoint /api/v1/* khong yeu cau xac thuc."
        )

    if not fingerprint_service.is_available():
        logger.warning(
            "Chromaprint chua san sang (%s) -> tang 1 se bi bo qua. "
            "Tai fpcalc: https://github.com/acoustid/chromaprint/releases",
            fingerprint_service.backend_status(),
        )
    if not audio_service.ffmpeg_available():
        logger.warning(
            "FFmpeg khong kha dung -> chi xu ly duoc file audio, khong xu ly video."
        )

    # Sau FAISS/rules/CSDL: chỉ làm nóng tầng sẽ thật sự chạy. Không có chỉ mục FAISS
    # thì cascade báo MODEL_FAILURE trước khi gọi MERT; thiếu fpcalc thì bỏ tầng 1.
    app.state.warmup = {"status": "DISABLED", "steps": {}}
    if config.WARMUP_ON_STARTUP:
        app.state.warmup = {"status": "RUNNING", "steps": {}}
        plan = {
            "fingerprint": db_ok and fingerprint_service.is_available(),
            "mert": app.state.vector_index is not None,
            "cover": config.COVER_ENABLED,
        }
        threading.Thread(target=warm_up, args=(app.state.warmup, plan),
                         name="warmup", daemon=True).start()

    yield


app = FastAPI(
    title="Music Rights AI Platform API",
    description="Hệ thống AI nhận diện âm nhạc và truy vấn bản quyền đa tầng",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    """
    Tham số sai (vd. platform=youtube thay vì YOUTUBE) -> 422 cùng khuôn mã lỗi.

    Mặc định FastAPI trả {"detail": [ {type, loc, msg, input, ...} ]} — một danh sách,
    nên apiError của frontend không đọc được mã lỗi, và trường `input` lặp lại nguyên
    giá trị người dùng gửi. Ở đây chỉ nêu TÊN trường và giá trị được chấp nhận.
    """
    details = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error.get("loc", ())
                         if part not in ("body", "query", "path", "form", "header"))
        details.append({"field": field or None, "message": error.get("msg", "Không hợp lệ")})
    summary = "; ".join(f"{d['field']}: {d['message']}" if d["field"] else d["message"]
                        for d in details)
    return JSONResponse(
        status_code=422,
        content={"detail": {"error_code": "INVALID_REQUEST",
                            "message": f"Tham số không hợp lệ — {summary}",
                            "details": details}},
    )


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    """
    Lưới an toàn cuối: lỗi không lường trước vẫn trả đúng khuôn của api_error
    ({"detail": {"error_code", "message"}}); chi tiết chỉ nằm trong log máy chủ (§12).
    Trước đây client nhận trang "Internal Server Error" trơn, không có mã lỗi.
    """
    logger.exception("Loi khong luong truoc tai %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": {"error_code": "INTERNAL_ERROR",
                            "message": "Lỗi nội bộ máy chủ. Chi tiết đã được ghi vào log."}},
    )


@app.get("/health", tags=["Health Check"])
def health_check():
    """Báo trạng thái từng thành phần — không che giấu thành phần đang hỏng."""
    db_ok, db_msg = check_connection()
    vector_index = getattr(app.state, "vector_index", None)

    components = {
        "database": db_msg,
        "faiss": {
            "status": "READY" if vector_index else "UNAVAILABLE",
            "total_vectors": vector_index.ntotal if vector_index else 0,
            "total_recordings": vector_index.n_recordings if vector_index else 0,
            # Chỉ báo CÓ lỗi; nội dung lỗi chứa đường dẫn trên máy chủ nên để trong log
            "error": "INDEX_LOAD_FAILED" if getattr(app.state, "index_error", None) else None,
        },
        "chromaprint": {
            "status": "AVAILABLE" if fingerprint_service.is_available() else "MISSING",
            # Bỏ fpcalc_path: đường dẫn tuyệt đối chứa cả tên người dùng Windows
            **{key: value for key, value in fingerprint_service.backend_status().items()
               if key != "fpcalc_path"},
        },
        "ffmpeg": "AVAILABLE" if audio_service.ffmpeg_available() else "MISSING (chi audio)",
        "mert": {
            "model": config.MERT_MODEL,
            "loaded": embedding_service.is_loaded(),
        },
        "rule_engine": {
            "status": "READY" if getattr(app.state, "rules_version", None) else "UNAVAILABLE",
            "version": getattr(app.state, "rules_version", None),
            "error": "RULES_LOAD_FAILED" if getattr(app.state, "rules_error", None) else None,
        },
        "cover": {
            "status": "READY" if os.path.exists(config.COVER_INDEX_PATH) else "STANDBY",
            "enabled": config.COVER_ENABLED,
            "threshold": config.COVER_THRESHOLD,
        },
    }
    degraded = (not db_ok) or vector_index is None or app.state.rules_error is not None
    warmup = getattr(app.state, "warmup", None) or {"status": "DISABLED", "steps": {}}
    health_res = {
        "status": "DEGRADED" if degraded else "ONLINE",
        "components": components,
        # Bản sao: luồng làm nóng vẫn đang ghi vào dict gốc trong lúc response
        # được tuần tự hoá. Chỉ trạng thái + số giây, không có nội dung lỗi.
        "warmup": {"status": warmup["status"], "steps": dict(warmup["steps"])},
    }
    if getattr(config, "EVIDENCE_DETAIL", "full") != "public":
        health_res["thresholds"] = {
            "fingerprint": config.FP_THRESHOLD,
            "embedding": config.MERT_THRESHOLD,
            "cover": config.COVER_THRESHOLD,
            "note": "fingerprint hieu chinh bang EXP-01, embedding bang EXP-06, cover bang EXP-07; chay lai khi mo rong du lieu tham chieu.",
        }
    else:
        if "cover" in components and isinstance(components["cover"], dict):
            components["cover"].pop("threshold", None)

    return health_res


# --------------------------------------------------------------------------
# Frontend (§13): 4 màn hình MVP được phục vụ tĩnh ngay từ backend, không cần
# web server riêng và không có bước build nào.
#
# Mount ở "/" nên phải đặt SAU router và /health — Starlette khớp route theo thứ
# tự đăng ký, mount cuối cùng chỉ nhận những đường dẫn còn lại.
# --------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(config.BASE_DIR, "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    logger.warning("Khong tim thay thu muc frontend/: %s", FRONTEND_DIR)
