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
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.api.routes import router
from backend.database.session import check_connection
from backend.services import audio_service, embedding_service, fingerprint_service
from backend.services.decision_service import load_rules
from backend.services.retrieval_service import load_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("music_rights_ai")


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

    yield


app = FastAPI(
    title="Music Rights AI Platform API",
    description="Hệ thống AI nhận diện âm nhạc và truy vấn bản quyền đa tầng",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(router)


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
            "error": getattr(app.state, "index_error", None),
        },
        "chromaprint": {
            "status": "AVAILABLE" if fingerprint_service.is_available() else "MISSING",
            **fingerprint_service.backend_status(),
        },
        "ffmpeg": "AVAILABLE" if audio_service.ffmpeg_available() else "MISSING (chi audio)",
        "mert": {
            "model": config.MERT_MODEL,
            "loaded": embedding_service.is_loaded(),
        },
        "rule_engine": {
            "status": "READY" if getattr(app.state, "rules_version", None) else "UNAVAILABLE",
            "version": getattr(app.state, "rules_version", None),
            "error": getattr(app.state, "rules_error", None),
        },
        "cover": {
            "status": "READY" if os.path.exists(config.COVER_INDEX_PATH) else "STANDBY",
            "enabled": config.COVER_ENABLED,
            "threshold": config.COVER_THRESHOLD,
        },
    }
    degraded = (not db_ok) or vector_index is None or app.state.rules_error is not None
    return {
        "status": "DEGRADED" if degraded else "ONLINE",
        "components": components,
        "thresholds": {
            "fingerprint": config.FP_THRESHOLD,
            "embedding": config.MERT_THRESHOLD,
            "cover": config.COVER_THRESHOLD,
            "note": "fingerprint hieu chinh bang EXP-01, embedding bang EXP-06, cover bang EXP-07; chay lai khi mo rong du lieu tham chieu.",
        },
    }


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
