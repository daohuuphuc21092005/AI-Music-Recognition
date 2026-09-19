"""
Các endpoint theo đặc tả CLAUDE.md §11.

    POST /api/v1/analyze            -> nhận file + mục đích sử dụng, trả job_id
    GET  /api/v1/jobs/{job_id}      -> QUEUED | PROCESSING | DONE | FAILED
    GET  /api/v1/results/{job_id}   -> kết quả đầy đủ kèm evidence
    GET  /api/v1/tracks/{recording_id}
    POST /api/v1/feedback           -> Correct | Incorrect | Unsure

    POST /api/v1/search             -> bản đồng bộ (giữ lại để demo/thử nhanh)

Mọi lỗi đều trả mã chuẩn hoá (§12), không bao giờ lộ traceback.
"""
import logging
import os
import shutil
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from backend import config
from backend.database.session import SessionLocal, get_db
from backend.schemas.analysis import (
    AnalyzeAccepted,
    FeedbackRequest,
    FeedbackResponse,
    JobStatusResponse,
    Platform,
)
from backend.security import (
    filter_public_evidence,
    get_client_ip,
    job_concurrency_limiter,
    rate_limiter,
    sanitize_filename,
    verify_api_key,
)
from backend.services import job_service
from backend.services.analysis_pipeline import analyze_audio
from backend.services.audio_service import AudioProcessingError
from backend.services.cascade_service import PipelineError
from backend.services.rights_service import get_full_music_rights

logger = logging.getLogger("music_rights_ai")

router = APIRouter(prefix="/api/v1", dependencies=[Depends(verify_api_key)])

# Mã lỗi chuẩn hoá theo §12 -> HTTP status
ERROR_STATUS = {
    "FILE_TOO_LARGE": 413,
    "UNSUPPORTED_FORMAT": 415,
    "RATE_LIMITED": 429,
    "UNAUTHORIZED": 401,
    "NO_AUDIO": 422,
    "NO_MUSIC": 422,
    "MODEL_FAILURE": 503,
    "DATABASE_FAILURE": 503,
    "TIMEOUT": 504,
    "UNKNOWN_TRACK": 404,
    "LOW_CONFIDENCE": 422,
    "INTERNAL_ERROR": 500,
}


def api_error(code: str, message: str, headers: dict = None) -> HTTPException:
    return HTTPException(
        status_code=ERROR_STATUS.get(code, 500),
        detail={"error_code": code, "message": message},
        headers=headers,
    )


def parse_uuid(value, what: str = "Mã") -> str:
    """
    Kiểm dạng UUID TRƯỚC khi chạm CSDL: `/jobs/abc` từng làm PostgreSQL ném lỗi ép
    kiểu và trả về một trang 500 trơn, không theo khuôn mã lỗi (§12).
    """
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        raise api_error("UNKNOWN_TRACK", f"{what} không phải UUID hợp lệ: {str(value)[:64]}") from None


def safe_mark_failed(db, job_id: str, code: str, message: str) -> None:
    """
    Ghi FAILED mà không để chính việc ghi làm hỏng luồng xử lý lỗi: khi CSDL sập,
    mark_failed cũng ném lỗi ngay trong khối except và job treo PROCESSING mãi.
    """
    try:
        job_service.mark_failed(db, job_id, code, message)
    except Exception as e:
        logger.error("job=%s khong ghi duoc trang thai FAILED [%s]: %s", job_id, code, e)


def check_upload_limits(request: Request, file: UploadFile) -> None:
    """Kiểm tra sớm Content-Length từ header và đuôi file trước khi ghi đĩa."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > config.MAX_UPLOAD_MB * 1024 * 1024:
                raise api_error(
                    "FILE_TOO_LARGE",
                    f"Dung lượng tải lên vượt quá giới hạn {config.MAX_UPLOAD_MB} MB",
                )
        except ValueError:
            pass

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise api_error(
            "UNSUPPORTED_FORMAT",
            f"Đuôi file '{ext or 'không rõ'}' không được hỗ trợ. "
            f"Chấp nhận: {', '.join(config.ALLOWED_EXTENSIONS)}",
        )


def save_upload(file: UploadFile) -> str:
    """
    Lưu file với tên do server sinh — kiểm tra đuôi file trước khi ghi
    và đếm byte khi ghi để huỷ sớm nếu vượt MAX_UPLOAD_MB.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise api_error(
            "UNSUPPORTED_FORMAT",
            f"Đuôi file '{ext or 'không rõ'}' không được hỗ trợ. "
            f"Chấp nhận: {', '.join(config.ALLOWED_EXTENSIONS)}",
        )

    os.makedirs(config.TEMP_UPLOAD_DIR, exist_ok=True)
    # Chỉ dùng đuôi đã chuẩn hoá thuộc danh sách cho phép để đặt tên
    path = os.path.join(config.TEMP_UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    total_bytes = 0
    chunk_size = 65536  # 64KB

    try:
        with open(path, "wb") as buffer:
            while True:
                chunk = file.file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise api_error(
                        "FILE_TOO_LARGE",
                        f"Kích thước file vượt quá giới hạn {config.MAX_UPLOAD_MB} MB",
                    )
                buffer.write(chunk)
    except Exception:
        cleanup(path)
        raise

    return path


def cleanup(path: str) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def run_job(job_id: str, audio_path: str, filename: str, usage_context: dict,
            vector_index) -> None:
    """Chạy nền: dùng session RIÊNG vì session của request đã đóng."""
    db = SessionLocal()
    safe_name = sanitize_filename(filename)
    try:
        job_service.mark_processing(db, job_id)
        result = analyze_audio(
            audio_path, safe_name, db, vector_index, usage_context,
            on_stage=lambda stage: job_service.mark_stage(db, job_id, stage),
        )
        job_service.mark_done(db, job_id, result)

        try:
            job_service.log_analysis_result(db, job_id, result)
        except Exception as log_err:
            logger.warning("job=%s khong ghi duoc analysis_results: %s", job_id, log_err)

        timings = (result.get("evidence", {}).get("identification", {})
                   .get("timings_ms", {}))
        logger.info(
            "job=%s file=%s match=%s risk=%s conf(id/rights/decision)=%.3f/%.3f/%.3f "
            "total=%.0fms stages=%s",
            job_id, safe_name, result["match"]["type"], result["assessment"]["risk"],
            result["assessment"]["identity_confidence"],
            result["assessment"]["rights_confidence"],
            result["assessment"]["decision_confidence"],
            result.get("latency_ms", 0), timings,
        )
    except (AudioProcessingError, PipelineError) as e:
        logger.info("job=%s that bai [%s]: %s", job_id, e.code, e.message)
        safe_mark_failed(db, job_id, e.code, e.message)
    except Exception as e:
        logger.exception("job=%s loi khong luong truoc: %s", job_id, e)
        safe_mark_failed(db, job_id, "INTERNAL_ERROR",
                         "Loi noi bo khi xu ly file. Xem log may chu.")
    finally:
        db.close()
        cleanup(audio_path)
        job_concurrency_limiter.release()


@router.post("/analyze", response_model=AnalyzeAccepted, status_code=202,
             tags=["Music Identification"])
def analyze(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(..., description="File audio hoặc video cần phân tích"),
    platform: Platform = Form(Platform.YOUTUBE),
    commercial_use: bool = Form(False),
    monetization: bool = Form(False),
    db: Session = Depends(get_db),
):
    """Nhận file và mục đích sử dụng, xử lý nền, trả về job_id để theo dõi."""
    # 1. Rate limiting theo IP
    client_ip = get_client_ip(request)
    allowed, retry_after = rate_limiter.check(client_ip)
    if not allowed:
        raise api_error(
            "RATE_LIMITED",
            f"Quá số lượng yêu cầu cho phép ({config.RATE_LIMIT_PER_MIN}/phút). Thử lại sau {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    # 2. Kiểm tra giới hạn upload sớm (Content-Length và đuôi file)
    check_upload_limits(request, file)
    safe_filename = sanitize_filename(file.filename)

    # 3. Giới hạn số job phân tích chạy đồng thời (Semaphore non-blocking)
    if not job_concurrency_limiter.acquire():
        raise api_error(
            "RATE_LIMITED",
            "Hệ thống đang quá tải với số tác vụ phân tích tối đa. Thử lại sau.",
            headers={"Retry-After": "30"},
        )

    usage_context = {
        "platform": platform.value,
        "commercial_use": commercial_use,
        "monetization": monetization,
    }

    try:
        audio_path = save_upload(file)
    except Exception:
        job_concurrency_limiter.release()
        raise

    try:
        job_id = job_service.create_job(db, safe_filename, usage_context)
    except Exception as e:
        cleanup(audio_path)
        job_concurrency_limiter.release()
        logger.error("Khong tao duoc job: %s", e)
        raise api_error("DATABASE_FAILURE", "Không tạo được job phân tích.")

    background_tasks.add_task(
        run_job, job_id, audio_path, safe_filename, usage_context,
        getattr(request.app.state, "vector_index", None),
    )

    return AnalyzeAccepted(
        job_id=job_id, status="QUEUED",
        message="Đã nhận file. Theo dõi tiến trình tại /api/v1/jobs/{job_id}.",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = job_service.get_job(db, parse_uuid(job_id, "job_id"))
    if not job:
        raise api_error("UNKNOWN_TRACK", f"Không tìm thấy job {job_id}")
    return JobStatusResponse(
        job_id=str(job.job_id),
        status=job.status,
        stage=job.stage or ("QUEUED" if job.status == "QUEUED" else None),
        filename=job.filename,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=str(job.created_at) if job.created_at else None,
        updated_at=str(job.updated_at) if job.updated_at else None,
    )


@router.get("/results/{job_id}", tags=["Jobs"])
def get_job_result(job_id: str, db: Session = Depends(get_db)):
    job = job_service.get_job(db, parse_uuid(job_id, "job_id"))
    if not job:
        raise api_error("UNKNOWN_TRACK", f"Không tìm thấy job {job_id}")

    if job.status == "FAILED":
        raise api_error(job.error_code or "INTERNAL_ERROR",
                        job.error_message or "Job thất bại.")
    if job.status != "DONE":
        # Chưa xong thì nói rõ là chưa xong, không trả kết quả rỗng
        return {"status": job.status, "job_id": str(job.job_id),
                "message": "Job chưa hoàn tất. Thử lại sau."}

    raw_result = {"status": "DONE", "job_id": str(job.job_id), **(job.result or {})}
    return filter_public_evidence(raw_result)


@router.get("/tracks/{recording_id}", tags=["Copyright & Licensing"])
def get_track(recording_id: str, db: Session = Depends(get_db)):
    result = get_full_music_rights(parse_uuid(recording_id, "recording_id"), db)
    if result.get("status") == "NOT_FOUND":
        raise api_error("UNKNOWN_TRACK", result["message"])
    return result


@router.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def submit_feedback(payload: FeedbackRequest, db: Session = Depends(get_db)):
    recording_id = (parse_uuid(payload.recording_id, "recording_id")
                    if payload.recording_id else None)
    if payload.job_id:
        job = job_service.get_job(db, parse_uuid(payload.job_id, "job_id"))
        if not job:
            raise api_error("UNKNOWN_TRACK", f"Không tìm thấy job {payload.job_id}")
        if not recording_id and job.result:
            recording_id = (job.result.get("identity") or {}).get("recording_id")

    try:
        feedback_id = job_service.save_feedback(
            db, payload.job_id, recording_id, payload.verdict.value, payload.note
        )
    except Exception as e:
        logger.error("Khong luu duoc feedback: %s", e)
        raise api_error("DATABASE_FAILURE", "Không lưu được phản hồi.")

    return FeedbackResponse(feedback_id=feedback_id,
                            message="Đã ghi nhận phản hồi. Cảm ơn bạn!")


@router.post("/search", tags=["Music Identification"])
def search_music(request: Request, file: UploadFile = File(...),
                 platform: Platform = Form(Platform.YOUTUBE),
                 commercial_use: bool = Form(False),
                 monetization: bool = Form(False),
                 db: Session = Depends(get_db)):
    """
    Bản ĐỒNG BỘ của /analyze: chạy xong mới trả về.
    Tiện để thử nhanh và cho các script thí nghiệm; luồng chính nên dùng /analyze.
    """
    # 1. Rate limiting theo IP
    client_ip = get_client_ip(request)
    allowed, retry_after = rate_limiter.check(client_ip)
    if not allowed:
        raise api_error(
            "RATE_LIMITED",
            f"Quá số lượng yêu cầu cho phép ({config.RATE_LIMIT_PER_MIN}/phút). Thử lại sau {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    # 2. Kiểm tra giới hạn upload sớm
    check_upload_limits(request, file)
    safe_filename = sanitize_filename(file.filename)

    # 3. Giới hạn số job phân tích chạy đồng thời
    if not job_concurrency_limiter.acquire():
        raise api_error(
            "RATE_LIMITED",
            "Hệ thống đang quá tải với số tác vụ phân tích tối đa. Thử lại sau.",
            headers={"Retry-After": "30"},
        )

    usage_context = {
        "platform": platform.value,
        "commercial_use": commercial_use,
        "monetization": monetization,
    }

    try:
        audio_path = save_upload(file)
    except Exception:
        job_concurrency_limiter.release()
        raise

    job_id = str(uuid.uuid4())

    try:
        result = analyze_audio(
            audio_path, safe_filename, db,
            getattr(request.app.state, "vector_index", None), usage_context,
        )
        logger.info(
            "job=%s file=%s match=%s risk=%s total=%.0fms",
            job_id, safe_filename, result["match"]["type"],
            result["assessment"]["risk"], result.get("latency_ms", 0),
        )
        full_res = {"status": "DONE", "job_id": job_id, **result}
        return filter_public_evidence(full_res)
    except (AudioProcessingError, PipelineError) as e:
        logger.info("job=%s that bai [%s]: %s", job_id, e.code, e.message)
        raise api_error(e.code, e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("job=%s loi khong luong truoc: %s", job_id, e)
        raise api_error("INTERNAL_ERROR", "Loi noi bo khi xu ly file. Xem log may chu.")
    finally:
        cleanup(audio_path)
        job_concurrency_limiter.release()
