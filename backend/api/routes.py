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
from backend.services import job_service
from backend.services.analysis_pipeline import analyze_audio
from backend.services.audio_service import AudioProcessingError
from backend.services.cascade_service import PipelineError
from backend.services.rights_service import get_full_music_rights

logger = logging.getLogger("music_rights_ai")

router = APIRouter(prefix="/api/v1")

# Mã lỗi chuẩn hoá theo §12 -> HTTP status
ERROR_STATUS = {
    "FILE_TOO_LARGE": 413,
    "UNSUPPORTED_FORMAT": 415,
    "NO_AUDIO": 422,
    "NO_MUSIC": 422,
    "MODEL_FAILURE": 503,
    "DATABASE_FAILURE": 503,
    "TIMEOUT": 504,
    "UNKNOWN_TRACK": 404,
    "LOW_CONFIDENCE": 422,
    "INTERNAL_ERROR": 500,
}


def api_error(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=ERROR_STATUS.get(code, 500),
        detail={"error_code": code, "message": message},
    )


def save_upload(file: UploadFile) -> str:
    """Lưu file với tên do server sinh — không tin tên file của client."""
    os.makedirs(config.TEMP_UPLOAD_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1].lower()
    path = os.path.join(config.TEMP_UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
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
    try:
        job_service.mark_processing(db, job_id)
        result = analyze_audio(
            audio_path, filename, db, vector_index, usage_context,
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
            job_id, filename, result["match"]["type"], result["assessment"]["risk"],
            result["assessment"]["identity_confidence"],
            result["assessment"]["rights_confidence"],
            result["assessment"]["decision_confidence"],
            result.get("latency_ms", 0), timings,
        )
    except (AudioProcessingError, PipelineError) as e:
        logger.info("job=%s that bai [%s]: %s", job_id, e.code, e.message)
        job_service.mark_failed(db, job_id, e.code, e.message)
    except Exception as e:
        logger.exception("job=%s loi khong luong truoc: %s", job_id, e)
        job_service.mark_failed(db, job_id, "INTERNAL_ERROR",
                                "Loi noi bo khi xu ly file. Xem log may chu.")
    finally:
        db.close()
        cleanup(audio_path)


@router.post("/analyze", response_model=AnalyzeAccepted, status_code=202,
             tags=["Music Identification"])
async def analyze(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(..., description="File audio hoặc video cần phân tích"),
    platform: Platform = Form(Platform.YOUTUBE),
    commercial_use: bool = Form(False),
    monetization: bool = Form(False),
    db: Session = Depends(get_db),
):
    """Nhận file và mục đích sử dụng, xử lý nền, trả về job_id để theo dõi."""
    usage_context = {
        "platform": platform.value,
        "commercial_use": commercial_use,
        "monetization": monetization,
    }

    audio_path = save_upload(file)
    try:
        job_id = job_service.create_job(db, file.filename, usage_context)
    except Exception as e:
        cleanup(audio_path)
        logger.error("Khong tao duoc job: %s", e)
        raise api_error("DATABASE_FAILURE", "Không tạo được job phân tích.")

    background_tasks.add_task(
        run_job, job_id, audio_path, file.filename, usage_context,
        getattr(request.app.state, "vector_index", None),
    )

    return AnalyzeAccepted(
        job_id=job_id, status="QUEUED",
        message="Đã nhận file. Theo dõi tiến trình tại /api/v1/jobs/{job_id}.",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = job_service.get_job(db, job_id)
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
    job = job_service.get_job(db, job_id)
    if not job:
        raise api_error("UNKNOWN_TRACK", f"Không tìm thấy job {job_id}")

    if job.status == "FAILED":
        raise api_error(job.error_code or "INTERNAL_ERROR",
                        job.error_message or "Job thất bại.")
    if job.status != "DONE":
        # Chưa xong thì nói rõ là chưa xong, không trả kết quả rỗng
        return {"status": job.status, "job_id": str(job.job_id),
                "message": "Job chưa hoàn tất. Thử lại sau."}

    return {"status": "DONE", "job_id": str(job.job_id), **(job.result or {})}


@router.get("/tracks/{recording_id}", tags=["Copyright & Licensing"])
def get_track(recording_id: str, db: Session = Depends(get_db)):
    result = get_full_music_rights(str(recording_id), db)
    if result.get("status") == "NOT_FOUND":
        raise api_error("UNKNOWN_TRACK", result["message"])
    return result


@router.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def submit_feedback(payload: FeedbackRequest, db: Session = Depends(get_db)):
    recording_id = payload.recording_id
    if payload.job_id:
        job = job_service.get_job(db, payload.job_id)
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
async def search_music(request: Request, file: UploadFile = File(...),
                       platform: Platform = Form(Platform.YOUTUBE),
                       commercial_use: bool = Form(False),
                       monetization: bool = Form(False),
                       db: Session = Depends(get_db)):
    """
    Bản ĐỒNG BỘ của /analyze: chạy xong mới trả về.
    Tiện để thử nhanh và cho các script thí nghiệm; luồng chính nên dùng /analyze.
    """
    usage_context = {
        "platform": platform.value,
        "commercial_use": commercial_use,
        "monetization": monetization,
    }
    audio_path = save_upload(file)
    job_id = str(uuid.uuid4())

    try:
        result = analyze_audio(
            audio_path, file.filename, db,
            getattr(request.app.state, "vector_index", None), usage_context,
        )
        logger.info(
            "job=%s file=%s match=%s risk=%s total=%.0fms",
            job_id, file.filename, result["match"]["type"],
            result["assessment"]["risk"], result.get("latency_ms", 0),
        )
        return {"status": "DONE", "job_id": job_id, **result}
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
