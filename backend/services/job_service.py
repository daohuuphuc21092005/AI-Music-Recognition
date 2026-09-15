"""
Quản lý vòng đời job phân tích (QUEUED -> PROCESSING -> DONE | FAILED).

Job được lưu trong PostgreSQL chứ không giữ trong bộ nhớ tiến trình, để kết quả
vẫn tra cứu được sau khi server khởi động lại và khi chạy nhiều worker.
"""
import json
import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session


def create_job(db: Session, filename: str, usage_context: dict) -> str:
    job_id = str(uuid.uuid4())
    db.execute(text("""
        INSERT INTO jobs (job_id, status, filename, platform, commercial_use, monetization)
        VALUES (:job_id, 'QUEUED', :filename, :platform, :commercial_use, :monetization)
    """), {
        "job_id": job_id,
        "filename": filename,
        "platform": usage_context.get("platform"),
        "commercial_use": usage_context.get("commercial_use"),
        "monetization": usage_context.get("monetization"),
    })
    db.commit()
    return job_id


def mark_processing(db: Session, job_id: str) -> None:
    db.execute(text("""
        UPDATE jobs SET status = 'PROCESSING', updated_at = CURRENT_TIMESTAMP
        WHERE job_id = :job_id
    """), {"job_id": job_id})
    db.commit()


def mark_stage(db: Session, job_id: str, stage: str) -> None:
    """
    Ghi bước đang chạy để client theo dõi tiến trình THẬT (§13: không hiển thị
    "AI is thinking"). Lỗi ghi tiến trình không được làm hỏng job.
    """
    try:
        db.execute(text("""
            UPDATE jobs SET stage = :stage, updated_at = CURRENT_TIMESTAMP
            WHERE job_id = :job_id
        """), {"job_id": job_id, "stage": stage})
        db.commit()
    except Exception:
        db.rollback()


def mark_done(db: Session, job_id: str, result: dict) -> None:
    db.execute(text("""
        UPDATE jobs
        SET status = 'DONE', stage = 'DONE', result = CAST(:result AS JSONB),
            updated_at = CURRENT_TIMESTAMP
        WHERE job_id = :job_id
    """), {"job_id": job_id, "result": json.dumps(result, ensure_ascii=False, default=str)})
    db.commit()


def mark_failed(db: Session, job_id: str, error_code: str, error_message: str) -> None:
    db.execute(text("""
        UPDATE jobs
        SET status = 'FAILED', error_code = :code, error_message = :message,
            updated_at = CURRENT_TIMESTAMP
        WHERE job_id = :job_id
    """), {"job_id": job_id, "code": error_code, "message": error_message})
    db.commit()


def get_job(db: Session, job_id: str):
    return db.execute(text("""
        SELECT job_id, status, stage, filename, platform, commercial_use,
               monetization, error_code, error_message, result,
               created_at, updated_at
        FROM jobs WHERE job_id = :job_id
    """), {"job_id": job_id}).fetchone()


def log_analysis_result(db: Session, job_id: str, result: dict,
                        query_id: str = None) -> None:
    """Ghi nhật ký vào analysis_results (§12). Lỗi ghi log không làm hỏng job."""
    evidence = result.get("evidence", {})
    identification = evidence.get("identification", {}) or {}
    fingerprint = identification.get("fingerprint") or {}
    embedding = identification.get("embedding") or {}
    top_k = (embedding.get("top_k") or []) if embedding else []

    try:
        db.execute(text("""
            INSERT INTO analysis_results (
                job_id, query_id, recording_candidate, composition_candidate,
                fingerprint_score, embedding_score, cover_score, match_type,
                risk_level, confidence, decision_reason, latency_ms, model_version
            ) VALUES (
                :job_id, :query_id, :rec_id, :comp_id,
                :fp_score, :emb_score, NULL, :match_type,
                :risk_level, :confidence, :reason, :latency, :model_ver
            )
            ON CONFLICT (job_id) DO NOTHING
        """), {
            "job_id": job_id,
            "query_id": query_id,
            "rec_id": result["identity"].get("recording_id"),
            "comp_id": result["identity"].get("composition_id"),
            "fp_score": fingerprint.get("fingerprint_score"),
            "emb_score": top_k[0]["similarity_score"] if top_k else None,
            "match_type": result["match"].get("type"),
            "risk_level": result["assessment"].get("risk"),
            "confidence": result["assessment"].get("decision_confidence"),
            "reason": result["assessment"].get("reason"),
            "latency": result.get("latency_ms"),
            "model_ver": result.get("model_version"),
        })
        db.commit()
    except Exception:
        db.rollback()
        raise


def save_feedback(db: Session, job_id: str, recording_id: str,
                  verdict: str, note: str) -> str:
    feedback_id = str(uuid.uuid4())
    db.execute(text("""
        INSERT INTO feedback (feedback_id, job_id, recording_id, verdict, note)
        VALUES (:feedback_id, :job_id, :recording_id, :verdict, :note)
    """), {
        "feedback_id": feedback_id,
        "job_id": job_id,
        "recording_id": recording_id,
        "verdict": verdict,
        "note": note,
    })
    db.commit()
    return feedback_id
