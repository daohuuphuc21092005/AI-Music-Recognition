"""
Phễu Cascade (CLAUDE.md §4):

    QUERY -> Chromaprint
      ├─ score >= τFP        -> EXACT_MATCH
      └─ score <  τFP        -> MERT embedding -> Top-K recording
                                  ├─ similarity >= τMERT -> NEAR_MATCH
                                  └─ similarity <  τMERT -> UNKNOWN

Ba điểm khác bản cũ:
  1. Có ngưỡng τMERT. Bản cũ luôn trả NEAR_MATCH kể cả similarity 0.1, nên hệ
     thống không bao giờ ra được UNKNOWN — vi phạm §2 ("không ép UNKNOWN").
  2. Top-K là K bản ghi KHÁC NHAU (gộp theo recording, xem retrieval_service).
  3. Luôn kèm `evidence` + `identity_confidence` tách bạch, không trả một con
     số "độ tin cậy" duy nhất (§2).
"""
import logging
import time

from sqlalchemy.orm import Session

from backend import config
from backend.services.embedding_service import (
    EmbeddingAudioError,
    EmbeddingModelError,
    extract_mert_embedding,
)
from backend.services.fingerprint_service import (
    AudioFingerprintError,
    FingerprintBackendUnavailable,
    search_fingerprint,
)
from backend.services import cover_service, license_classifier_service
from backend.services.retrieval_service import search_recordings

logger = logging.getLogger("music_rights_ai")


class PipelineError(RuntimeError):
    """Lỗi có mã chuẩn hoá để tầng API map sang HTTP response (§12)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def process_music_query(audio_path: str, db: Session, vector_index=None,
                        top_k: int = None, mert_audio_path: str = None,
                        on_stage=None, cover_index=None) -> dict:
    """
    audio_path      : file dùng cho Chromaprint & Cover (nên là file gốc)
    mert_audio_path : file WAV 24kHz đã chuẩn hoá cho MERT (mặc định = audio_path)
    on_stage        : callback(tên_bước) — để client theo dõi tiến trình THẬT (§13)
    cover_index     : CoverIndex đã tải sẵn (nếu None sẽ lazy-load theo cấu hình)
    """
    top_k = top_k or config.TOP_K
    mert_audio_path = mert_audio_path or audio_path
    report = on_stage or (lambda _stage: None)

    thresholds = {
        "fingerprint": config.FP_THRESHOLD,
        "embedding": config.MERT_THRESHOLD,
        "cover": config.COVER_THRESHOLD,
    }
    timings = {}  # độ trễ từng bước, bắt buộc ghi log theo §12

    # ----------------------------------------------------------------------
    # TẦNG 1: CHROMAPRINT
    # ----------------------------------------------------------------------
    fingerprint_available = True
    report("FINGERPRINTING")
    t0 = time.perf_counter()
    # Tầng 1 không chạy vẫn phải ghi ngưỡng lẽ ra được áp và LÝ DO không chạy.
    # Điểm để None chứ không phải 0.0: "không so được" khác hẳn "so rồi, 0 điểm",
    # và hai nguyên nhân (thiếu CSDL / thiếu fpcalc) cần sửa theo hai cách khác nhau.
    try:
        if db is None:
            fingerprint_available = False
            fp_result = {
                "match_type": "UNAVAILABLE",
                "fingerprint_score": None,
                "threshold": config.FP_THRESHOLD,
                "reason_code": "DATABASE_UNAVAILABLE",
                "message": "Không kết nối được PostgreSQL nên không có fingerprint "
                           "tham chiếu để so.",
            }
        else:
            fp_result = search_fingerprint(db, audio_path)
    except FingerprintBackendUnavailable as e:
        # Không có fpcalc: ghi nhận rõ ràng rồi đi tiếp bằng MERT, KHÔNG im lặng.
        fingerprint_available = False
        fp_result = {
            "match_type": "UNAVAILABLE",
            "fingerprint_score": None,
            "threshold": config.FP_THRESHOLD,
            "reason_code": "FPCALC_MISSING",
            "message": str(e),
        }
    except AudioFingerprintError as e:
        # File qua được kiểm tra đuôi nhưng không giải mã được: trước đây lỗi này
        # thoát ra thành INTERNAL_ERROR, che mất nguyên nhân thật là file hỏng.
        logger.info("fpcalc khong doc duoc %s: %s", audio_path, e)
        raise PipelineError(
            "NO_AUDIO",
            "Không đọc được luồng âm thanh của file (file hỏng hoặc không chứa audio).",
        ) from e
    timings["fingerprint_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if fp_result.get("match_type") == "EXACT_MATCH":
        return {
            "pipeline_stage": "STAGE_1_CHROMAPRINT",
            "match_type": "EXACT_MATCH",
            "recording_id": fp_result["recording_id"],
            "score": fp_result["fingerprint_score"],
            "identity_confidence": fp_result["fingerprint_score"],
            "candidates": [],
            "evidence": {
                "fingerprint": fp_result,
                "embedding": None,
                "thresholds": thresholds,
                "timings_ms": timings,
                "decision_reason": (
                    f"Chromaprint đạt {fp_result['fingerprint_score']:.4f} "
                    f">= ngưỡng {config.FP_THRESHOLD} -> khớp chính xác."
                ),
            },
        }

    # ----------------------------------------------------------------------
    # TẦNG 2: MERT DEEP RETRIEVAL
    # ----------------------------------------------------------------------
    if vector_index is None or vector_index.ntotal == 0:
        raise PipelineError(
            "MODEL_FAILURE",
            "Chỉ mục vector MERT chưa sẵn sàng. Chạy scripts/rebuild_faiss_index.py "
            "rồi khởi động lại server.",
        )

    report("EMBEDDING")
    t1 = time.perf_counter()
    try:
        query_vector = extract_mert_embedding(
            mert_audio_path,
            target_sr=config.MERT_SAMPLE_RATE,
            max_duration=config.MERT_MAX_DURATION,
        )
    except EmbeddingAudioError as e:
        raise PipelineError("NO_AUDIO", "Không trích xuất được tín hiệu âm thanh từ file.") from e
    except EmbeddingModelError as e:
        # Lỗi phía MÁY CHỦ (nạp/chạy model), không phải lỗi của file người dùng —
        # trước đây cũng hiện ra là NO_AUDIO, tức đổ lỗi sai chỗ.
        logger.exception("MERT loi: %s", e)
        raise PipelineError("MODEL_FAILURE", "Model MERT không chạy được trên máy chủ.") from e
    timings["embedding_ms"] = round((time.perf_counter() - t1) * 1000, 2)

    report("VECTOR_SEARCH")
    t2 = time.perf_counter()
    candidates, search_stats = search_recordings(vector_index, query_vector, top_k=top_k)
    timings["vector_search_ms"] = round((time.perf_counter() - t2) * 1000, 2)

    embedding_evidence = {
        "top_k": candidates,
        "threshold": config.MERT_THRESHOLD,
        "model_version": vector_index.meta.get("model_version", config.MERT_MODEL_VERSION),
        "reference_vectors": vector_index.ntotal,
        "reference_recordings": vector_index.n_recordings,
        **(search_stats or {}),
    }

    best = candidates[0] if candidates else None
    best_score = best["similarity_score"] if best else 0.0

    if best is None or best_score < config.MERT_THRESHOLD:
        # ----------------------------------------------------------------------
        # TẦNG 3: COVER / VERSION IDENTIFICATION (CQT CHROMA + OTI)
        # ----------------------------------------------------------------------
        cover_evidence = None
        if config.COVER_ENABLED:
            report("COVER_SEARCH")
            t_cover = time.perf_counter()
            try:
                cover_res = cover_service.identify_cover(
                    audio_path=audio_path,
                    cover_index=cover_index,
                    candidate_recordings=candidates,
                    top_k=top_k,
                )
                timings["cover_ms"] = round((time.perf_counter() - t_cover) * 1000, 2)
                cover_index_used = (cover_index if cover_index is not None
                                    else cover_service.get_default_cover_index())
                cover_evidence = {
                    "matched": cover_res.get("matched", False),
                    "threshold": config.COVER_THRESHOLD,
                    # τCover phụ thuộc SỐ BÀI trong chỉ mục: cùng bộ truy vấn, τ = 0.70
                    # cho nhận nhầm 4,9% khi chỉ mục có 100 bài nhưng 17,4% khi có
                    # 24.375 bài (EXP-07). Không hiện con số này thì điểm tương đồng ở
                    # trên không đọc được đúng.
                    "reference_recordings": cover_index_used.n_items if cover_index_used else 0,
                    "top_candidate": cover_res.get("top_candidate"),
                    "candidates": cover_res.get("candidates", []),
                }

                if cover_res.get("matched") and cover_res.get("best_match"):
                    matched_cand = cover_res["best_match"]
                    matched_score = matched_cand["similarity_score"]
                    matched_oti = matched_cand["oti"]
                    matched_rec_id = matched_cand["recording_id"]

                    return {
                        "pipeline_stage": "STAGE_3_COVER",
                        "match_type": "COVER_MATCH",
                        "recording_id": matched_rec_id,
                        "score": matched_score,
                        "identity_confidence": matched_score,
                        "candidates": candidates,
                        "cover_candidates": cover_res.get("candidates", []),
                        "evidence": {
                            "fingerprint": fp_result,
                            "embedding": embedding_evidence,
                            "cover": cover_evidence,
                            "thresholds": thresholds,
                            "timings_ms": timings,
                            "decision_reason": (
                                f"Chromaprint và MERT dưới ngưỡng; Cover/OTI đạt {matched_score:.4f} "
                                f">= {config.COVER_THRESHOLD} (dịch cao độ {matched_oti} bán cung) "
                                f"-> nhận diện phiên bản/cover qua CQT chroma."
                            ),
                        },
                    }
            except Exception as e:
                # Nội dung lỗi có thể chứa đường dẫn trên máy chủ -> chỉ vào log
                logger.exception("Tang Cover loi: %s", e)
                cover_evidence = {"error": "COVER_STAGE_FAILED",
                                  "threshold": config.COVER_THRESHOLD}

        # KHÔNG ép thành NEAR_MATCH (§2): thiếu bằng chứng thì trả UNKNOWN.
        #
        # Đây cũng là chỗ duy nhất bộ phân loại giấy phép được gọi: bài không
        # khớp bản ghi nào thì không có quyền nào để tra, nên nếu muốn nói gì về
        # bản quyền thì chỉ còn cách suy đoán. EXP-09 đã đo và bộ này KHÔNG vượt
        # baseline ở đúng tình huống đó — vì vậy nó chỉ được phép sinh bằng
        # chứng, còn `match_type` vẫn là UNKNOWN và `identity_confidence` vẫn là
        # điểm MERT thật, không bị con số của classifier chạm vào.
        predicted_license = license_classifier_service.predict(query_vector)

        if fingerprint_available:
            reason = (f"Chromaprint {fp_result.get('fingerprint_score') or 0.0:.4f} "
                      f"< {fp_result.get('threshold', config.FP_THRESHOLD)}")
        else:
            reason = f"Chromaprint không chạy ({fp_result.get('reason_code')})"
        reason += f"; MERT cao nhất {best_score:.4f} < {config.MERT_THRESHOLD}"
        if cover_evidence and cover_evidence.get("top_candidate"):
            top_c = cover_evidence["top_candidate"]
            reason += f"; Cover đạt {top_c['similarity_score']:.4f} < {config.COVER_THRESHOLD}"
        reason += " -> không đủ bằng chứng để định danh."

        return {
            "pipeline_stage": "STAGE_2_MERT_RETRIEVAL" if not cover_evidence else "STAGE_3_COVER",
            "match_type": "UNKNOWN",
            "recording_id": None,
            "score": best_score,
            "identity_confidence": best_score,
            "condition": "HUMAN_REVIEW_REQUIRED",
            "candidates": candidates,
            "predicted_license": predicted_license,
            "evidence": {
                "fingerprint": fp_result,
                "embedding": embedding_evidence,
                "cover": cover_evidence,
                "thresholds": thresholds,
                "timings_ms": timings,
                "predicted_license": predicted_license,
                "decision_reason": reason,
            },
        }

    return {
        "pipeline_stage": "STAGE_2_MERT_RETRIEVAL",
        "match_type": "NEAR_MATCH",
        "recording_id": best["recording_id"],
        "score": best_score,
        "identity_confidence": best_score,
        "candidates": candidates,
        "evidence": {
            "fingerprint": fp_result,
            "embedding": embedding_evidence,
            "thresholds": thresholds,
            "timings_ms": timings,
            "decision_reason": (
                f"Chromaprint dưới ngưỡng; MERT similarity {best_score:.4f} >= "
                f"{config.MERT_THRESHOLD} -> khớp gần đúng ở mức bản ghi."
            ),
        },
    }
