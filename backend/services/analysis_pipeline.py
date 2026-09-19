"""
Ghép toàn bộ chuỗi xử lý thành một kết quả duy nhất:

    audio -> cascade (nhận diện) -> rights (tra cứu) -> Rule Engine (quyết định)
          -> response chuẩn §11 + evidence

Trước Giai đoạn 2, `decision_service` (Rule Engine) không hề được gọi ở đâu và
API tự sinh khuyến nghị bằng một chuỗi if/else riêng trong `rights_service`.
Nay chỉ có MỘT nơi ra quyết định.
"""
import logging
import os
import time

from sqlalchemy.orm import Session

from backend import config
from backend.services import audio_service, production_features_service
from backend.services.cascade_service import process_music_query
from backend.services.decision_service import evaluate_rights_and_risk
from backend.services.license_mapping import rights_for_license_type
from backend.services.rights_service import (
    generate_rights_recommendation,
    get_full_music_rights,
)
from backend.services.window_scan import get_audio_duration, plan_windows, slice_audio

logger = logging.getLogger("music_rights_ai")


def _label_candidates(db: Session, *candidate_lists) -> None:
    """
    Gắn `track` + `artist` vào từng ứng viên (sửa tại chỗ).

    Evidence chỉ có UUID thì người đọc không kiểm chứng được gì: "0.9348 với
    b83646be-a05b-…" không cho biết đó là bài nào của ai. Tra một lần cho toàn bộ
    ứng viên của cả tầng 2 lẫn tầng 3; tra hỏng thì bỏ qua, vì thiếu tên trong
    evidence vẫn tốt hơn là làm hỏng cả lượt phân tích.
    """
    from sqlalchemy import bindparam, text

    ids = {str(c["recording_id"]) for lst in candidate_lists for c in (lst or [])
           if isinstance(c, dict) and c.get("recording_id")}
    if not ids or db is None:
        return
    try:
        # recording_id là UUID trong CSDL còn ứng viên mang chuỗi -> ép về text,
        # nếu không PostgreSQL báo "operator does not exist: uuid = text".
        rows = db.execute(
            text("SELECT recording_id::text, title, artist FROM recordings "
                 "WHERE recording_id::text IN :ids")
            .bindparams(bindparam("ids", expanding=True)),
            {"ids": sorted(ids)},
        ).fetchall()
    except Exception:
        return

    by_id = {row[0]: (row[1], row[2]) for row in rows}
    for lst in candidate_lists:
        for candidate in (lst or []):
            if isinstance(candidate, dict):
                title, artist = by_id.get(str(candidate.get("recording_id")), (None, None))
                candidate["track"] = title
                candidate["artist"] = artist


def analyze_audio(audio_path: str, filename: str = None, db: Session = None,
                  vector_index=None, usage_context: dict = None, on_stage=None,
                  exclude_recording_ids=None) -> dict:
    """
    Chạy trọn pipeline cho một file. Ném AudioProcessingError / PipelineError
    với mã lỗi chuẩn hoá để tầng API map sang HTTP response.

    `on_stage(tên_bước)` được gọi trước mỗi bước thật của pipeline, để màn hình
    Processing hiển thị đúng bước đang chạy chứ không phải một thanh chờ giả (§13).
    """
    start = time.time()
    report = on_stage or (lambda _stage: None)
    filename = filename or os.path.basename(audio_path)
    usage_context = usage_context or {"platform": "OTHER", "commercial_use": False, "monetization": False}

    report("VALIDATING")
    audio_service.validate_upload(audio_path, filename)

    # Chromaprint đọc file gốc; MERT đọc bản đã tách/chuẩn hoá (nếu là video)
    report("EXTRACTING_AUDIO")
    mert_path, temp_to_cleanup = audio_service.prepare_for_embedding(audio_path, filename)

    source_for_scan = mert_path if (mert_path and os.path.exists(mert_path)) else audio_path
    total_duration = get_audio_duration(source_for_scan)

    try:
        # File <= 30s hoặc SCAN_MODE=first: kết quả Y HỆT trước đây (cascade nhận file gốc)
        if total_duration <= 30.0 or getattr(config, "SCAN_MODE", "multi") == "first":
            cascade = process_music_query(
                audio_path=audio_path,
                db=db,
                vector_index=vector_index,
                top_k=config.TOP_K,
                mert_audio_path=mert_path,
                on_stage=report,
                exclude_recording_ids=exclude_recording_ids,
            )
            if "evidence" not in cascade:
                cascade["evidence"] = {}
            score_val = round(float(cascade.get("identity_confidence") or cascade.get("score") or 0.0), 4)
            cascade["evidence"]["windows"] = [{
                "start_s": 0.0,
                "end_s": round(total_duration, 2) if total_duration > 0 else 30.0,
                "stage": cascade.get("pipeline_stage"),
                "match_type": cascade.get("match_type"),
                "score": score_val,
            }]
            cascade["evidence"]["windows_scanned"] = 1
            cascade["evidence"]["stopped_early"] = False
        else:
            window_starts = plan_windows(
                total_duration, window_s=30.0, max_windows=config.SCAN_MAX_WINDOWS
            )
            if len(window_starts) <= 1:
                cascade = process_music_query(
                    audio_path=audio_path,
                    db=db,
                    vector_index=vector_index,
                    top_k=config.TOP_K,
                    mert_audio_path=mert_path,
                    on_stage=report,
                    exclude_recording_ids=exclude_recording_ids,
                )
                if "evidence" not in cascade:
                    cascade["evidence"] = {}
                score_val = round(float(cascade.get("identity_confidence") or cascade.get("score") or 0.0), 4)
                cascade["evidence"]["windows"] = [{
                    "start_s": 0.0,
                    "end_s": round(total_duration, 2),
                    "stage": cascade.get("pipeline_stage"),
                    "match_type": cascade.get("match_type"),
                    "score": score_val,
                }]
                cascade["evidence"]["windows_scanned"] = 1
                cascade["evidence"]["stopped_early"] = False
            else:
                scan_budget = getattr(config, "SCAN_TIME_BUDGET_S", 60.0)
                scan_start = time.time()
                window_records = []
                window_results = []
                windows_skipped = 0

                for idx, start_s in enumerate(window_starts):
                    # Kiểm tra ngân sách thời gian
                    if idx > 0 and (time.time() - scan_start) >= scan_budget:
                        windows_skipped = len(window_starts) - idx
                        logger.info(
                            "Het ngan sach thoi gian quet cua so (%.1fs). Bo qua %s cua so con lai.",
                            scan_budget, windows_skipped,
                        )
                        break

                    # Cắt lát 30s
                    slice_path = slice_audio(source_for_scan, start_s=start_s, duration_s=30.0)
                    try:
                        slice_res = process_music_query(
                            audio_path=slice_path,
                            db=db,
                            vector_index=vector_index,
                            top_k=config.TOP_K,
                            mert_audio_path=slice_path,
                            on_stage=report,
                            exclude_recording_ids=exclude_recording_ids,
                        )
                    finally:
                        if slice_path and os.path.exists(slice_path):
                            try:
                                os.remove(slice_path)
                            except OSError:
                                pass

                    score_val = round(
                        float(slice_res.get("identity_confidence") or slice_res.get("score") or 0.0), 4
                    )
                    end_s = round(min(start_s + 30.0, total_duration), 2)
                    record = {
                        "start_s": round(float(start_s), 2),
                        "end_s": end_s,
                        "stage": slice_res.get("pipeline_stage"),
                        "match_type": slice_res.get("match_type"),
                        "score": score_val,
                    }
                    window_records.append(record)
                    window_results.append(slice_res)

                    # Dừng sớm khi gặp EXACT_MATCH
                    if slice_res.get("match_type") == "EXACT_MATCH":
                        logger.info("Dung som tai cua so start=%.1fs (EXACT_MATCH)", start_s)
                        break

                # Tổng hợp kết quả:
                # 1. Ưu tiên các cửa sổ khớp
                matched_candidates = [
                    (res, rec) for res, rec in zip(window_results, window_records)
                    if res.get("match_type") in ("EXACT_MATCH", "COVER_MATCH", "NEAR_MATCH")
                ]

                priority = {"EXACT_MATCH": 3, "COVER_MATCH": 2, "NEAR_MATCH": 1}

                if matched_candidates:
                    # Chọn cửa sổ có identity_confidence cao nhất; hoà thì EXACT > COVER > NEAR
                    best_candidate = max(
                        matched_candidates,
                        key=lambda item: (
                            float(item[0].get("identity_confidence") or 0.0),
                            priority.get(item[0].get("match_type"), 0),
                        ),
                    )
                    cascade = best_candidate[0]
                else:
                    # Không cửa sổ nào khớp -> UNKNOWN với best_score cao nhất
                    best_unknown = max(
                        zip(window_results, window_records),
                        key=lambda item: float(
                            item[0].get("score") or item[0].get("identity_confidence") or 0.0
                        ),
                    )
                    cascade = best_unknown[0]

                if "evidence" not in cascade:
                    cascade["evidence"] = {}
                stopped_early = any(r.get("match_type") == "EXACT_MATCH" for r in window_records)
                cascade["evidence"]["windows"] = window_records
                cascade["evidence"]["windows_scanned"] = len(window_records)
                cascade["evidence"]["stopped_early"] = stopped_early
                if windows_skipped > 0:
                    cascade["evidence"]["windows_skipped"] = windows_skipped
    finally:
        if temp_to_cleanup:
            if os.path.exists(temp_to_cleanup):
                try:
                    os.remove(temp_to_cleanup)
                except OSError:
                    pass

    # --- Tra cứu quyền (chỉ khi đã định danh được bản ghi) -------------------
    report("RIGHTS_LOOKUP")
    recording_id = cascade.get("recording_id")
    rights_lookup = get_full_music_rights(str(recording_id), db) if recording_id else None
    rights_data = (rights_lookup or {}).get("rights_and_licensing")
    track_metadata = (rights_lookup or {}).get("track_metadata") or {}
    composition = track_metadata.get("composition") or {}

    # --- Quyền SUY ĐOÁN, chỉ khi không tra được quyền thật -------------------
    # Bài không khớp bản ghi nào thì không có gì để tra, nên nếu muốn nói được
    # điều gì về bản quyền thì chỉ còn cách suy đoán từ âm thanh. Bộ phân loại
    # được tích hợp theo yêu cầu của chủ dự án, sau khi EXP-09 đo và cho thấy nó
    # KHÔNG vượt baseline ở đúng tình huống này (0.382 so với 0.569).
    #
    # Hai chốt an toàn được giữ nguyên, và chúng mới là thứ quyết định:
    #   1. `source` mang tiền tố PREDICTED -> compute_rights_confidence trừ 0.60,
    #      đủ để kéo rights_confidence xuống dưới ngưỡng, nên Rule Engine sẽ ra
    #      UNKNOWN + HUMAN_REVIEW_REQUIRED thay vì một kết luận chắc nịch.
    #   2. `identity_confidence` vẫn là điểm MERT thật, không bị xác suất của
    #      classifier chạm vào — §2 cấm gộp hai loại tin cậy làm một.
    predicted_license = cascade.get("predicted_license")
    rights_predicted = False
    if not rights_data and predicted_license:
        flags = rights_for_license_type(predicted_license["predicted_license_type"])
        if flags.get("license_type") != "UNKNOWN":
            rights_predicted = True
            rights_data = {
                **flags,
                "source": (
                    f"PREDICTED bởi {predicted_license['model_version']} "
                    f"(xác suất {predicted_license['probability']}) — "
                    f"KHÔNG tra từ nguồn nào"
                ),
                "source_url": None,
                "verified_at": None,
                "revenue_share_required": False,
                "policy_action": "NONE",
                "license_purchased": False,
                "revenue_share_agreed": False,
            }

    # --- Đặc trưng sản xuất (chỉ để làm bằng chứng) -------------------------
    # Đo dấu vết mastering từ chính tín hiệu. Đây là thứ THẬT SỰ nằm trong âm
    # thanh, khác hẳn giấy phép (EXP-09). Nhưng nó KHÔNG được chạm vào mức rủi
    # ro: nhạc Creative Commons cũng master chuyên nghiệp, còn nhiều bản thu
    # thương mại lại cố ý giữ dải động rộng. Suy từ đây ra quyền sử dụng chính
    # là loại suy diễn §2 cấm. Vì vậy nó chỉ đi vào evidence.
    production = None
    try:
        import os as _os

        import librosa
        # `mert_path` có thể là file tạm ĐÃ BỊ XOÁ ở khối finally phía trên (khi
        # đầu vào là video). `mert_path or audio_path` không cứu được vì chuỗi
        # vẫn khác rỗng — phải kiểm tra file có thật trên đĩa.
        source = mert_path if (mert_path and _os.path.exists(mert_path)) else audio_path
        signal, signal_sr = librosa.load(source,
                                         sr=config.MERT_SAMPLE_RATE, mono=True,
                                         duration=config.MERT_MAX_DURATION)
        production = production_features_service.analyze(signal, signal_sr)
    except Exception:
        # Không đo được thì bỏ qua: đây là thông tin phụ, không đáng làm hỏng
        # cả một lần phân tích.
        production = None

    # --- Rule Engine --------------------------------------------------------
    report("RULE_ENGINE")
    # Quyền suy đoán đi qua một match_type RIÊNG. Không đổi `identity_confidence`
    # thành xác suất của classifier: §2 cấm gộp hai loại tin cậy, và điểm MERT
    # thật vẫn là thứ mô tả đúng "nhận diện được tới đâu". Cái bị hạ là
    # rights_confidence, và decision_confidence = min(hai giá trị) nên vẫn thấp.
    match_type = cascade.get("match_type")
    if rights_predicted:
        match_type = "LICENSE_PREDICTED"

    decision = evaluate_rights_and_risk(
        rights_data=rights_data,
        match_info={
            "match_type": match_type,
            "identity_confidence": cascade.get("identity_confidence", 0.0),
        },
        usage_context=usage_context,
    )

    # --- Đánh giá ngữ cảnh xấu nhất (Worst-Case Context) --------------------
    # Ngữ cảnh xấu nhất: cùng nền tảng nhưng dùng thương mại và bật kiếm tiền
    worst_case_context = {
        "platform": usage_context.get("platform", "YOUTUBE"),
        "commercial_use": True,
        "monetization": True,
    }
    is_already_worst = (
        bool(usage_context.get("commercial_use")) is True
        and bool(usage_context.get("monetization")) is True
    )

    worst_case = None
    if not is_already_worst:
        worst_decision = evaluate_rights_and_risk(
            rights_data=rights_data,
            match_info={
                "match_type": match_type,
                "identity_confidence": cascade.get("identity_confidence", 0.0),
            },
            usage_context=worst_case_context,
        )
        if (worst_decision["risk_level"] != decision["risk_level"]
                or worst_decision["condition"] != decision["condition"]):
            worst_case = {
                "risk": worst_decision["risk_level"],
                "condition": worst_decision["condition"],
                "rule_id": worst_decision["evidence"].get("rule_id"),
            }

    # --- Ghi chú nguồn gốc giấy phép (Rights Source Note) --------------------
    source_val = (rights_data or {}).get("source") or ""
    if not source_val:
        rights_source_note = None
    elif source_val == "fma_metadata":
        rights_source_note = "Dữ liệu khai báo bởi nghệ sĩ trên Free Music Archive (không được xác minh độc lập)"
    elif source_val == "jamendo_api":
        rights_source_note = "Dữ liệu giấy phép từ Jamendo Licensing API"
    elif source_val == "simulated":
        rights_source_note = "Dữ liệu thử nghiệm nội bộ (giả lập)"
    else:
        rights_source_note = f"Dữ liệu từ nguồn: {source_val}"

    # Gắn tên bài cho ứng viên của tầng 2 và tầng 3 trước khi đóng gói evidence.
    # `top_candidate` của tầng Cover trỏ vào chính phần tử đầu của `candidates`
    # nên được gắn theo, không cần xử lý riêng.
    cascade_evidence = cascade.get("evidence") or {}
    _label_candidates(
        db,
        cascade.get("candidates"),
        (cascade_evidence.get("embedding") or {}).get("top_k"),
        (cascade_evidence.get("cover") or {}).get("candidates"),
    )

    latency_ms = (time.time() - start) * 1000

    evidence_dict = {
        "identification": cascade.get("evidence", {}),
        "candidates": cascade.get("candidates", []),
        "rule_engine": decision["evidence"],
        "composition": composition,
        "rights_record": rights_data,
        "license_prediction": predicted_license,
        "production_features": production,
        "windows": cascade_evidence.get("windows", []),
        "windows_scanned": cascade_evidence.get("windows_scanned", len(cascade_evidence.get("windows", []))),
        "stopped_early": cascade_evidence.get("stopped_early", False),
    }
    if "windows_skipped" in cascade_evidence:
        evidence_dict["windows_skipped"] = cascade_evidence["windows_skipped"]

    return {
        "identity": {
            "track": track_metadata.get("recording_title"),
            "artist": track_metadata.get("artist"),
            "recording_id": track_metadata.get("recording_id") or cascade.get("recording_id"),
            "composition_id": composition.get("composition_id"),
        },
        "match": {
            "type": cascade.get("match_type"),
            "confidence": round(float(cascade.get("identity_confidence") or cascade.get("score") or 0.0), 4),
            "pipeline_stage": cascade.get("pipeline_stage"),
        },
        "rights": {
            "license": (rights_data or {}).get("license_type"),
            "copyright_status": (rights_data or {}).get("copyright_status"),
            "attribution_required": (rights_data or {}).get("attribution_required"),
            "commercial_use_allowed": (rights_data or {}).get("commercial_use_allowed"),
            "monetization_allowed": (rights_data or {}).get("monetization_allowed"),
            "source": (rights_data or {}).get("source"),
            "source_url": (rights_data or {}).get("source_url"),
            "verified_at": (rights_data or {}).get("verified_at"),
            "rights_found": bool(rights_data),
            # Phân biệt rạch ròi quyền TRA ĐƯỢC với quyền SUY ĐOÁN. Thiếu cờ này
            # thì giao diện sẽ hiển thị hai thứ khác hẳn nhau y như nhau.
            "predicted": rights_predicted,
        },
        "assessment": {
            "risk": decision["risk_level"],
            "category": decision["category"],
            "condition": decision["condition"],
            "reason": decision["decision_reason"],
            "identity_confidence": decision["identity_confidence"],
            "rights_confidence": decision["rights_confidence"],
            "decision_confidence": decision["decision_confidence"],
            "worst_case": worst_case,
            "context_declared_by_user": True,
            "rights_source_note": rights_source_note,
        },
        "recommendation": generate_rights_recommendation(decision),
        "evidence": evidence_dict,
        "latency_ms": round(latency_ms, 2),
        "model_version": config.MERT_MODEL_VERSION,
    }
