"""
Ghép toàn bộ chuỗi xử lý thành một kết quả duy nhất:

    audio -> cascade (nhận diện) -> rights (tra cứu) -> Rule Engine (quyết định)
          -> response chuẩn §11 + evidence

Trước Giai đoạn 2, `decision_service` (Rule Engine) không hề được gọi ở đâu và
API tự sinh khuyến nghị bằng một chuỗi if/else riêng trong `rights_service`.
Nay chỉ có MỘT nơi ra quyết định.
"""
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
    if not ids:
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


def analyze_audio(audio_path: str, filename: str, db: Session,
                  vector_index, usage_context: dict, on_stage=None) -> dict:
    """
    Chạy trọn pipeline cho một file. Ném AudioProcessingError / PipelineError
    với mã lỗi chuẩn hoá để tầng API map sang HTTP response.

    `on_stage(tên_bước)` được gọi trước mỗi bước thật của pipeline, để màn hình
    Processing hiển thị đúng bước đang chạy chứ không phải một thanh chờ giả (§13).
    """
    start = time.time()
    report = on_stage or (lambda _stage: None)

    report("VALIDATING")
    audio_service.validate_upload(audio_path, filename)

    # Chromaprint đọc file gốc; MERT đọc bản đã tách/chuẩn hoá (nếu là video)
    report("EXTRACTING_AUDIO")
    mert_path, temp_to_cleanup = audio_service.prepare_for_embedding(audio_path, filename)

    try:
        cascade = process_music_query(
            audio_path=audio_path,
            db=db,
            vector_index=vector_index,
            top_k=config.TOP_K,
            mert_audio_path=mert_path,
            on_stage=report,
        )
    finally:
        if temp_to_cleanup:
            import os
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

    return {
        "identity": {
            "track": track_metadata.get("recording_title"),
            "artist": track_metadata.get("artist"),
            "recording_id": track_metadata.get("recording_id"),
            "composition_id": composition.get("composition_id"),
        },
        "match": {
            "type": cascade.get("match_type"),
            "confidence": round(float(cascade.get("identity_confidence") or 0.0), 4),
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
        },
        "recommendation": generate_rights_recommendation(decision),
        "evidence": {
            "identification": cascade.get("evidence", {}),
            "candidates": cascade.get("candidates", []),
            "rule_engine": decision["evidence"],
            "composition": composition,
            "rights_record": rights_data,
            "license_prediction": predicted_license,
            "production_features": production,
        },
        "latency_ms": round(latency_ms, 2),
        "model_version": config.MERT_MODEL_VERSION,
    }
