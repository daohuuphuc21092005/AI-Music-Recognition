"""
Tra cứu quyền sử dụng của một bản ghi.

Nhiệm vụ của service này là LẤY DỮ LIỆU, không phải ra quyết định. Việc phân
loại 5 nhóm và đánh giá rủi ro thuộc về `decision_service` (Rule Engine).

Thay đổi so với bản cũ:
  - Trả thêm các trường Rule Engine cần: policy_action, license_purchased,
    revenue_share_agreed, recording_public_domain.
  - Trả `composition` như một khối riêng để PD của tác phẩm và PD của bản thu
    được xét ĐỘC LẬP (§2).
  - Không còn tự bịa giá trị mặc định khi thiếu rights. Bản cũ mặc định
    `copyright_status = "PROTECTED"` và `attribution_required = True` — tức là
    dựng ra một giấy phép không hề tồn tại rồi kết luận trên đó.
"""
import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session

RIGHTS_FIELDS = (
    "rights_id", "license_type", "copyright_status",
    "attribution_required", "commercial_use_allowed", "modification_allowed",
    "monetization_allowed", "revenue_share_required",
    "policy_action", "license_purchased", "revenue_share_agreed",
    "recording_public_domain",
    "territory", "platform", "valid_from", "valid_until",
    "source", "source_url", "verified_at",
)


def is_valid_uuid(val) -> bool:
    try:
        uuid.UUID(str(val))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def get_recording_with_composition(recording_id: str, db: Session):
    query = text("""
        SELECT r.recording_id, r.title AS recording_title, r.artist, r.album,
               r.release_year, r.duration, r.source_dataset, r.audio_path,
               c.composition_id, c.title AS composition_title, c.composer,
               c.year AS composition_year, c.public_domain_status
        FROM recordings r
        LEFT JOIN compositions c ON r.composition_id = c.composition_id
        WHERE r.recording_id = :rec_id
    """)
    return db.execute(query, {"rec_id": recording_id}).fetchone()


def get_rights_row(recording_id: str, composition_id: str, db: Session):
    """
    Ưu tiên quyền gắn với BẢN THU; chỉ khi không có mới xét quyền ở mức tác phẩm.
    Bản cũ dùng `WHERE recording_id = :x OR composition_id = :y LIMIT 1` nên có
    thể trả về quyền của một bản thu KHÁC cùng tác phẩm.
    """
    query = text(f"""
        SELECT {', '.join(RIGHTS_FIELDS)}
        FROM rights
        WHERE recording_id = :rec_id
        ORDER BY verified_at DESC NULLS LAST
        LIMIT 1
    """)
    row = db.execute(query, {"rec_id": recording_id}).fetchone()
    if row is not None:
        return row, "recording"

    if composition_id and composition_id != "None":
        query = text(f"""
            SELECT {', '.join(RIGHTS_FIELDS)}
            FROM rights
            WHERE composition_id = :comp_id AND recording_id IS NULL
            ORDER BY verified_at DESC NULLS LAST
            LIMIT 1
        """)
        row = db.execute(query, {"comp_id": composition_id}).fetchone()
        if row is not None:
            return row, "composition"

    return None, None


def get_full_music_rights(recording_id: str, db: Session) -> dict:
    if not recording_id or not is_valid_uuid(recording_id):
        return {
            "status": "NOT_FOUND",
            "message": f"ID không hợp lệ hoặc không đúng định dạng UUID: {recording_id}",
        }

    recording = get_recording_with_composition(recording_id, db)
    if not recording:
        return {
            "status": "NOT_FOUND",
            "message": f"Không tìm thấy bản ghi với ID {recording_id}",
        }

    composition_id = str(recording.composition_id) if recording.composition_id else None
    rights_row, rights_scope = get_rights_row(recording_id, composition_id, db)

    composition = {
        "composition_id": composition_id,
        "title": recording.composition_title,
        "composer": recording.composer,
        "year": recording.composition_year,
        # PD của TÁC PHẨM. Không dùng để suy ra PD của bản thu (§2).
        "public_domain_status": recording.public_domain_status,
    }

    if rights_row is None:
        # Thiếu dữ liệu quyền là một sự kiện có ý nghĩa, không được che lấp bằng
        # giá trị mặc định. Rule Engine sẽ trả UNKNOWN + HUMAN_REVIEW_REQUIRED.
        return {
            "status": "SUCCESS",
            "rights_found": False,
            "track_metadata": {
                "recording_id": str(recording.recording_id),
                "recording_title": recording.recording_title,
                "artist": recording.artist,
                "album": recording.album,
                "release_year": recording.release_year,
                "duration": recording.duration,
                "source_dataset": recording.source_dataset,
                "composition": composition,
            },
            "rights_and_licensing": None,
            "message": "Không tìm thấy dữ liệu quyền cho bản ghi này.",
        }

    rights = {field: getattr(rights_row, field) for field in RIGHTS_FIELDS}
    rights["rights_id"] = str(rights["rights_id"]) if rights["rights_id"] else None
    for date_field in ("valid_from", "valid_until", "verified_at"):
        if rights.get(date_field) is not None:
            rights[date_field] = str(rights[date_field])
    rights["rights_scope"] = rights_scope
    rights["composition"] = composition

    return {
        "status": "SUCCESS",
        "rights_found": True,
        "track_metadata": {
            "recording_id": str(recording.recording_id),
            "recording_title": recording.recording_title,
            "artist": recording.artist,
            "album": recording.album,
            "release_year": recording.release_year,
            "duration": recording.duration,
            "source_dataset": recording.source_dataset,
            "composition": composition,
        },
        "rights_and_licensing": rights,
    }


# Câu hành động cho từng mã `condition` của Rule Engine. Thêm điều kiện mới vào
# configs/rules_v1.yaml thì thêm câu ở đây — tests/test_rights_service.py kiểm đủ.
ACTION_BY_CONDITION = {
    "ATTRIBUTION_REQUIRED": "Hãy ghi công tác giả trong phần mô tả.",
    "REVENUE_SHARE_APPLIED": "Doanh thu sẽ bị chia sẻ theo thoả thuận.",
    "REVENUE_REDIRECTED": "Doanh thu sẽ chuyển cho chủ sở hữu bản quyền.",
    "VIDEO_BLOCKED_OR_STRIKE": "Nên thay bằng bản nhạc khác.",
    "NON_COMMERCIAL_VIOLATION": "Hãy chuyển sang mục đích phi thương mại hoặc đổi nhạc.",
    "MONETIZATION_NOT_PERMITTED": "Hãy tắt kiếm tiền hoặc đổi nhạc.",
    "RECORDING_PERMISSION_REQUIRED": "Cần xin phép chủ bản thu, hoặc dùng bản thu khác thuộc miền công cộng.",
    "COMPOSITION_PERMISSION_REQUIRED": "Cần xử lý quyền của phần sáng tác.",
    "LICENSE_REQUIRED": "Cần mua giấy phép trước khi sử dụng.",
    "HUMAN_REVIEW_REQUIRED": "Cần người kiểm tra thủ công trước khi phát hành.",
    "MONETIZABLE_UNTIL_RETROACTIVE_CLAIM": "Có thể kiếm tiền, nhưng vẫn có rủi ro bị khiếu nại hồi tố.",
    "FREE_TO_USE": "Được sử dụng tự do.",
    "FULL_REVENUE_RETAINED": "Bạn giữ toàn bộ doanh thu.",
}


def generate_rights_recommendation(decision: dict) -> str:
    """
    Câu khuyến nghị bằng ngôn ngữ tự nhiên, DỰA TRÊN quyết định của Rule Engine.

    Bản cũ tự suy luận lại từ đầu bằng một chuỗi if/else riêng, nên có thể mâu
    thuẫn với Rule Engine. Nay chỉ diễn giải lại quyết định đã có.
    """
    risk = decision.get("risk_level", "UNKNOWN")
    condition = decision.get("condition", "")
    reason = decision.get("decision_reason", "")

    prefix = {
        "LOW": "🟢 RỦI RO THẤP.",
        "CONDITIONAL": "🟡 CÓ ĐIỀU KIỆN.",
        "HIGH": "🔴 RỦI RO CAO.",
        "UNKNOWN": "⚪ CHƯA XÁC ĐỊNH.",
    }.get(risk, "⚪ CHƯA XÁC ĐỊNH.")

    # Khuyến nghị = mức rủi ro + HÀNH ĐỘNG (schema: "khuyến nghị hành động cụ thể").
    # Lý do nằm riêng ở assessment.reason và màn Result in nó ngay bên dưới; ghép cả
    # hai thì câu lặp ý ("...cần người kiểm tra trước khi sử dụng. Cần người kiểm tra
    # thủ công trước khi phát hành.") và màn hình in lý do hai lần. Điều kiện chưa có
    # câu hành động thì mới dùng lý do, để câu không cụt còn mỗi mức rủi ro.
    action = ACTION_BY_CONDITION.get(condition, "")
    return " ".join(part for part in (prefix, action or reason) if part)
