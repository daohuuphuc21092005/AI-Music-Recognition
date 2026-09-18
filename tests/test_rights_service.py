"""Kiểm thử tra cứu bản quyền."""
import pytest
from sqlalchemy import text

from backend.services.rights_service import get_full_music_rights
from tests.conftest import requires_db


@requires_db
def test_get_full_music_rights(db_session):
    row = db_session.execute(
        text("SELECT recording_id FROM recordings LIMIT 1")
    ).fetchone()
    if not row:
        pytest.skip("Bang recordings trong. Chay: python init_db.py")

    result = get_full_music_rights(str(row[0]), db_session)

    assert result["status"] == "SUCCESS"
    assert "track_metadata" in result
    assert "rights_and_licensing" in result
    # Khuyến nghị KHÔNG còn sinh ở đây nữa: nó phải bắt nguồn từ quyết định của
    # Rule Engine, nếu không hai nơi sẽ nói hai đằng.
    assert "usage_recommendation" not in result

    if result["rights_found"]:
        rights = result["rights_and_licensing"]
        # Các trường Rule Engine cần phải có mặt
        for field in ("policy_action", "license_purchased", "revenue_share_agreed",
                      "recording_public_domain", "license_type"):
            assert field in rights
        # PD của tác phẩm nằm ở khối composition, tách khỏi PD của bản thu (§2)
        assert "public_domain_status" in result["track_metadata"]["composition"]


@requires_db
def test_id_khong_hop_le_tra_not_found(db_session):
    result = get_full_music_rights("khong-phai-uuid", db_session)
    assert result["status"] == "NOT_FOUND"


# --------------------------------------------------------------------------
# Câu khuyến nghị: hành động, không lặp lại lý do
# --------------------------------------------------------------------------
def _conditions_in_rules(node) -> set:
    """Mọi giá trị của khoá `condition` trong cây rules (bỏ qua chú thích)."""
    found = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "condition" and isinstance(value, str):
                found.add(value)
            else:
                found |= _conditions_in_rules(value)
    elif isinstance(node, list):
        for item in node:
            found |= _conditions_in_rules(item)
    return found


def test_moi_dieu_kien_trong_rules_deu_co_cau_hanh_dong():
    """Thiếu câu hành động thì khuyến nghị rơi về lặp lại lý do — thêm điều kiện
    mới vào rules_v1.yaml phải thêm câu vào ACTION_BY_CONDITION."""
    from backend.services.decision_service import load_rules
    from backend.services.rights_service import ACTION_BY_CONDITION

    conditions = _conditions_in_rules(load_rules())
    assert conditions, "không đọc được điều kiện nào từ rules"
    assert conditions - set(ACTION_BY_CONDITION) == set()


def test_khuyen_nghi_la_hanh_dong_khong_lap_lai_ly_do():
    from backend.services.rights_service import generate_rights_recommendation

    decision = {
        "risk_level": "UNKNOWN",
        "condition": "HUMAN_REVIEW_REQUIRED",
        "decision_reason": "Dữ liệu quyền không đủ tin cậy để kết luận, cần người kiểm "
                           "tra trước khi sử dụng.",
    }
    recommendation = generate_rights_recommendation(decision)

    assert recommendation == "⚪ CHƯA XÁC ĐỊNH. Cần người kiểm tra thủ công trước khi phát hành."
    assert decision["decision_reason"] not in recommendation


def test_khuyen_nghi_dung_ly_do_khi_dieu_kien_chua_co_cau_hanh_dong():
    from backend.services.rights_service import generate_rights_recommendation

    recommendation = generate_rights_recommendation(
        {"risk_level": "LOW", "condition": "MA_CHUA_CO", "decision_reason": "Lý do X."})

    assert recommendation == "🟢 RỦI RO THẤP. Lý do X."
