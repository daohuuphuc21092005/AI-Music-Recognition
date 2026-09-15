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
