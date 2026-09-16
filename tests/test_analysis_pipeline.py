"""
Ứng viên trong evidence phải có TÊN BÀI, không chỉ UUID.

Vì sao đáng một bộ test riêng: "0.9348 với b83646be-a05b-…" là thứ không ai đối
chiếu được, tức là hộp đen — đúng thứ §2.4 cấm. Tính năng này lại nằm ở một hàm
phụ trợ dễ bị bỏ quên khi sửa `analyze_audio`, nên cần chốt bằng test.
"""
import pytest
from sqlalchemy import text

from backend.services.analysis_pipeline import _label_candidates
from tests.conftest import requires_db


def some_recording_ids(db_session, limit: int = 2) -> list:
    rows = db_session.execute(
        text("SELECT recording_id::text FROM recordings "
             "WHERE title IS NOT NULL LIMIT :n"),
        {"n": limit},
    ).fetchall()
    return [row[0] for row in rows]


@requires_db
def test_gan_ten_bai_va_nghe_si_cho_ung_vien(db_session):
    ids = some_recording_ids(db_session)
    if not ids:
        pytest.skip("Bảng recordings trống")

    candidates = [{"recording_id": rec_id, "similarity_score": 0.9} for rec_id in ids]
    _label_candidates(db_session, candidates)

    assert all("track" in c and "artist" in c for c in candidates)
    assert any(c["track"] for c in candidates), "không tra được tên bài nào"


@requires_db
def test_gan_nhan_cho_nhieu_danh_sach_trong_mot_lan(db_session):
    """Tầng 2 và tầng 3 được gắn nhãn trong CÙNG một truy vấn."""
    ids = some_recording_ids(db_session, 1)
    if not ids:
        pytest.skip("Bảng recordings trống")

    embedding_candidates = [{"recording_id": ids[0], "similarity_score": 0.93}]
    cover_candidates = [{"recording_id": ids[0], "similarity_score": 0.42, "oti": 3}]
    _label_candidates(db_session, embedding_candidates, None, cover_candidates)

    assert embedding_candidates[0]["track"] == cover_candidates[0]["track"]


@requires_db
def test_id_khong_ton_tai_tra_none_chu_khong_nem_loi(db_session):
    """Bản ghi biến mất khỏi CSDL không được làm hỏng cả lượt phân tích."""
    ghost = [{"recording_id": "00000000-0000-0000-0000-000000000000"}]
    _label_candidates(db_session, ghost)

    assert ghost[0]["track"] is None
    assert ghost[0]["artist"] is None


@requires_db
def test_ung_vien_thieu_recording_id_khong_bi_dung_toi(db_session):
    """Không có id thì không có gì để tra — và cũng không được ném lỗi."""
    odd = [{"similarity_score": 0.5}]
    _label_candidates(db_session, None, [], odd)

    assert "track" not in odd[0]
