"""Kiểm thử phễu cascade trên index thật."""
import pytest

from backend import config
from backend.services.cascade_service import PipelineError, process_music_query
from tests.conftest import requires_db


@requires_db
def test_full_cascade_pipeline(db_session, vector_index, test_audio):
    result = process_music_query(test_audio, db_session, vector_index, top_k=3)

    assert result["pipeline_stage"] in ("STAGE_1_CHROMAPRINT", "STAGE_2_MERT_RETRIEVAL", "STAGE_3_COVER")
    assert result["match_type"] in ("EXACT_MATCH", "NEAR_MATCH", "COVER_MATCH", "UNKNOWN")
    assert "candidates" in result

    evidence = result["evidence"]
    assert evidence["thresholds"]["fingerprint"] == config.FP_THRESHOLD
    assert evidence["thresholds"]["embedding"] == config.MERT_THRESHOLD
    assert evidence["decision_reason"]
    assert "timings_ms" in evidence


@requires_db
def test_top_k_la_cac_ban_ghi_khac_nhau(db_session, vector_index, test_audio):
    result = process_music_query(test_audio, db_session, vector_index, top_k=5)
    if result["pipeline_stage"] != "STAGE_2_MERT_RETRIEVAL":
        pytest.skip("Truy van khop ngay o tang 1 nen khong co Top-K")

    ids = [c["recording_id"] for c in result["candidates"]]
    assert len(ids) == len(set(ids)), "Top-K phai la cac ban ghi KHAC NHAU"


@requires_db
def test_duoi_nguong_thi_khong_dinh_danh(db_session, vector_index, test_audio):
    """Dưới ngưỡng phải là UNKNOWN, tuyệt đối không ép thành NEAR_MATCH (§2)."""
    result = process_music_query(test_audio, db_session, vector_index, top_k=5)

    if result["match_type"] == "UNKNOWN":
        assert result["recording_id"] is None
        assert result["condition"] == "HUMAN_REVIEW_REQUIRED"
        assert result["score"] < config.MERT_THRESHOLD
    elif result["match_type"] == "NEAR_MATCH":
        assert result["score"] >= config.MERT_THRESHOLD


@requires_db
def test_thieu_index_thi_bao_loi_thay_vi_bia_ket_qua(db_session, unmatched_audio):
    """Không có index thì phải MODEL_FAILURE, không được trả kết quả ngẫu nhiên."""
    with pytest.raises(PipelineError) as exc:
        process_music_query(unmatched_audio, db_session, vector_index=None)
    assert exc.value.code == "MODEL_FAILURE"


@requires_db
def test_khong_khop_tang_1_thi_xuong_tang_2(db_session, vector_index, unmatched_audio):
    """Nhiễu trắng phải trượt Chromaprint rồi được tầng 2 xử lý."""
    result = process_music_query(unmatched_audio, db_session, vector_index, top_k=5)

    assert result["pipeline_stage"] == "STAGE_2_MERT_RETRIEVAL"
    assert result["match_type"] in ("NEAR_MATCH", "UNKNOWN")
    evidence = result["evidence"]
    fingerprint = evidence["fingerprint"]

    # So với ngưỡng HIỆU DỤNG, không phải τFP cơ sở. Fixture là nhiễu trắng 8
    # giây, mà điểm nền của một đoạn ngắn cao hơn hẳn đoạn 30 giây (đo được:
    # 0.2558 so với 0.0814), nên `search_fingerprint` tự nâng ngưỡng theo độ dài.
    # Chính con số 0.1628 của đoạn 8 giây này VƯỢT τFP cơ sở 0.15 — nếu chỉ dùng
    # một ngưỡng cố định thì nhiễu trắng đã bị gán EXACT_MATCH.
    assert fingerprint["fingerprint_score"] < fingerprint["threshold"]
    assert fingerprint["threshold"] >= config.FP_THRESHOLD
    assert "embedding_ms" in evidence["timings_ms"]


@requires_db
def test_nguong_duoc_nang_cho_truy_van_ngan(db_session, vector_index, unmatched_audio):
    """
    Truy vấn ngắn phải được áp ngưỡng CAO HƠN τFP cơ sở.

    Đây là chốt chặn cho một dương tính giả có thật: đoạn 8 giây bất kỳ, kể cả
    nhiễu trắng thuần, đạt điểm Chromaprint quanh 0.16 — vượt τFP = 0.15 hiệu
    chỉnh trên tập truy vấn 30 giây.
    """
    result = process_music_query(unmatched_audio, db_session, vector_index, top_k=5)
    fingerprint = result["evidence"]["fingerprint"]

    assert fingerprint["threshold_raised_for_short_query"] is True
    assert fingerprint["threshold"] > config.FP_THRESHOLD
    assert fingerprint["threshold_base"] == config.FP_THRESHOLD


def test_cascade_stage3_cover_match_unit(monkeypatch):
    """Giả lập query trượt Chromaprint & MERT, nhưng đạt ngưỡng Cover -> STAGE_3_COVER."""
    from unittest.mock import MagicMock
    from backend.services import cascade_service

    monkeypatch.setattr(cascade_service, "search_fingerprint",
                        lambda db, path: {"match_type": "NOT_FOUND", "fingerprint_score": 0.02})
    monkeypatch.setattr(cascade_service, "extract_mert_embedding",
                        lambda path, **kw: [0.1] * 768)

    mock_index = MagicMock()
    mock_index.ntotal = 100
    mock_index.n_recordings = 50
    mock_index.meta = {"model_version": "v1"}

    # MERT similarity = 0.91 < 0.97 (trượt MERT)
    monkeypatch.setattr(
        cascade_service,
        "search_recordings",
        lambda idx, vec, top_k: ([{"recording_id": "rec_candidate_1", "similarity_score": 0.91}], {}),
    )

    # Cover similarity = 0.985 >= 0.97 (khớp Cover)
    monkeypatch.setattr(
        cascade_service.cover_service,
        "identify_cover",
        lambda **kw: {
            "matched": True,
            "best_match": {
                "recording_id": "rec_cover_winner",
                "similarity_score": 0.985,
                "oti": 3,
            },
            "candidates": [{"recording_id": "rec_cover_winner", "similarity_score": 0.985, "oti": 3}],
            "top_candidate": {"recording_id": "rec_cover_winner", "similarity_score": 0.985, "oti": 3},
            "threshold": 0.97,
        },
    )

    result = process_music_query("mock_audio.wav", db=None, vector_index=mock_index)

    assert result["pipeline_stage"] == "STAGE_3_COVER"
    assert result["match_type"] == "COVER_MATCH"
    assert result["recording_id"] == "rec_cover_winner"
    assert result["score"] == 0.985
    assert result["identity_confidence"] == 0.985
    assert result["evidence"]["cover"]["matched"] is True
    assert "dịch cao độ 3 bán cung" in result["evidence"]["decision_reason"]


def test_cascade_stage3_cover_below_threshold_returns_unknown(monkeypatch):
    """Khi cả 3 tầng đều dưới ngưỡng -> trả UNKNOWN kèm HUMAN_REVIEW_REQUIRED."""
    from unittest.mock import MagicMock
    from backend.services import cascade_service

    monkeypatch.setattr(cascade_service, "search_fingerprint",
                        lambda db, path: {"match_type": "NOT_FOUND", "fingerprint_score": 0.02})
    monkeypatch.setattr(cascade_service, "extract_mert_embedding",
                        lambda path, **kw: [0.1] * 768)

    mock_index = MagicMock()
    mock_index.ntotal = 100
    mock_index.n_recordings = 50
    mock_index.meta = {"model_version": "v1"}

    monkeypatch.setattr(
        cascade_service,
        "search_recordings",
        lambda idx, vec, top_k: ([{"recording_id": "rec_candidate_1", "similarity_score": 0.85}], {}),
    )

    monkeypatch.setattr(
        cascade_service.cover_service,
        "identify_cover",
        lambda **kw: {
            "matched": False,
            "best_match": None,
            "candidates": [{"recording_id": "rec_candidate_1", "similarity_score": 0.88, "oti": 0}],
            "top_candidate": {"recording_id": "rec_candidate_1", "similarity_score": 0.88, "oti": 0},
            "threshold": 0.97,
        },
    )

    result = process_music_query("mock_audio.wav", db=None, vector_index=mock_index)

    assert result["pipeline_stage"] == "STAGE_3_COVER"
    assert result["match_type"] == "UNKNOWN"
    assert result["recording_id"] is None
    assert result["condition"] == "HUMAN_REVIEW_REQUIRED"
    assert result["evidence"]["cover"]["matched"] is False


def test_cascade_cover_disabled_flag(monkeypatch):
    """Khi COVER_ENABLED = False, hệ thống bỏ qua tầng 3 và giữ STAGE_2_MERT_RETRIEVAL."""
    from unittest.mock import MagicMock
    from backend.services import cascade_service

    monkeypatch.setattr(config, "COVER_ENABLED", False)
    monkeypatch.setattr(cascade_service, "search_fingerprint",
                        lambda db, path: {"match_type": "NOT_FOUND", "fingerprint_score": 0.02})
    monkeypatch.setattr(cascade_service, "extract_mert_embedding",
                        lambda path, **kw: [0.1] * 768)

    mock_index = MagicMock()
    mock_index.ntotal = 100
    mock_index.n_recordings = 50
    mock_index.meta = {"model_version": "v1"}

    monkeypatch.setattr(
        cascade_service,
        "search_recordings",
        lambda idx, vec, top_k: ([{"recording_id": "rec_1", "similarity_score": 0.85}], {}),
    )

    result = process_music_query("mock_audio.wav", db=None, vector_index=mock_index)

    assert result["pipeline_stage"] == "STAGE_2_MERT_RETRIEVAL"
    assert result["match_type"] == "UNKNOWN"
    assert result["evidence"]["cover"] is None
