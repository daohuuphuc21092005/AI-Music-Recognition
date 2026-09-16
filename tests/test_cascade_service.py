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

    # MERT trượt ngưỡng thì tầng 3 (Cover) chạy tiếp, nên khi Cover được bật tầng
    # sâu nhất đã chạy có thể là STAGE_3_COVER. Điều cần kiểm ở đây là đã rời tầng 1.
    allowed = {"STAGE_2_MERT_RETRIEVAL"} | ({"STAGE_3_COVER"} if config.COVER_ENABLED else set())
    assert result["pipeline_stage"] in allowed
    assert result["match_type"] in ("NEAR_MATCH", "UNKNOWN")
    evidence = result["evidence"]
    fingerprint = evidence["fingerprint"]

    # So với ngưỡng HIỆU DỤNG, không phải τFP cơ sở. Fixture là nhiễu trắng 8
    # giây, mà điểm nền của một đoạn ngắn cao hơn hẳn đoạn 30 giây (đo được:
    # 0.2558 so với 0.0814), nên `search_fingerprint` tự nâng ngưỡng theo độ dài.
    # Con số 0.1628 của đoạn 8 giây này từng VƯỢT τFP cơ sở hồi τFP = 0.15 — nếu
    # chỉ dùng một ngưỡng cố định thì nhiễu trắng đã bị gán EXACT_MATCH. τFP hiện
    # là 0.30 nên nhiễu trắng trượt sẵn, nhưng phép nâng ngưỡng vẫn phải giữ: nó
    # bảo vệ theo ĐỘ DÀI truy vấn, độc lập với giá trị τFP đang hiệu chỉnh.
    assert fingerprint["fingerprint_score"] < fingerprint["threshold"]
    assert fingerprint["threshold"] >= config.FP_THRESHOLD
    assert "embedding_ms" in evidence["timings_ms"]


@requires_db
def test_nguong_duoc_nang_cho_truy_van_ngan(db_session, vector_index, unmatched_audio):
    """
    Truy vấn ngắn phải được áp ngưỡng CAO HƠN τFP cơ sở.

    Đây là chốt chặn cho một dương tính giả có thật: đoạn 8 giây bất kỳ, kể cả
    nhiễu trắng thuần, đạt điểm Chromaprint quanh 0.16 — từng vượt τFP = 0.15
    hiệu chỉnh trên tập truy vấn 30 giây.

    Kỳ vọng KHÔNG đóng đinh "luôn được nâng": phép nâng chỉ kích hoạt khi sàn nhiễu
    của độ dài đó cao hơn τFP. Ở τFP = 0.30 thì sàn nhiễu 8 giây (0.2558 × 1.15 =
    0.2942) đã nằm dưới τFP, nên không còn gì để nâng — chính τFP đang gánh. Điều
    phải đúng trong MỌI trường hợp là hai điều dưới: ngưỡng hiệu dụng bằng đúng
    hàm theo độ dài, và nhiễu trắng bị từ chối.
    """
    from backend.services.fingerprint_service import min_score_for_duration

    result = process_music_query(unmatched_audio, db_session, vector_index, top_k=5)
    fingerprint = result["evidence"]["fingerprint"]
    expected = min_score_for_duration(fingerprint["query_duration"])

    assert fingerprint["threshold_base"] == config.FP_THRESHOLD
    assert fingerprint["threshold"] == pytest.approx(expected)
    assert fingerprint["threshold"] >= config.FP_THRESHOLD
    assert fingerprint["threshold_raised_for_short_query"] is (expected > config.FP_THRESHOLD)
    # Thứ thực sự bảo vệ hệ thống: nhiễu trắng không được nhận là EXACT_MATCH
    assert fingerprint["fingerprint_score"] < fingerprint["threshold"]


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


def test_file_hong_o_tang_1_tra_no_audio(monkeypatch):
    """fpcalc không đọc được file -> NO_AUDIO, không để thoát ra thành INTERNAL_ERROR."""
    from unittest.mock import MagicMock
    from backend.services import cascade_service
    from backend.services.fingerprint_service import AudioFingerprintError

    def fpcalc_hong(db, path):
        raise AudioFingerprintError(r"Không đọc được luồng audio: C:\Users\ai-do\temp\x.mp3")

    monkeypatch.setattr(cascade_service, "search_fingerprint", fpcalc_hong)
    with pytest.raises(PipelineError) as exc:
        process_music_query("hong.mp3", db=None, vector_index=MagicMock(ntotal=1))
    assert exc.value.code == "NO_AUDIO"
    assert "C:\\" not in exc.value.message


@pytest.mark.parametrize("error_name, expected_code", [
    ("EmbeddingAudioError", "NO_AUDIO"),
    ("EmbeddingModelError", "MODEL_FAILURE"),
])
def test_loi_mert_duoc_quy_dung_cho(monkeypatch, error_name, expected_code):
    """Lỗi FILE và lỗi MÁY CHỦ phải ra hai mã khác nhau (§12) — bản cũ gộp cả hai thành NO_AUDIO."""
    from unittest.mock import MagicMock
    from backend.services import cascade_service, embedding_service

    error_cls = getattr(embedding_service, error_name)

    def mert_hong(path, **kw):
        raise error_cls("loi gia lap")

    monkeypatch.setattr(cascade_service, "search_fingerprint",
                        lambda db, path: {"match_type": "NOT_FOUND", "fingerprint_score": 0.02})
    monkeypatch.setattr(cascade_service, "extract_mert_embedding", mert_hong)
    with pytest.raises(PipelineError) as exc:
        process_music_query("x.mp3", db=None, vector_index=MagicMock(ntotal=1))
    assert exc.value.code == expected_code


def test_loi_tang_cover_khong_lo_noi_dung_ngoai_le(monkeypatch):
    """Nội dung ngoại lệ tầng Cover (có thể chứa đường dẫn máy chủ) chỉ vào log."""
    from unittest.mock import MagicMock
    from backend.services import cascade_service

    monkeypatch.setattr(config, "COVER_ENABLED", True)
    monkeypatch.setattr(cascade_service, "search_fingerprint",
                        lambda db, path: {"match_type": "NOT_FOUND", "fingerprint_score": 0.02})
    monkeypatch.setattr(cascade_service, "extract_mert_embedding", lambda path, **kw: [0.1] * 768)
    monkeypatch.setattr(
        cascade_service, "search_recordings",
        lambda idx, vec, top_k: ([{"recording_id": "rec_1", "similarity_score": 0.5}], {}),
    )

    def cover_hong(**kw):
        raise FileNotFoundError(r"D:\music-rights-data\audio\bi-mat.npy")

    monkeypatch.setattr(cascade_service.cover_service, "identify_cover", cover_hong)
    index = MagicMock(ntotal=10, n_recordings=5, meta={"model_version": "v1"})
    result = process_music_query("x.mp3", db=None, vector_index=index)

    assert result["match_type"] == "UNKNOWN"
    assert result["evidence"]["cover"]["error"] == "COVER_STAGE_FAILED"
    assert "music-rights-data" not in str(result["evidence"])
