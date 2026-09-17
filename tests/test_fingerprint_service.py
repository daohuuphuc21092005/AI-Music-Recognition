"""Kiểm thử tầng 1 (Chromaprint) trên database thật."""
import numpy as np

from backend import config
from backend.services.fingerprint_service import (
    backend_status,
    is_available,
    search_fingerprint,
)
from tests.conftest import requires_db, requires_fpcalc


def test_backend_status_co_du_khoa():
    status = backend_status()
    assert "fpcalc" in status
    assert "libchromaprint" in status
    assert isinstance(is_available(), bool)


@requires_db
@requires_fpcalc
def test_search_fingerprint_tra_ve_quyet_dinh_ro_rang(db_session, test_audio):
    result = search_fingerprint(db_session, test_audio)

    assert result["match_type"] in ("EXACT_MATCH", "NO_MATCH")
    assert 0.0 <= result["fingerprint_score"] <= 1.0
    assert result["threshold"] == config.FP_THRESHOLD

    if result["match_type"] == "EXACT_MATCH":
        assert result["recording_id"] is not None
        assert result["fingerprint_score"] >= config.FP_THRESHOLD
    else:
        # Không được trả recording_id khi điểm dưới ngưỡng
        assert result["recording_id"] is None


# ---------------------------------------------------------------------------
# Ngưỡng phụ thuộc độ dài truy vấn
# ---------------------------------------------------------------------------
def test_nguong_giam_dan_khi_truy_van_dai_ra():
    """
    Đoạn càng dài thì ngưỡng càng thấp, vì điểm nền của một truy vấn không có
    trong CSDL giảm theo độ dài (ít offset để dò -> ít cơ hội gặp offset may mắn).
    """
    from backend.services.fingerprint_service import min_score_for_duration

    thresholds = [min_score_for_duration(d) for d in (5, 8, 10, 15, 20, 30)]
    assert thresholds == sorted(thresholds, reverse=True)


def test_nguong_khong_bao_gio_thap_hon_tau_fp():
    """Hàm chỉ được NÂNG ngưỡng, không bao giờ hạ dưới giá trị đã hiệu chỉnh."""
    from backend.services.fingerprint_service import min_score_for_duration

    for duration in (1, 5, 8, 15, 30, 60, 600):
        assert min_score_for_duration(duration) >= config.FP_THRESHOLD


def test_co_che_nang_nguong_van_con_song():
    """
    Phải còn ÍT NHẤT một độ dài mà phép nâng thực sự kích hoạt.

    Nếu τFP hiệu chỉnh lại mà vượt mọi sàn nhiễu đã đo, cơ chế này lặng lẽ thành
    mã chết: mọi truy vấn ngắn đều dùng τFP, và bảng NOISE_FLOOR_BY_DURATION không
    còn bảo vệ gì. Lúc đó phải đo lại sàn nhiễu trên corpus mới chứ không phải xoá
    test. (Ở τFP = 0.30, phép nâng còn kích hoạt cho đoạn dưới 8 giây: 5 giây ->
    0.4237, trong khi 8 giây trở lên đã bằng chính τFP.)
    """
    from backend.services.fingerprint_service import (
        NOISE_FLOOR_BY_DURATION,
        NOISE_FLOOR_MARGIN,
        min_score_for_duration,
    )

    engaged = [d for d, floor in NOISE_FLOOR_BY_DURATION
               if floor * NOISE_FLOOR_MARGIN > config.FP_THRESHOLD]
    assert engaged, (
        f"τFP = {config.FP_THRESHOLD} đã cao hơn mọi sàn nhiễu đã đo -> cơ chế nâng "
        f"ngưỡng theo độ dài không còn tác dụng, cần đo lại sàn nhiễu"
    )
    for duration in engaged:
        assert min_score_for_duration(duration) > config.FP_THRESHOLD


def test_nguong_chan_duoc_diem_nen_da_do():
    """
    Ngưỡng phải nằm TRÊN điểm nền lớn nhất đã đo cho từng độ dài — nếu không thì
    một truy vấn không hề có trong CSDL vẫn có thể bị gán EXACT_MATCH.
    """
    from backend.services.fingerprint_service import (
        NOISE_FLOOR_BY_DURATION,
        min_score_for_duration,
    )

    for duration, measured_max in NOISE_FLOOR_BY_DURATION:
        assert min_score_for_duration(duration) > measured_max, (
            f"{duration}s: ngưỡng {min_score_for_duration(duration):.4f} không "
            f"vượt điểm nền đo được {measured_max}"
        )


def test_do_dai_khong_hop_le_thi_dung_tau_fp():
    from backend.services.fingerprint_service import min_score_for_duration

    assert min_score_for_duration(0) == config.FP_THRESHOLD
    assert min_score_for_duration(None) == config.FP_THRESHOLD
    assert min_score_for_duration(-5) == config.FP_THRESHOLD


def test_loai_tru_ban_ghi_o_tang_chromaprint(monkeypatch):
    """Bản ghi bị loại không được thắng, kể cả khi nó trùng hệt truy vấn."""
    import numpy as np

    from backend.services import fingerprint_service

    query = np.arange(200, dtype=np.uint32)
    references = [("goc", query.copy(), 30.0),
                  ("khac", np.arange(1000, 1200, dtype=np.uint32), 30.0)]
    monkeypatch.setattr(fingerprint_service, "extract_query_fingerprint",
                        lambda path: (30.0, "fp"))
    monkeypatch.setattr(fingerprint_service, "_decode_array", lambda fp: query)
    monkeypatch.setattr(fingerprint_service, "_reference_fingerprints",
                        lambda db: (references, 0))

    production = fingerprint_service.search_fingerprint(None, "q.wav")
    assert production["match_type"] == "EXACT_MATCH"
    assert production["recording_id"] == "goc"

    held = fingerprint_service.search_fingerprint(
        None, "q.wav", exclude_recording_ids=frozenset({"goc"}))
    assert held["recording_id"] != "goc"
    assert held.get("best_candidate_below_threshold") != "goc"
    # Bản ghi bị loại không chiếm chỗ trong top-K của bộ lọc; "khac" không có hash
    # trùng nào nên không cần chấm
    assert held["candidates_compared"] == 0

    monkeypatch.setattr(config, "FP_PREFILTER_TOP_K", 0)
    full = fingerprint_service.search_fingerprint(
        None, "q.wav", exclude_recording_ids=frozenset({"goc"}))
    assert full["recording_id"] != "goc"
    assert full["candidates_compared"] == 1


# ---------------------------------------------------------------------------
# Bộ lọc ứng viên theo hash trùng
# ---------------------------------------------------------------------------
def _patch_references(monkeypatch, query, references, duration=30.0):
    from backend.services import fingerprint_service

    monkeypatch.setattr(fingerprint_service, "extract_query_fingerprint",
                        lambda path: (duration, "fp"))
    monkeypatch.setattr(fingerprint_service, "_decode_array", lambda fp: query)
    monkeypatch.setattr(fingerprint_service, "_reference_fingerprints",
                        lambda db: (references, 0))
    return fingerprint_service


def _random_references(rng, n, length=240):
    return [(f"r{i}", rng.integers(0, 2 ** 32, length, dtype=np.uint64).astype(np.uint32), 30.0)
            for i in range(n)]


def test_dem_hash_trung_khop_cach_dem_vong_lap_da_doi_chung():
    """Bản vector hoá phải đếm y hệt vòng lặp np.add.at của phép đối chứng 1.900 truy vấn."""
    from backend.services.fingerprint_service import _hash_index, hash_hits

    rng = np.random.default_rng(0)
    # Giá trị nhỏ để có nhiều hash trùng lặp, cả trong truy vấn lẫn giữa các bản ghi
    references = [(f"r{i}", rng.integers(0, 50, 80).astype(np.uint32), 30.0) for i in range(30)]
    query = rng.integers(0, 50, 120).astype(np.uint32)

    hashes, owners = _hash_index(references)
    expected = np.zeros(len(references), dtype=np.int64)
    for value in query:
        lo, hi = np.searchsorted(hashes, value, "left"), np.searchsorted(hashes, value, "right")
        np.add.at(expected, owners[lo:hi], 1)
    assert np.array_equal(hash_hits(query, hashes, owners, len(references)), expected)


def test_loc_ung_vien_cho_cung_ket_qua_voi_quet_day_du(monkeypatch):
    rng = np.random.default_rng(1)
    references = _random_references(rng, 300)
    target = references[137][1].copy()
    # Truy vấn = bản ghi đích bị lật 1 bit ở một nửa số item: vẫn khớp (bit-error <= 2)
    # nhưng chỉ nửa còn lại là hash trùng tuyệt đối
    target[::2] ^= np.uint32(1)
    service = _patch_references(monkeypatch, target, references)

    fast = service.search_fingerprint(None, "q.wav")
    monkeypatch.setattr(config, "FP_PREFILTER_TOP_K", 0)
    full = service.search_fingerprint(None, "q.wav")

    assert fast["recording_id"] == full["recording_id"] == "r137"
    assert fast["fingerprint_score"] == full["fingerprint_score"]
    assert fast["prefilter"]["full_scan"] is False
    assert fast["candidates_compared"] < full["candidates_compared"] == 300
    assert full["prefilter"]["full_scan_reason"] == "PREFILTER_DISABLED"


def test_diem_sat_nguong_thi_quay_ve_quet_day_du(monkeypatch):
    rng = np.random.default_rng(2)
    references = _random_references(rng, 100)
    query = references[5][1].copy()
    # Chỉ 30% item còn khớp -> điểm ~0.30, nằm trong dải τ ± 0.15
    query[: int(len(query) * 0.7)] = rng.integers(0, 2 ** 32, int(len(query) * 0.7),
                                                   dtype=np.uint64).astype(np.uint32)
    service = _patch_references(monkeypatch, query, references)

    result = service.search_fingerprint(None, "q.wav")
    assert abs(result["fingerprint_score"] - result["threshold"]) <= config.FP_PREFILTER_BAND
    assert result["prefilter"]["full_scan_reason"] == "NEAR_THRESHOLD"
    assert result["candidates_compared"] == 100


def test_truy_van_ngan_luon_quet_day_du(monkeypatch):
    rng = np.random.default_rng(3)
    references = _random_references(rng, 50)
    service = _patch_references(monkeypatch, references[7][1].copy(), references,
                                duration=config.FP_PREFILTER_MIN_QUERY_S - 1)

    result = service.search_fingerprint(None, "q.wav")
    assert result["recording_id"] == "r7"
    assert result["prefilter"]["full_scan_reason"] == "QUERY_TOO_SHORT"
    assert result["candidates_compared"] == 50
