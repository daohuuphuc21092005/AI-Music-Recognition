"""Kiểm thử tầng 1 (Chromaprint) trên database thật."""
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
