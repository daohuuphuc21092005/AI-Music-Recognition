"""
Test cho bộ đo đặc trưng sản xuất.

Khác với bộ phân loại giấy phép, module này không có model nào để nghi ngờ: mọi
chỉ số đều là đại lượng vật lý tính thẳng từ tín hiệu. Nên test ở đây kiểm hai
thứ: (1) các chỉ số phản ứng ĐÚNG CHIỀU với thao tác đã biết trên tín hiệu, và
(2) kết quả luôn mang theo cảnh báo rằng nó không phải chỉ báo bản quyền.
"""
import numpy as np
import pytest

from backend.services.production_features_service import (
    analyze,
    extract_production_features,
    production_polish,
)

SR = 22050


def tone(seconds: float = 3.0, freq: float = 220.0, amplitude: float = 0.5):
    t = np.linspace(0, seconds, int(seconds * SR), dtype="float32")
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype("float32")


def compressed(seconds: float = 3.0):
    """Tín hiệu bị nén/limit kịch: gần như vuông, sát trần."""
    return (np.tanh(tone(seconds) * 40) * 0.99).astype("float32")


def wide_dynamics(seconds: float = 3.0):
    """Tín hiệu có dải động rộng: biên độ tăng dần từ rất nhỏ tới lớn."""
    base = tone(seconds)
    envelope = np.linspace(0.01, 0.9, len(base), dtype="float32")
    return (base * envelope).astype("float32")


# ---------------------------------------------------------------------------
# Chỉ số phản ứng đúng chiều
# ---------------------------------------------------------------------------
def test_tin_hieu_qua_ngan_tra_none():
    """Thà không có số còn hơn một con số tính trên vài mẫu rồi bị đọc như thật."""
    assert extract_production_features(np.zeros(100, dtype="float32"), SR) is None
    assert extract_production_features(None, SR) is None
    assert analyze(np.zeros(10, dtype="float32"), SR) is None


def test_nen_manh_lam_crest_factor_nho_hon():
    loose = extract_production_features(wide_dynamics(), SR)
    tight = extract_production_features(compressed(), SR)
    assert tight["crest_factor_db"] < loose["crest_factor_db"]


def test_tang_bien_do_lam_rms_tang():
    quiet = extract_production_features(tone(amplitude=0.05), SR)
    loud = extract_production_features(tone(amplitude=0.9), SR)
    assert loud["rms_dbfs"] > quiet["rms_dbfs"]


def test_dai_dong_rong_cho_dynamic_range_lon_hon():
    steady = extract_production_features(tone(), SR)
    varying = extract_production_features(wide_dynamics(), SR)
    assert varying["dynamic_range_db"] > steady["dynamic_range_db"]


def test_phat_hien_mau_sat_tran():
    clipped = np.clip(tone(amplitude=2.0), -1.0, 1.0).astype("float32")
    features = extract_production_features(clipped, SR)
    assert features["clipping_ratio"] > 0.01

    clean = extract_production_features(tone(amplitude=0.3), SR)
    assert clean["clipping_ratio"] == 0.0


def test_diem_polish_cao_hon_khi_nen_manh_va_to():
    tight = production_polish(extract_production_features(compressed(), SR))
    loose = production_polish(extract_production_features(tone(amplitude=0.05), SR))
    assert tight["polish_score"] > loose["polish_score"]


# ---------------------------------------------------------------------------
# Ràng buộc về cách trình bày (§2)
# ---------------------------------------------------------------------------
def test_ket_qua_luon_kem_canh_bao_khong_phai_ban_quyen():
    result = analyze(compressed(), SR)
    assessment = result["assessment"]
    assert assessment["khong_phai_chi_bao_ban_quyen"] is True
    assert "bản quyền" in assessment["canh_bao"].lower()


def test_diem_so_luon_kem_dien_giai_tung_thanh_phan():
    """
    §2 cấm dùng một điểm duy nhất làm kết luận. Người đọc phải thấy điểm đến từ
    đâu để tự phản bác được, nên `contributions` là bắt buộc chứ không tuỳ chọn.
    """
    assessment = analyze(compressed(), SR)["assessment"]
    factors = {c["yeu_to"] for c in assessment["contributions"]}
    assert factors == {"loudness", "nen_dai_dong", "on_dinh_do_to"}
    for contribution in assessment["contributions"]:
        assert 0.0 <= contribution["diem"] <= 1.0
        assert contribution["giai_thich"]


@pytest.mark.parametrize("signal", [compressed(), tone(), wide_dynamics()])
def test_diem_nam_trong_khoang_va_nhan_hop_le(signal):
    assessment = analyze(signal, SR)["assessment"]
    assert 0.0 <= assessment["polish_score"] <= 1.0
    assert assessment["nhan"] in {
        "co_dau_vet_mastering_thuong_mai",
        "khong_ket_luan_duoc",
        "giong_ban_thu_moc_hoac_nghiep_du",
    }
