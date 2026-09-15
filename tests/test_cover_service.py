"""
Test cho CoverService — baseline CQT/chroma + OTI.

Tín hiệu dùng trong test là TỔNG HỢP chứ không phải file nhạc: một hợp âm dựng
từ các sóng sin có tần số biết trước. Nhờ vậy "dịch lên k bán cung" là phép nhân
tần số với 2^(k/12) — chính xác tuyệt đối, không phụ thuộc phase vocoder của
librosa, và test chạy trong vài trăm ms thay vì vài giây.

Tính chất cần khẳng định: dịch cao độ k bán cung phải làm chroma XOAY đúng k bậc,
nên OTI phải trả về (-k) mod 12. Đây là toàn bộ lý do tầng này tồn tại (EXP-05:
dịch cao độ là điểm mù 0% của cả Chromaprint lẫn MERT).
"""
import numpy as np
import pytest

from backend import config
from backend.services.cover_service import (
    CHROMA_SR,
    DESCRIPTOR_FRAMES,
    N_CHROMA,
    CoverFeatureError,
    _resample_time,
    build_descriptor,
    cover_similarity,
    extract_chroma,
    normalize_frames,
    search,
    transpositions,
)

DURATION_S = 6.0
# Đô trưởng: C4 - E4 - G4
BASE_FREQUENCIES = (261.63, 329.63, 392.00)


def chord(semitones: float = 0.0, duration: float = DURATION_S,
          seed: int = 0) -> np.ndarray:
    """Hợp âm ba nốt đã dịch lên `semitones` bán cung, kèm chút nhiễu rất nhỏ."""
    t = np.linspace(0, duration, int(CHROMA_SR * duration), endpoint=False)
    ratio = 2 ** (semitones / 12)
    signal = sum(np.sin(2 * np.pi * f * ratio * t) for f in BASE_FREQUENCIES)
    signal += np.random.RandomState(seed).normal(0, 1e-4, signal.shape)
    return (signal / np.max(np.abs(signal))).astype(np.float32)


@pytest.fixture(scope="module")
def reference_descriptor():
    return build_descriptor(chord(0.0), CHROMA_SR)


# --------------------------------------------------------------------------
# Đặc trưng
# --------------------------------------------------------------------------
def test_chroma_co_dung_12_bac():
    assert extract_chroma(chord(), CHROMA_SR).shape[0] == N_CHROMA


def test_descriptor_chuan_hoa_L2(reference_descriptor):
    assert reference_descriptor.shape == (N_CHROMA * DESCRIPTOR_FRAMES,)
    assert np.isclose(np.linalg.norm(reference_descriptor), 1.0, atol=1e-5)


def test_audio_qua_ngan_bao_loi_ro_rang():
    with pytest.raises(CoverFeatureError):
        build_descriptor(np.zeros(100, dtype=np.float32), CHROMA_SR)


@pytest.mark.parametrize("n_frames_in", [7, DESCRIPTOR_FRAMES, 500])
def test_resample_time_luon_ra_dung_so_khung(n_frames_in):
    chroma = np.random.RandomState(1).random((N_CHROMA, n_frames_in))
    assert _resample_time(chroma).shape == (N_CHROMA, DESCRIPTOR_FRAMES)


def test_resample_khong_dem_so_0_cho_doan_ngan():
    """Đệm 0 sẽ tạo ra khung 'im lặng' giả và kéo tụt similarity vô căn cứ."""
    chroma = np.ones((N_CHROMA, 5))
    assert np.all(_resample_time(chroma) > 0)


# --------------------------------------------------------------------------
# Bất biến với dịch cao độ — lý do tầng này tồn tại
# --------------------------------------------------------------------------
def test_tu_so_khop_voi_chinh_no_bang_1(reference_descriptor):
    score, oti = cover_similarity(reference_descriptor, reference_descriptor)
    assert np.isclose(score, 1.0, atol=1e-5)
    assert oti == 0


@pytest.mark.parametrize("semitones", [1, 2, 3, -1, -2, 5])
def test_oti_bat_dung_luong_dich_cao_do(reference_descriptor, semitones):
    query = build_descriptor(chord(semitones), CHROMA_SR)
    score, oti = cover_similarity(query, reference_descriptor)

    assert oti == (-semitones) % 12, (
        f"dịch {semitones:+d} bán cung phải cho OTI {(-semitones) % 12}, nhận {oti}"
    )
    # Xoay đúng phải khá hơn hẳn so với không xoay
    assert score > float(query @ reference_descriptor)
    assert score > 0.8


def test_transpositions_dong_dau_la_chinh_no(reference_descriptor):
    rotated = transpositions(reference_descriptor)
    assert rotated.shape == (N_CHROMA, reference_descriptor.size)
    assert np.allclose(rotated[0], reference_descriptor)


def test_moi_phep_xoay_deu_giu_chuan_L2(reference_descriptor):
    assert np.allclose(np.linalg.norm(transpositions(reference_descriptor), axis=1),
                       1.0, atol=1e-5)


# --------------------------------------------------------------------------
# Tìm kiếm
# --------------------------------------------------------------------------
def test_search_xep_hang_giam_dan_va_tra_dung_index():
    references = np.vstack([
        build_descriptor(chord(0.0), CHROMA_SR),        # 0: đúng
        build_descriptor(chord(7.0), CHROMA_SR),        # 1: quãng năm
        build_descriptor(chord(0.0, seed=99), CHROMA_SR),  # 2: gần như trùng 0
    ])
    ranked = search(build_descriptor(chord(2.0), CHROMA_SR), references, top_k=3)

    assert [c["index"] for c in ranked] == sorted(
        (c["index"] for c in ranked),
        key=lambda i: -next(c["similarity_score"] for c in ranked if c["index"] == i))
    scores = [c["similarity_score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 + 1e-6 for s in scores)
    assert all(0 <= c["oti"] < N_CHROMA for c in ranked)


def test_search_reference_rong_tra_danh_sach_rong(reference_descriptor):
    assert search(reference_descriptor, np.empty((0, reference_descriptor.size))) == []
    assert search(reference_descriptor, None) == []


def test_search_ton_trong_top_k(reference_descriptor):
    references = np.vstack([build_descriptor(chord(k), CHROMA_SR) for k in range(4)])
    assert len(search(reference_descriptor, references, top_k=2)) == 2


# --------------------------------------------------------------------------
# Không có hoà âm thì không được giống bài nào
# --------------------------------------------------------------------------
def test_khung_chroma_phang_thanh_vector_0():
    assert not np.any(normalize_frames(np.ones((N_CHROMA, DESCRIPTOR_FRAMES))))


def test_nhieu_trang_khong_khop_cover():
    """Trước khi trừ trung bình, nhiễu trắng đạt 0.985 so với nhạc thật — vượt τCover."""
    noise = np.random.RandomState(42).normal(
        0, 0.2, int(CHROMA_SR * DURATION_S)).astype(np.float32)
    references = np.vstack([build_descriptor(chord(k), CHROMA_SR) for k in range(N_CHROMA)])
    best = search(build_descriptor(noise, CHROMA_SR), references, top_k=1)[0]
    # 0.8 là mức một phép dịch cao độ thật phải vượt (test_oti_bat_dung_luong_dich_cao_do)
    assert best["similarity_score"] < 0.8
    assert best["similarity_score"] < config.COVER_THRESHOLD


def test_chuan_hoa_lai_descriptor_da_luu_bang_tinh_tu_audio():
    """Nhờ tính chất này, index cũ nâng cấp được mà không cần audio gốc."""
    chroma = _resample_time(extract_chroma(chord(3.0), CHROMA_SR))
    per_frame = chroma / np.linalg.norm(chroma, axis=0, keepdims=True)
    assert np.allclose(normalize_frames(per_frame), normalize_frames(chroma), atol=1e-5)
