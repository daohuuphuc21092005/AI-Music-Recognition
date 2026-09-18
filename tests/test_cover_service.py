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
    describe_oti,
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


@pytest.mark.parametrize("semitones", [1, 2, 3, -1, -2, 5, -5])
def test_mo_ta_oti_dung_chieu_va_do_lon(reference_descriptor, semitones):
    """
    Câu giải thích hiện trên màn Result phải nói đúng điều đã xảy ra với truy vấn:
    dịch LÊN 1 bán cung cho OTI 11, và câu phải là "cao hơn 1", không phải "11".
    Neo vào tín hiệu thật (hợp âm đã dịch), không vào công thức, để hai phía cùng sai
    dấu thì test vẫn bắt được.
    """
    query = build_descriptor(chord(semitones), CHROMA_SR)
    _, oti = cover_similarity(query, reference_descriptor)

    direction = "cao" if semitones > 0 else "thấp"
    assert describe_oti(oti) == (
        f"OTI {oti}: truy vấn {direction} hơn bản gốc {abs(semitones)} bán cung")


def test_mo_ta_oti_khong_lech_va_nua_quang_tam():
    assert describe_oti(0) == "OTI 0: cùng cao độ với bản gốc"
    assert describe_oti(12) == "OTI 0: cùng cao độ với bản gốc"
    assert "nửa quãng tám" in describe_oti(6)


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


def test_loai_tru_cot_khong_bao_gio_tra_ve_ban_ghi_bi_loai(monkeypatch):
    """Held-out của tầng Cover: bản ghi bị loại không được xuất hiện ở bất kỳ vị trí nào."""
    from backend.services import cover_service

    rng = np.random.default_rng(3)
    dim = N_CHROMA * DESCRIPTOR_FRAMES
    matrix = rng.standard_normal((30, dim)).astype("float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    id_map = [f"rec_{i}" for i in range(30)]
    query = matrix[4].copy()  # truy vấn trùng hệt rec_4

    monkeypatch.setattr(cover_service, "query_descriptors_from_file",
                        lambda *a, **k: (query[None, :], [1.0]))
    index = cover_service.CoverIndex(matrix=matrix, id_map=id_map)

    base = cover_service.identify_cover("x.wav", cover_index=index, top_k=5)
    assert base["top_candidate"]["recording_id"] == "rec_4"

    excluded = frozenset({"rec_4", "rec_7"})
    held = cover_service.identify_cover("x.wav", cover_index=index, top_k=30,
                                        exclude_recording_ids=excluded)
    returned = {c["recording_id"] for c in held["candidates"]}
    assert not returned & excluded
    assert len(returned) == 28
    assert held["top_candidate"]["recording_id"] != "rec_4"


# --------------------------------------------------------------------------
# Cắt truy vấn theo nhiều hệ số nhịp độ
# --------------------------------------------------------------------------
# Tám hợp âm ba nốt đổi lần lượt: tín hiệu có CẤU TRÚC THỜI GIAN, nên cắt lệch
# đoạn nội dung là điểm tụt thấy rõ (một hợp âm giữ nguyên thì cắt kiểu gì cũng khớp).
PROGRESSION = ((0, 4, 7), (5, 9, 12), (7, 11, 14), (9, 12, 16),
               (2, 5, 9), (4, 7, 11), (10, 14, 17), (3, 7, 10))
PROGRESSION_S = 30.0


def progression(tempo: float = 1.0) -> np.ndarray:
    """
    Chuỗi hợp âm dài PROGRESSION_S / tempo giây, cao độ giữ nguyên — đổi nhịp độ
    CHÍNH XÁC (mỗi hợp âm kéo dài 1/tempo lần), không qua phase vocoder.
    """
    segment = PROGRESSION_S / len(PROGRESSION) / tempo
    t = np.linspace(0, segment, int(CHROMA_SR * segment), endpoint=False)
    parts = []
    for notes in PROGRESSION:
        signal = sum(np.sin(2 * np.pi * 261.63 * 2 ** (n / 12) * t) for n in notes)
        parts.append(signal / np.max(np.abs(signal)))
    return np.concatenate(parts).astype(np.float32)


@pytest.fixture(scope="module")
def progression_reference():
    from backend.services.cover_service import build_descriptor

    return build_descriptor(progression(1.0)[: int(PROGRESSION_S * CHROMA_SR)], CHROMA_SR)


def test_do_dai_cat_theo_he_so_nhip_do():
    from backend.services.cover_service import query_spans

    spans = dict(query_spans(30.0, (0.9, 1.0, 1.1)))
    assert spans[1.0] == 30.0
    assert spans[0.9] == pytest.approx(33.333, abs=1e-3)  # bản chậm cần đọc dài hơn
    assert spans[1.1] == pytest.approx(27.273, abs=1e-3)


def test_dong_nhip_1_trung_khit_cach_dung_reference(progression_reference):
    from backend.services.cover_service import query_descriptors

    rows, factors = query_descriptors(progression(1.0), CHROMA_SR, 30.0, (0.9, 1.0, 1.1))
    # Audio đúng 30 s: nhịp 0.9 (cắt 33,3 s) và 1.0 cho cùng một đoạn -> một dòng
    assert factors == [1.0, 1.1]
    assert np.array_equal(rows[0], progression_reference)


def test_he_so_gan_cho_dong_trung_la_do_dai_gan_nhat_khong_phai_dong_dau():
    """Bản gốc 30 s không được báo là "khớp khi coi là chậm 0.90×"."""
    from backend.services.cover_service import query_descriptors

    factors = (0.9, 0.95, 1.0, 1.05, 1.1)
    assert query_descriptors(progression(1.0), CHROMA_SR, 30.0, factors)[1] == [1.0, 1.05, 1.1]
    # File 27,25 s (tempo 1.10 của tập kiểm thử) — mọi hệ số cùng một đoạn
    fast = progression(1.1)[: int(27.25 * CHROMA_SR)]
    assert query_descriptors(fast, CHROMA_SR, 30.0, factors)[1] == [1.1]


def test_ban_cham_chi_khop_khi_cat_dung_do_dai(progression_reference):
    """Bằng chứng cho lý do đổi: cắt cố định 30 s làm bản chậm 0.90× lệch nội dung."""
    from backend.services.cover_service import query_descriptors

    slowed = progression(0.9)
    rows, factors = query_descriptors(slowed, CHROMA_SR, 30.0, (0.9, 0.95, 1.0, 1.05, 1.1))
    matches = search(rows, progression_reference[None, :], top_k=1)
    fixed = search(rows[factors.index(1.0)], progression_reference[None, :], top_k=1)

    assert factors[matches[0]["query_row"]] == 0.9
    assert matches[0]["similarity_score"] > 0.99
    assert matches[0]["similarity_score"] - fixed[0]["similarity_score"] > 0.05


def test_audio_ngan_hon_moi_do_dai_cat_chi_tinh_mot_lan():
    from backend.services import cover_service

    calls = []
    original = cover_service.build_descriptor

    def counting(audio, sr=CHROMA_SR):
        calls.append(len(audio))
        return original(audio, sr)

    audio = chord(duration=5.0)
    cover_service.build_descriptor = counting
    try:
        rows, factors = cover_service.query_descriptors(audio, CHROMA_SR, 30.0, (0.9, 1.0, 1.1))
    finally:
        cover_service.build_descriptor = original
    assert len(calls) == 1
    assert len(rows) == 1
    assert factors == [None]  # 5 giây: không ứng với hệ số nhịp độ nào


def test_search_nhieu_dong_lay_dong_diem_cao_nhat_cho_tung_ung_vien():
    rng = np.random.default_rng(5)
    dim = N_CHROMA * DESCRIPTOR_FRAMES
    matrix = rng.standard_normal((6, dim)).astype("float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    queries = np.vstack([matrix[2], matrix[4]])

    results = {r["index"]: r for r in search(queries, matrix, top_k=6)}
    assert results[2]["query_row"] == 0 and results[4]["query_row"] == 1
    assert results[2]["similarity_score"] == pytest.approx(1.0, abs=1e-5)
    assert results[4]["similarity_score"] == pytest.approx(1.0, abs=1e-5)
