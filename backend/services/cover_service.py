"""
CoverService — nhận dạng phiên bản/cover bằng đặc trưng CQT/chroma (CLAUDE.md §10).

Vì sao cần tầng này, bằng chứng chứ không phải phỏng đoán: EXP-05 đo được **dịch
cao độ là điểm mù 0% của CẢ HAI tầng hiện có** — Chromaprint tụt từ 0.893 xuống
0.009 (gần như ngẫu nhiên), và MERT tuy vẫn cho similarity ~0.91 nhưng không đủ
vượt τMERT = 0.95. Cả hai đều mã hoá cao độ TUYỆT ĐỐI, nên dịch giọng một cung
là đủ để phá.

Chroma thì khác: nó gập toàn bộ phổ về 12 bậc trong một quãng tám, nên **dịch cao
độ k bán cung = xoay vòng vector chroma đi đúng k bậc**. Đây chính là ý tưởng
Optimal Transposition Index (OTI) của Serrà và cộng sự: thử cả 12 phép xoay, lấy
phép cho điểm cao nhất, và trả kèm k để giải thích được kết quả.

Phạm vi trung thực: đây là **baseline CQT/chroma** đúng như §5 mô tả, KHÔNG phải
mô hình cover chuyên dụng (CoverHunter/CSI). Nó bất biến với dịch cao độ, và bất
biến một phần với đổi tốc độ nhờ chuẩn hoá độ dài chuỗi; nó KHÔNG xử lý được
mashup, phối lại cấu trúc, hay đổi hoà âm nhiều.

Ngưỡng chấp nhận τCover phải hiệu chỉnh bằng EXP-07, không được tự chọn (§2, §4).
"""
import json
import os

import librosa
import numpy as np

from backend import config

# Chroma không cần tần số lấy mẫu cao: nó chỉ quan tâm 12 bậc trong quãng tám.
CHROMA_SR = 22050
HOP_LENGTH = 512
# 3 bin mỗi bán cung -> CQT đủ mịn để chroma không bị nhoè giữa hai bậc kề nhau.
BINS_PER_OCTAVE = 36
# Chuẩn hoá mọi đoạn về cùng số khung, để hai đoạn dài ngắn khác nhau (do đổi
# tốc độ) vẫn so sánh được với nhau.
DESCRIPTOR_FRAMES = 64
N_CHROMA = 12


class CoverIndex:
    """Chứa ma trận descriptors và mapping id cho việc tra cứu cover."""

    def __init__(self, matrix: np.ndarray, id_map: list):
        self.matrix = matrix
        self.id_map = id_map
        self.n_items = len(matrix) if matrix is not None else 0


_default_cover_index = None
_index_loaded = False


def load_cover_index(matrix_path: str = None, id_map_path: str = None) -> CoverIndex | None:
    """Tải chỉ mục cover từ đĩa (matrix .npy và id_map .json)."""
    matrix_path = matrix_path or config.COVER_INDEX_PATH
    id_map_path = id_map_path or config.COVER_ID_MAP_PATH

    if not os.path.exists(matrix_path) or not os.path.exists(id_map_path):
        return None

    try:
        matrix = np.load(matrix_path)
        with open(id_map_path, "r", encoding="utf-8") as f:
            id_map = json.load(f)
        return CoverIndex(matrix=matrix, id_map=id_map)
    except Exception:
        return None


def get_default_cover_index() -> CoverIndex | None:
    """Lazy-load chỉ mục cover mặc định."""
    global _default_cover_index, _index_loaded
    if not _index_loaded:
        _default_cover_index = load_cover_index()
        _index_loaded = True
    return _default_cover_index


class CoverFeatureError(RuntimeError):
    """Không trích được đặc trưng chroma từ tín hiệu đưa vào."""


def extract_chroma(audio: np.ndarray, sr: int = CHROMA_SR) -> np.ndarray:
    """Trả ma trận chroma (12, T) từ CQT. Chưa chuẩn hoá."""
    if audio is None or len(audio) < sr // 10:
        raise CoverFeatureError("Đoạn audio quá ngắn để tính chroma (< 0.1 s).")
    return librosa.feature.chroma_cqt(
        y=np.asarray(audio, dtype=np.float32), sr=sr,
        hop_length=HOP_LENGTH, bins_per_octave=BINS_PER_OCTAVE, n_chroma=N_CHROMA,
    )


def _resample_time(chroma: np.ndarray, n_frames: int = DESCRIPTOR_FRAMES) -> np.ndarray:
    """
    Gộp trục thời gian về đúng `n_frames` khung bằng trung bình theo từng ô.

    Đây là chỗ xử lý đổi tốc độ: một đoạn bị kéo dài 10% vẫn cho ra chuỗi cùng độ
    dài, nên phép so khớp không bị lệch pha thời gian ngay từ đầu.
    """
    total = chroma.shape[1]
    if total == 0:
        raise CoverFeatureError("Chroma rỗng.")
    if total < n_frames:
        # Đoạn quá ngắn: lặp lại chỉ số thay vì đệm số 0 (đệm 0 sẽ tạo ra những
        # khung "im lặng" giả, kéo tụt similarity một cách vô căn cứ).
        index = np.linspace(0, total - 1, n_frames).round().astype(int)
        return chroma[:, index]

    edges = np.linspace(0, total, n_frames + 1).round().astype(int)
    return np.stack(
        [chroma[:, start:max(stop, start + 1)].mean(axis=1)
         for start, stop in zip(edges[:-1], edges[1:])],
        axis=1,
    )


def normalize_frames(chroma: np.ndarray) -> np.ndarray:
    """Chroma (12, F) -> descriptor (12 * F,): trừ trung bình từng khung rồi chuẩn hoá L2."""
    # Chroma không âm nên cosine giữa hai vector bất kỳ đã cao sẵn, và khung phẳng
    # (nhiễu, không có hoà âm) gần mọi bài: nhiễu trắng từng đạt 0.985 > τCover.
    # Trừ trung bình đưa khung phẳng về 0; phép này bất biến theo tỉ lệ từng khung.
    centered = chroma - chroma.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0, keepdims=True)
    centered = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 1e-8)

    flat = centered.reshape(-1)
    total = np.linalg.norm(flat)
    return (flat / total).astype("float32") if total > 1e-8 else flat.astype("float32")


def build_descriptor(audio: np.ndarray, sr: int = CHROMA_SR) -> np.ndarray:
    """
    Đặc trưng cover của một đoạn: vector (12 * DESCRIPTOR_FRAMES,) đã chuẩn hoá L2.

    Giữ nguyên trục 12 bậc ở chiều ĐẦU khi trải phẳng, để phép xoay cao độ vẫn là
    một phép `np.roll` đơn giản trên ma trận (12, F) trước khi trải.
    """
    return normalize_frames(_resample_time(extract_chroma(audio, sr)))


def descriptor_from_file(audio_path: str, offset: float = 0.0,
                         duration: float = None) -> np.ndarray:
    audio, sr = librosa.load(audio_path, sr=CHROMA_SR, mono=True,
                             offset=offset, duration=duration)
    return build_descriptor(audio, sr)


def transpositions(descriptor: np.ndarray) -> np.ndarray:
    """
    Trả (12, D): descriptor sau cả 12 phép dịch cao độ.

    Dòng r ứng với "đoạn này bị dịch lên r bán cung". Tính sẵn một lần rồi nhân
    ma trận với toàn bộ reference sẽ rẻ hơn nhiều so với lặp 12 lần cho từng ứng viên.
    """
    matrix = descriptor.reshape(N_CHROMA, -1)
    return np.stack([np.roll(matrix, r, axis=0).reshape(-1) for r in range(N_CHROMA)])


def cover_similarity(query_descriptor: np.ndarray,
                     reference_descriptor: np.ndarray) -> tuple:
    """
    Trả (điểm cao nhất, OTI). OTI là số bán cung phải dịch truy vấn lên để khớp
    reference tốt nhất — trả kèm để kết quả giải thích được, không phải hộp đen (§2).
    """
    scores = transpositions(query_descriptor) @ reference_descriptor
    best = int(np.argmax(scores))
    return float(scores[best]), best


def search(query_descriptor: np.ndarray, reference_matrix: np.ndarray,
           top_k: int = 5, excluded_columns: list = None) -> list:
    """
    Tìm Top-K trong ma trận reference (N, D).

    Trả danh sách dict đã xếp hạng giảm dần, mỗi phần tử kèm `index`,
    `similarity_score` và `oti`. `excluded_columns`: các cột coi như không có.
    """
    if reference_matrix is None or len(reference_matrix) == 0:
        return []

    # (12, D) @ (D, N) -> (12, N): điểm của mọi ứng viên ở mọi phép dịch cao độ
    scores = transpositions(query_descriptor) @ reference_matrix.T
    best_oti = scores.argmax(axis=0)
    best_scores = scores.max(axis=0)
    if excluded_columns:
        best_scores[excluded_columns] = -np.inf

    order = [i for i in np.argsort(-best_scores)[:max(1, top_k)]
             if np.isfinite(best_scores[i])]
    return [
        {"index": int(i),
         "similarity_score": float(best_scores[i]),
         "oti": int(best_oti[i])}
        for i in order
    ]


def identify_cover(audio_path: str, cover_index: CoverIndex = None,
                   candidate_recordings: list = None,
                   top_k: int = 5,
                   offset: float = 0.0,
                   duration: float = 30.0,
                   exclude_recording_ids: frozenset = None) -> dict:
    """
    Nhận diện phiên bản/cover bằng CQT chroma + OTI.
    Trả kết quả gồm matched, best_match, candidates, oti, threshold.

    `exclude_recording_ids`: che các bản ghi này khỏi chỉ mục (giao thức held-out).
    """
    excluded = exclude_recording_ids or frozenset()
    try:
        query_desc = descriptor_from_file(audio_path, offset=offset, duration=duration)
    except Exception as e:
        return {
            "matched": False,
            "best_match": None,
            "candidates": [],
            "error": str(e),
            "threshold": config.COVER_THRESHOLD,
        }

    index = cover_index if cover_index is not None else get_default_cover_index()
    candidates = []

    if index is not None and index.matrix is not None and len(index.matrix) > 0:
        excluded_columns = ([column for column, rec in enumerate(index.id_map)
                             if str(rec) in excluded] if excluded else None)
        raw_matches = search(query_desc, index.matrix, top_k=top_k,
                             excluded_columns=excluded_columns)
        for m in raw_matches:
            idx = m["index"]
            rec_id = index.id_map[idx] if (index.id_map and idx < len(index.id_map)) else str(idx)
            candidates.append({
                "recording_id": rec_id,
                "similarity_score": round(float(m["similarity_score"]), 4),
                "oti": int(m["oti"]),
            })
    elif candidate_recordings:
        # Nếu chưa có toàn bộ ma trận index, nhưng có danh sách candidate từ MERT:
        # So khớp trực tiếp với các candidate nếu có audio trên máy
        for cand in candidate_recordings:
            rec_id = cand.get("recording_id")
            if str(rec_id) in excluded:
                continue
            audio_file = cand.get("audio_path")
            if not audio_file and rec_id:
                audio_file = config.resolve_audio_path(f"fma/{rec_id}.mp3")
            if audio_file and os.path.exists(audio_file):
                try:
                    ref_desc = descriptor_from_file(audio_file, offset=offset, duration=duration)
                    sim, oti = cover_similarity(query_desc, ref_desc)
                    candidates.append({
                        "recording_id": rec_id,
                        "similarity_score": round(float(sim), 4),
                        "oti": int(oti),
                    })
                except Exception:
                    continue
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

    best = candidates[0] if candidates else None
    matched = bool(best and best["similarity_score"] >= config.COVER_THRESHOLD)

    return {
        "matched": matched,
        "best_match": best if matched else None,
        "top_candidate": best,
        "candidates": candidates[:top_k],
        "threshold": config.COVER_THRESHOLD,
    }

