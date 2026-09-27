"""
Đặc trưng "đã học" của sổ bài chưa nhận diện: vector MERT từng đoạn + descriptor Cover.

Chủ dự án muốn (2026-09-27) mỗi bài UNKNOWN được hệ thống ghi nhớ để lần sau nhận ra
được cả bản đã nén, cắt, đổi tông của nó. Ở đây "học" là GHI NHỚ đặc trưng rồi tìm lại,
đúng cách kho tham chiếu chính hoạt động. Không huấn luyện model (CLAUDE.md §2.9). Bài
chưa từng gặp thì không cách nào nhận ra được.

Đặc trưng tính y như chỉ mục chính, để ngưỡng đã hiệu chỉnh còn đúng nghĩa:
  - MERT: đoạn 15 s, bước 10 s (khớp `scripts/build_embeddings.py`), mean pooling,
    chuẩn hoá L2. So bằng cosine, nhận khi >= τMERT (EXP-06).
  - Cover: 30 giây đầu ở nhịp gốc làm reference; truy vấn cắt theo
    `COVER_TEMPO_FACTORS` (khớp `cover_service.identify_cover`). Nhận khi >= τCover (EXP-07).

Không lưu audio: vector mean-pooled và chroma 64 khung không dựng lại được âm thanh.
"""
import math
import threading

import numpy as np

from backend import config
from backend.services import cover_service, embedding_service

MERT = "MERT"
COVER = "COVER"

MERT_WINDOW_S = 15.0   # scripts/build_embeddings.py: WINDOW_S
MERT_HOP_S = 10.0      # scripts/build_embeddings.py: HOP_S
COVER_BASE_S = 30.0    # reference Cover = 30 giây đầu ở nhịp gốc, như chỉ mục chính
MAX_COVER_PER_ENTRY = 8   # mỗi lần gửi một descriptor (30 giây đầu của lần gửi đó)

COVER_VERSION = f"chroma_cqt_{cover_service.DESCRIPTOR_FRAMES}f_meansub_oti_v1"


def mert_version() -> str:
    """Phiên bản đặc trưng MERT (§9): đổi model/revision/cách cắt thì vector cũ không so được."""
    revision = (getattr(config, "MERT_MODEL_REVISION", "") or "latest")[:12]
    return f"{config.MERT_MODEL_VERSION}@{revision}/mean/{int(MERT_WINDOW_S)}s"


def plan_segments(duration: float, max_segments: int = None) -> list:
    """
    [(start, end)] các đoạn MERT của một file. Bài ngắn: đúng lưới 15 s / bước 10 s của
    chỉ mục chính. Bài dài hơn `max_segments` đoạn: rải đều từ đầu tới đuôi bài, để lần
    sau một đoạn cắt ở giữa hay cuối bài vẫn có đoạn tham chiếu gần nó.
    """
    max_segments = max(1, int(max_segments or config.UNKNOWN_REGISTRY_MAX_SEGMENTS))
    if duration < 1.0:
        return []
    if duration <= MERT_WINDOW_S:
        return [(0.0, round(float(duration), 2))]
    tail = duration - MERT_WINDOW_S
    count = int(math.floor(tail / MERT_HOP_S + 1e-9)) + 1
    if count <= max_segments:
        starts = [i * MERT_HOP_S for i in range(count)]
    else:
        starts = np.linspace(0.0, tail, num=max_segments).tolist()
    return [(round(float(s), 2), round(float(min(s + MERT_WINDOW_S, duration)), 2))
            for s in starts]


NOISE_KINDS = ("white", "pink", "brown")
_noise_lock = threading.Lock()
_noise_matrix = None


def _colored_noise(kind: str, n: int, rng) -> np.ndarray:
    white = rng.normal(size=n)
    if kind == "white":
        signal = white
    else:
        spectrum = np.fft.rfft(white)
        freqs = np.arange(len(spectrum), dtype=float)
        freqs[0] = 1.0
        spectrum /= np.sqrt(freqs) if kind == "pink" else freqs   # 1/f và 1/f²
        signal = np.fft.irfft(spectrum, n)
    return (0.3 * signal / np.abs(signal).max()).astype(np.float32)


def noise_prototypes() -> np.ndarray:
    """
    Vector MERT (3, 768) của tiếng ồn trắng / hồng / nâu 15 s, tính một lần mỗi tiến trình.

    Lý do (đo 2026-09-27 với model thật): hai đoạn ồn trắng KHÁC NHAU có cosine MERT 0,998,
    cao hơn cả một bài và bản đổi nhịp 1,05× của chính nó (0,983). Chỉ mục chính chỉ chứa
    nhạc nên không gặp chuyện này. Còn sổ học mọi thứ người dùng gửi, nên ồn sẽ "nhận ra"
    ồn khác. Đoạn nào gần một mẫu ồn tới mức >= τMERT thì MERT không phân biệt nổi nó với
    tiếng ồn ở đúng điểm làm việc đã hiệu chỉnh, nên không được dùng để học hay so.
    """
    global _noise_matrix
    with _noise_lock:
        if _noise_matrix is None:
            sr = config.MERT_SAMPLE_RATE
            rng = np.random.default_rng(0)
            _noise_matrix = np.vstack([
                embedding_service.embed_signal(_colored_noise(kind, int(MERT_WINDOW_S * sr), rng), sr)
                for kind in NOISE_KINDS]).astype(np.float32)
    return _noise_matrix


def extract_features(audio_path: str) -> dict:
    """
    Đặc trưng của một file audio ĐÃ GIẢI MÃ ĐƯỢC bằng librosa (m4a/video: truyền bản
    FFmpeg đã tách). Trả {"duration", "mert": [(start, end, vec)], "noise_like_segments",
    "cover_ref", "cover_queries": (ma trận, hệ số nhịp)}; Cover là None nếu không tính được.
    Đoạn giống tiếng ồn (`noise_prototypes`) bị bỏ khỏi "mert", chỉ được đếm.
    """
    import librosa

    sr = config.MERT_SAMPLE_RATE
    audio, _ = librosa.load(audio_path, sr=sr, mono=True, duration=config.MAX_MEDIA_SECONDS)
    duration = len(audio) / sr

    mert, noise_like = [], 0
    segments = plan_segments(duration)
    prototypes = noise_prototypes() if segments else None
    for start, end in segments:
        vector = np.asarray(
            embedding_service.embed_signal(audio[int(start * sr):int(end * sr)], sr), dtype=np.float32)
        if float((prototypes @ vector).max()) >= config.MERT_THRESHOLD:
            noise_like += 1
            continue
        mert.append((start, end, vector))
    del audio

    cover_ref, cover_queries = None, None
    if not config.COVER_ENABLED:   # tắt tầng Cover thì sổ cũng không học/so bằng Cover
        return {"duration": round(duration, 2), "mert": mert, "noise_like_segments": noise_like,
                "cover_ref": None, "cover_queries": None}
    try:
        longest = max(seconds for _, seconds in cover_service.query_spans(COVER_BASE_S))
        chroma_audio, chroma_sr = librosa.load(audio_path, sr=cover_service.CHROMA_SR,
                                               mono=True, duration=longest)
        cover_ref = cover_service.build_descriptor(
            chroma_audio[:int(COVER_BASE_S * chroma_sr)], chroma_sr).astype(np.float32)
        cover_queries = cover_service.query_descriptors(chroma_audio, chroma_sr, COVER_BASE_S)
    except Exception:
        # Đoạn quá ngắn / không trích được chroma: vẫn học được phần MERT
        cover_ref, cover_queries = None, None

    return {"duration": round(duration, 2), "mert": mert, "noise_like_segments": noise_like,
            "cover_ref": cover_ref, "cover_queries": cover_queries}


def to_bytes(vector) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def from_bytes(raw, dimension: int) -> np.ndarray:
    vector = np.frombuffer(bytes(raw), dtype="<f4")
    if dimension and vector.size != dimension:
        raise ValueError(f"vector {vector.size} chiều, mong đợi {dimension}")
    return vector


def _stack(references):
    owners = [owner for owner, _ in references]
    matrix = np.vstack([np.asarray(vec, dtype=np.float32) for _, vec in references])
    return owners, matrix


def best_mert_match(query_vectors, references, threshold: float = None):
    """
    Mục khớp nhất theo MERT: `(owner, điểm)` hoặc None.

    `query_vectors`: (k, D) các đoạn của lần gửi này. `references`: [(owner, vec D)].
    Điểm của một mục = cosine lớn nhất qua mọi cặp (đoạn truy vấn, đoạn đã lưu) — như
    `search_recordings` lấy max qua các segment của một bản ghi.
    """
    threshold = config.MERT_THRESHOLD if threshold is None else threshold
    if query_vectors is None or len(query_vectors) == 0 or not references:
        return None
    owners, matrix = _stack(references)
    per_reference = (np.atleast_2d(query_vectors).astype(np.float32) @ matrix.T).max(axis=0)
    best = int(np.argmax(per_reference))
    score = float(per_reference[best])
    return (owners[best], score) if score >= threshold else None


def novel_features(features: dict, stored_mert: list, stored_cover: list,
                   mert_room: int, cover_room: int) -> tuple:
    """
    (các đoạn MERT mới, descriptor Cover mới hoặc None) đáng lưu thêm cho một mục.

    "Mới" = chưa nhận ra được bằng những gì mục đã có, theo đúng ngưỡng nhận lại: đoạn có
    cosine >= τMERT với một vector đã lưu (hoặc với đoạn vừa nhận trong cùng lượt) thì bỏ;
    descriptor có điểm Cover >= τCover với một descriptor đã lưu thì bỏ. Gửi lại đúng một
    file không làm sổ phình ra, còn bản cắt ở đoạn khác / bản đổi tông thì được học thêm.
    """
    accepted, known = [], [np.asarray(v, dtype=np.float32) for v in stored_mert]
    for start, end, vector in features.get("mert") or []:
        if len(accepted) >= mert_room:
            break
        if known and float((np.vstack(known) @ vector).max()) >= config.MERT_THRESHOLD:
            continue
        accepted.append((start, end, vector))
        known.append(np.asarray(vector, dtype=np.float32))

    cover = features.get("cover_ref")
    if cover is None or cover_room <= 0:
        return accepted, None
    if stored_cover:
        scores, _oti, _rows = cover_service.best_scores(cover, np.vstack(stored_cover))
        if float(scores.max()) >= config.COVER_THRESHOLD:
            return accepted, None
    return accepted, cover


def best_cover_match(query_descriptors, references, threshold: float = None):
    """
    Mục khớp nhất theo Cover: `(owner, điểm, oti, dòng truy vấn)` hoặc None.

    Dùng đúng `cover_service.best_scores` (max qua 12 phép xoay cao độ và mọi độ dài cắt
    theo nhịp độ) — cách tầng Cover chấm chỉ mục chính.
    """
    threshold = config.COVER_THRESHOLD if threshold is None else threshold
    if query_descriptors is None or len(query_descriptors) == 0 or not references:
        return None
    owners, matrix = _stack(references)
    scores, oti, rows = cover_service.best_scores(query_descriptors, matrix)
    best = int(np.argmax(scores))
    score = float(scores[best])
    if not np.isfinite(score) or score < threshold:
        return None
    return owners[best], score, int(oti[best]), int(rows[best])
