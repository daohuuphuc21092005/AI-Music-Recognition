"""
Tầng 1 của phễu cascade: nhận diện chính xác bằng Chromaprint.

Hai lỗi đã được sửa ở đây (xem plan Giai đoạn 0 mục D):

1. `acoustid.compare_fingerprints(a, b)` nhận CẶP `(duration, fingerprint)`,
   code cũ truyền thẳng chuỗi nên `a[1]` là ký tự thứ 2 của chuỗi -> luôn ném
   lỗi -> bị `except: return 0.0` nuốt mất -> tầng 1 KHÔNG BAO GIỜ khớp.
2. Mỗi truy vấn giải nén lại toàn bộ ~2.650 fingerprint trong DB, và hàm so
   khớp thuần Python của pyacoustid chạy O(n·240) -> hàng chục phút/truy vấn.
   Nay: giải nén có cache + so khớp vector hoá bằng numpy (cùng thuật toán,
   cùng hằng số; `tests/test_fingerprint_match.py` chứng minh hai bản cho kết
   quả trùng khớp).

Việc giải nén dùng `backend.services.chromaprint_codec` (thuần Python) thay cho
thư viện native `libchromaprint` — trên Windows x64 không có gói nào cung cấp
thư viện đó. Chỉ còn phụ thuộc duy nhất là binary `fpcalc` để tạo fingerprint.

⚠️ ĐIỂM SỐ PHỤ THUỘC ĐỘ DÀI TRUY VẤN — đọc trước khi chọn τFP

Điểm Chromaprint là tỉ lệ bit trùng khớp ở offset căn chỉnh tốt nhất. Truy vấn
càng NGẮN thì càng ít offset để dò, nên càng dễ gặp một offset "may mắn" — tức
điểm nền của một truy vấn KHÔNG hề có trong CSDL tăng lên khi đoạn ngắn đi.

Đo trên reference 4.000 bản ghi, dùng nhiễu trắng (chắc chắn không có trong DB),
trung bình 5 seed:

    độ dài   điểm TB   điểm max
      5 s     0.2842    0.3684
      8 s     0.1628    0.2558
     10 s     0.1288    0.2034
     15 s     0.1120    0.1800
     20 s     0.0829    0.1286
     30 s     0.0579    0.0814

Hệ quả trực tiếp: **một τFP thấp chỉ an toàn cho truy vấn dài**. Với τFP = 0.10,
đoạn 30 giây là an toàn (điểm nền tối đa 0.0814) nhưng đoạn 8 giây thì nhiễu
trắng đã vượt ngưỡng và bị gán EXACT_MATCH. Ngưỡng hiệu chỉnh bằng EXP-01 trên
tập truy vấn 30 giây KHÔNG chuyển thẳng sang truy vấn ngắn được.
"""
import os
import shutil
from functools import lru_cache

import acoustid
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend import config
from backend.services.chromaprint_codec import (
    InvalidFingerprintError,
    decode_fingerprint,
)

# Hằng số lấy trực tiếp từ pyacoustid để bản vector hoá không bị lệch chuẩn
MAX_ALIGN_OFFSET = acoustid.MAX_ALIGN_OFFSET   # 120
MAX_BIT_ERROR = acoustid.MAX_BIT_ERROR         # 2

# CẢNH BÁO: ngưỡng tạm, phải hiệu chỉnh bằng EXP-01 (CLAUDE.md §4).
FP_THRESHOLD = config.FP_THRESHOLD


class FingerprintBackendUnavailable(RuntimeError):
    """Thiếu fpcalc hoặc libchromaprint -> tầng 1 không thể hoạt động."""


class AudioFingerprintError(RuntimeError):
    """Không trích xuất được fingerprint từ file (file hỏng/không phải audio)."""


# Các vị trí thường gặp của fpcalc trên máy dev, dò khi PATH không có. Cần thiết
# vì PATH của shell và PATH của tiến trình server KHÁC NHAU: Git Bash tự thêm
# ~/bin vào PATH nên `which fpcalc` chạy được trong terminal, trong khi uvicorn
# khởi động từ Windows lại không thấy — và /health báo Chromaprint MISSING dù
# binary vẫn nằm ngay trên máy.
_FPCALC_CANDIDATES = (
    os.path.join(os.path.expanduser("~"), "bin", "fpcalc.exe"),
    os.path.join(os.path.expanduser("~"), "bin", "fpcalc"),
    os.path.join(os.path.expanduser("~"), ".cache", "music-rights-ai", "fpcalc.exe"),
    os.path.join(os.path.expanduser("~"), ".cache", "music-rights-ai", "fpcalc"),
)


@lru_cache(maxsize=1)
def find_fpcalc() -> str:
    """
    Đường dẫn tới fpcalc, hoặc chuỗi rỗng nếu không có ở đâu cả.

    Thứ tự: biến môi trường FPCALC_PATH (cho phép chỉ định tường minh) -> PATH
    -> các vị trí quen thuộc. Kết quả được cache vì đây là thao tác chạm đĩa và
    câu trả lời không đổi trong một vòng đời tiến trình.
    """
    explicit = os.environ.get("FPCALC_PATH", "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit

    found = shutil.which("fpcalc")
    if found:
        return found

    for candidate in _FPCALC_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return ""


def backend_status() -> dict:
    """Trạng thái phụ thuộc ngoài, dùng cho endpoint /health."""
    fpcalc = find_fpcalc()
    return {
        "fpcalc": bool(fpcalc),
        "fpcalc_path": fpcalc or None,
        "libchromaprint": bool(getattr(acoustid, "have_chromaprint", False)),
        "decoder": "pure-python (backend.services.chromaprint_codec)",
    }


def is_available() -> bool:
    """Chỉ cần một trong hai cách tạo fingerprint; giải nén luôn sẵn sàng."""
    status = backend_status()
    return status["fpcalc"] or status["libchromaprint"]


# --------------------------------------------------------------------------
# Giải nén fingerprint (có cache)
# --------------------------------------------------------------------------
@lru_cache(maxsize=8192)
def _decode(fp: bytes) -> tuple:
    """Giải nén fingerprint base64 -> tuple số nguyên. Cache theo chuỗi gốc."""
    ints, _algorithm = decode_fingerprint(fp)
    return tuple(ints)


def _as_bytes(fp) -> bytes:
    if isinstance(fp, bytes):
        return fp
    return str(fp).encode("utf-8")


def _decode_array(fp) -> np.ndarray:
    return np.asarray(_decode(_as_bytes(fp)), dtype=np.uint32)


# --------------------------------------------------------------------------
# So khớp
# --------------------------------------------------------------------------
def match_decoded(a: np.ndarray, b: np.ndarray, max_align_offset: int = None) -> float:
    """
    Điểm khớp Chromaprint: với mỗi độ lệch d = i - j, đếm số cặp item có
    bit-error <= MAX_BIT_ERROR, lấy giá trị lớn nhất rồi chia cho min(len).

    `max_align_offset`:
      * None hoặc <= 0 -> DÒ TOÀN BỘ độ lệch (mặc định của hệ thống này)
      * 120            -> đúng dải của `acoustid._match_fingerprints`

    Vì sao mặc định là dò toàn bộ: pyacoustid chỉ dò ±120 item ≈ ±15 giây, nên
    một đoạn cắt từ giữa bài KHÔNG THỂ khớp. EXP-01 đo được điểm của cùng một
    cặp nhảy từ 0.0498 lên 0.8688 khi mở rộng dải dò (điểm khớp thật nằm ở
    offset -242 item = đúng giây thứ 30 của bài).

    Cách tính: dựng ma trận so khớp rồi cộng theo từng đường chéo bằng
    `np.bincount`. Cách này vừa CHÍNH XÁC vừa nhanh hơn vòng lặp theo offset
    (~1.4 ms/cặp cho fingerprint 120 giây).
    """
    asize, bsize = int(a.size), int(b.size)
    if asize == 0 or bsize == 0:
        return 0.0

    if max_align_offset is None:
        max_align_offset = config.FP_MAX_ALIGN_OFFSET

    counts = np.zeros(asize + bsize + 1, dtype=np.int64)
    block = 1024  # chặn bộ nhớ: mỗi lần chỉ dựng ma trận block x bsize
    for start in range(0, asize, block):
        chunk = a[start:start + block]
        matched = np.bitwise_count(chunk[:, None] ^ b[None, :]) <= MAX_BIT_ERROR
        rows, cols = np.nonzero(matched)
        offsets = (rows + start) - cols + bsize
        counts += np.bincount(offsets, minlength=counts.size)

    if max_align_offset and max_align_offset > 0:
        # Giới hạn dải d về [-(M-1), M] rồi mới lấy max
        low = bsize - (max_align_offset - 1)
        high = bsize + max_align_offset
        window = counts[max(0, low):min(counts.size, high + 1)]
        best = int(window.max()) if window.size else 0
    else:
        best = int(counts.max())

    return best / min(asize, bsize)


def calculate_similarity(fp_query, fp_db) -> float:
    """So sánh hai fingerprint base64 (chuỗi hoặc bytes), trả điểm [0, 1]."""
    return match_decoded(_decode_array(fp_query), _decode_array(fp_db))


# --------------------------------------------------------------------------
# Trích xuất & tra cứu
# --------------------------------------------------------------------------
def extract_query_fingerprint(audio_path: str):
    """Trả (duration, fingerprint_bytes). Ném lỗi thay vì trả (None, None)."""
    # pyacoustid tìm binary qua biến môi trường FPCALC rồi mới tới PATH. Gán nó
    # từ đường dẫn ta tự dò được, để tầng 1 vẫn chạy khi tiến trình server có
    # PATH khác với shell (xem find_fpcalc).
    fpcalc = find_fpcalc()
    if fpcalc and not os.environ.get("FPCALC"):
        os.environ["FPCALC"] = fpcalc

    try:
        duration, fp = acoustid.fingerprint_file(audio_path)
    except acoustid.NoBackendError as e:
        raise FingerprintBackendUnavailable(
            "Không tìm thấy fpcalc. Tải binary chính thức tại "
            "https://github.com/acoustid/chromaprint/releases rồi đặt vào PATH, "
            "hoặc trỏ thẳng bằng biến môi trường FPCALC_PATH."
        ) from e
    except acoustid.FingerprintGenerationError as e:
        raise AudioFingerprintError(f"Không đọc được luồng audio: {e}") from e
    return duration, _as_bytes(fp)


# Điểm nền của một truy vấn KHÔNG có trong CSDL, đo bằng nhiễu trắng trên
# reference 4.000 bản ghi (10 seed mỗi độ dài). Cột dùng ở đây là GIÁ TRỊ LỚN
# NHẤT quan sát được, không phải trung bình: ngưỡng phải nằm trên cả trường hợp
# xấu nhất thì mới thực sự chặn được dương tính giả.
#
#     độ dài   TB       max      p95
#       5 s    0.2632   0.3684   0.3684
#       8 s    0.1535   0.2558   0.2244
#      10 s    0.1237   0.2034   0.1729
#      15 s    0.0990   0.1800   0.1575
#      20 s    0.0757   0.1286   0.1125
#      25 s    0.0644   0.1000   0.0925
#      30 s    0.0593   0.0814   0.0774
NOISE_FLOOR_BY_DURATION = (
    (5.0, 0.3684),
    (8.0, 0.2558),
    (10.0, 0.2034),
    (15.0, 0.1800),
    (20.0, 0.1286),
    (25.0, 0.1000),
    (30.0, 0.0814),
)

# Biên an toàn trên điểm nền đo được. Bảng trên lấy từ 10 seed nên giá trị lớn
# nhất THẬT còn cao hơn; 1.15 đủ để không bám sát mép mà cũng không siết tới mức
# vứt bỏ các truy vấn ngắn hợp lệ.
NOISE_FLOOR_MARGIN = 1.15


def min_score_for_duration(duration: float) -> float:
    """
    Ngưỡng tối thiểu để chấp nhận một khớp, theo ĐỘ DÀI của truy vấn.

    Vì sao cần: điểm Chromaprint là tỉ lệ bit trùng ở offset căn chỉnh tốt nhất.
    Đoạn càng ngắn thì càng ít offset để dò, nên càng dễ gặp một offset "may
    mắn" — điểm nền của truy vấn KHÔNG hề có trong CSDL tăng vọt khi đoạn ngắn
    đi. Một τFP duy nhất vì thế không thể vừa an toàn cho đoạn 8 giây vừa không
    quá khắt khe với đoạn 30 giây: đo được nhiễu trắng 8 giây đạt tới 0.2558,
    trong khi 30 giây chỉ 0.0814.

    τFP hiệu chỉnh bằng EXP-01 (trên tập truy vấn 30 giây) vẫn là sàn; hàm này
    chỉ NÂNG ngưỡng lên khi truy vấn ngắn, không bao giờ hạ xuống.
    """
    if not duration or duration <= 0:
        return FP_THRESHOLD

    points = NOISE_FLOOR_BY_DURATION
    if duration <= points[0][0]:
        floor = points[0][1]
    elif duration >= points[-1][0]:
        floor = points[-1][1]
    else:
        floor = points[-1][1]
        for (d0, f0), (d1, f1) in zip(points, points[1:]):
            if d0 <= duration <= d1:
                ratio = (duration - d0) / (d1 - d0)
                floor = f0 + ratio * (f1 - f0)
                break

    return max(FP_THRESHOLD, floor * NOISE_FLOOR_MARGIN)


_reference_cache = {"signature": None, "rows": [], "undecodable": 0}


def _reference_fingerprints(db: Session) -> tuple:
    """
    ([(recording_id, vector, duration)], số dòng không giải nén được), cache theo tiến trình.

    Tải + giải nén lại cả bảng ở mỗi truy vấn đo được ~1,3 ms/dòng — hơn 35 giây ở
    28.000 fingerprint, gấp nhiều lần phần so khớp. fingerprint_id là UUID sinh mới
    mỗi lần nạp dữ liệu, nên (số dòng, min id, max id) đổi khi bảng được nạp lại.
    """
    signature = tuple(db.execute(text(
        "SELECT count(*), min(fingerprint_id::text), max(fingerprint_id::text) FROM fingerprints"
    )).fetchone())
    if _reference_cache["signature"] != signature:
        rows = db.execute(
            text("SELECT recording_id, fingerprint, duration FROM fingerprints")
        ).fetchall()
        decoded, undecodable = [], 0
        for rec_id, db_fp, db_duration in rows:
            try:
                vector = np.asarray(decode_fingerprint(_as_bytes(db_fp))[0], dtype=np.uint32)
            except (InvalidFingerprintError, ValueError):
                undecodable += 1  # một dòng hỏng không được làm sập cả truy vấn
                continue
            decoded.append((str(rec_id), vector, db_duration))
        _reference_cache.update(signature=signature, rows=decoded, undecodable=undecodable)
    return _reference_cache["rows"], _reference_cache["undecodable"]


def search_fingerprint(db: Session, audio_path: str,
                       exclude_recording_ids: frozenset = None) -> dict:
    """
    Quét tuyến tính mọi fingerprint tham chiếu (đã giải nén sẵn, cache theo tiến trình).

    Trả EXACT_MATCH khi điểm cao nhất >= FP_THRESHOLD, ngược lại NO_MATCH kèm
    điểm tốt nhất để tầng 2 (MERT) tiếp quản.

    `exclude_recording_ids`: bỏ qua các bản ghi này như thể chúng không có trong
    CSDL (giao thức held-out của thí nghiệm). Lọc trên danh sách đã giải nén chứ
    không đụng câu SQL, nên bộ đệm reference dùng chung vẫn đúng. None = production.
    """
    query_duration, query_fp = extract_query_fingerprint(audio_path)
    query_vec = _decode_array(query_fp)

    references, undecodable = _reference_fingerprints(db)

    window = config.FP_DURATION_WINDOW_S
    best_match_id, best_score = None, 0.0
    compared, skipped_by_duration = 0, 0

    excluded = exclude_recording_ids or ()
    for rec_id, db_vec, db_duration in references:
        if rec_id in excluded:
            continue
        # Lọc theo độ dài (mặc định TẮT: window = 0) — bật lên sẽ nhanh hơn
        # nhiều nhưng có thể ảnh hưởng recall với truy vấn bị cắt (crop).
        if window > 0 and db_duration and query_duration:
            if abs(float(db_duration) - float(query_duration)) > window:
                skipped_by_duration += 1
                continue

        score = match_decoded(query_vec, db_vec)
        compared += 1
        if score > best_score:
            best_score, best_match_id = score, rec_id

    if undecodable:
        print(f"⚠️ Bỏ qua {undecodable} fingerprint không giải nén được trong DB")

    # Ngưỡng hiệu dụng = max(τFP đã hiệu chỉnh, điểm nền theo độ dài truy vấn)
    effective_threshold = min_score_for_duration(query_duration)

    result = {
        "recording_id": best_match_id if best_score >= effective_threshold else None,
        "fingerprint_score": best_score,
        "threshold": effective_threshold,
        "threshold_base": FP_THRESHOLD,
        "threshold_raised_for_short_query": effective_threshold > FP_THRESHOLD,
        "candidates_compared": compared,
        "candidates_skipped_by_duration": skipped_by_duration,
        "query_duration": query_duration,
    }

    if best_score >= effective_threshold:
        result["match_type"] = "EXACT_MATCH"
    else:
        result["match_type"] = "NO_MATCH"
        result["message"] = (
            f"Dưới ngưỡng Chromaprint ({effective_threshold:.4f}"
            + (f", nâng từ {FP_THRESHOLD} vì truy vấn chỉ {query_duration:.1f}s"
               if effective_threshold > FP_THRESHOLD else "")
            + "), chuyển sang MERT Retrieval."
        )
        result["best_candidate_below_threshold"] = best_match_id
    return result
