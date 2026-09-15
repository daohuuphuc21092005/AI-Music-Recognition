"""
Cấu hình tập trung cho toàn hệ thống (đọc từ file .env ở thư mục gốc).

Nguyên tắc: KHÔNG hardcode DSN / đường dẫn / ngưỡng trong bất kỳ file nào khác.
Mọi module (backend, scripts, tests) đều import từ đây.
"""
import os
import sys
from pathlib import Path

# Console Windows mặc định dùng cp1252 -> mọi lệnh print tiếng Việt/emoji trong
# project sẽ ném UnicodeEncodeError. config.py được import bởi mọi entry point
# nên đặt việc chuẩn hoá stdout sang UTF-8 ở đây là chỗ rẻ và chắc chắn nhất.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

try:
    from dotenv import load_dotenv
except ImportError:  # Fallback tối giản nếu chưa cài python-dotenv
    def load_dotenv(path=None, override=False):
        if path is None or not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if override or key not in os.environ:
                    os.environ[key] = value
        return True

# Thư mục gốc của project (backend/config.py -> ..)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get(key: str, default: str) -> str:
    value = os.getenv(key)
    return default if value is None or value == "" else value


def _get_float(key: str, default: float) -> float:
    try:
        return float(_get(key, str(default)))
    except ValueError:
        return default


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get(key, str(default)))
    except ValueError:
        return default


def _path(key: str, default: str) -> str:
    """Trả về đường dẫn tuyệt đối, cho phép chạy script từ bất kỳ thư mục nào."""
    raw = _get(key, default)
    p = Path(raw)
    return str(p if p.is_absolute() else (BASE_DIR / p))


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
DATABASE_URL = _get(
    "DATABASE_URL",
    "postgresql://postgres:postgrespassword@localhost:5432/music_rights_ai",
)

# --------------------------------------------------------------------------
# Đường dẫn dữ liệu
# --------------------------------------------------------------------------
DATA_DIR = _path("DATA_DIR", "data/processed")
FAISS_INDEX_PATH = _path("FAISS_INDEX_PATH", "data/processed/mert_faiss.index")
FAISS_ID_MAP_PATH = _path("FAISS_ID_MAP_PATH", "data/processed/faiss_id_map.json")
EMBEDDINGS_CSV = _path("EMBEDDINGS_CSV", "data/processed/embeddings_master.csv")
METADATA_CSV = _path("METADATA_CSV", "data/processed/metadata_master.csv")
TEMP_UPLOAD_DIR = _path("TEMP_UPLOAD_DIR", "temp_uploads")

# Thư mục gốc chứa audio THẬT của corpus tham chiếu. Tách khỏi repo vì corpus vài
# nghìn bài nặng nhiều GB: không nên nằm trong git, và ổ hệ thống thường không đủ
# chỗ (đặt sang ổ khác qua .env).
AUDIO_ROOT = _path("AUDIO_ROOT", "data/raw")

def resolve_audio_path(audio_path: str) -> str:
    """
    Đổi `recordings.audio_path` thành đường dẫn tuyệt đối trên máy đang chạy.

    CSDL lưu đường dẫn TƯƠNG ĐỐI vì đường dẫn tuyệt đối sẽ chết ngay khi đổi máy
    hoặc chạy trong container. Thử AUDIO_ROOT trước (corpus thật), rồi tới thư
    mục gốc repo (vài file audio đi kèm repo như test.mp3).

    Trả chuỗi rỗng nếu không tìm thấy ở đâu cả — người gọi phải xử lý, tuyệt đối
    không được im lặng coi như file tồn tại.
    """
    if not audio_path:
        return ""
    if os.path.isabs(audio_path):
        return audio_path if os.path.exists(audio_path) else ""

    for root in (AUDIO_ROOT, str(BASE_DIR)):
        candidate = os.path.join(root, audio_path)
        if os.path.exists(candidate):
            return candidate
    return ""


# --------------------------------------------------------------------------
# Ngưỡng nhận diện
#
# HIỆU CHỈNH TRÊN CORPUS FMA THẬT: 4.000 bản ghi có fingerprint, 8.000 vector
# embedding, 1.900 truy vấn biến đổi. Bộ ngưỡng cũ (0.70 / 0.95) hiệu chỉnh
# trên đúng 2 file audio -> 36 truy vấn, và đo lại trên corpus thật cho thấy nó
# lệch CÓ HỆ THỐNG về phía lạc quan: tập nhỏ hiếm khi chứa cặp gây nhầm.
#
# FP_THRESHOLD = 0.15 — hiệu chỉnh bằng EXP-01 (Fingerprint Baseline),
# xem experiments/results/exp01_fingerprint_baseline.json:
#   τ = 0.05 -> P 0.9905 / R 0.6042 / F1 0.7506 / FPR 0.0216
#   τ = 0.10 -> P 0.9936 / R 0.5721 / F1 0.7261 / FPR 0.0095
#   τ = 0.15 -> P 0.9963 / R 0.5632 / F1 0.7196 / FPR 0.0032   <- đang dùng
#   τ = 0.70 -> P 0.9990 / R 0.5268 / F1 0.6899 / FPR 0.0005
#   Tiêu chí chọn, tường minh chứ không theo cảm tính: F1 cao nhất trong nhóm
#   giữ FPR <= 0.005. Trên corpus thật KHÔNG ngưỡng nào cho FPR = 0, nên tiêu
#   chí "FPR = 0" của bản cũ không còn dùng được nữa.
#   Giá trị cũ 0.70 không sai về precision, nhưng bỏ phí ~3,6 điểm recall để
#   đổi lấy vỏn vẹn 0.0027 FPR.
#
# MERT_THRESHOLD = 0.97 — hiệu chỉnh bằng EXP-06 (Unknown Track Detection),
# xem experiments/results/exp06_unknown_detection.json (8.000 truy vấn known +
# 8.000 truy vấn unknown):
#   τ = 0.90 -> False Match Rate 74.86%
#   τ = 0.95 -> False Match Rate 14.52%   (KHÔNG đạt ngưỡng <=5% của §16)
#   τ = 0.97 -> False Match Rate  4.40%, unknown recall 0.956   <- đang dùng
#   τ = 0.98 -> False Match Rate  1.84%, nhưng true accept tụt còn 0.155
#   Giá trị cũ 0.95 từng được ghi là "FMR 1.87%" khi reference chỉ có 114 bản
#   ghi. Trên 8.000 vector, CHÍNH ngưỡng đó cho 14.52% — gấp gần 8 lần.
#
# NGƯỠNG PHỤ THUỘC QUY MÔ REFERENCE. Đây là quan hệ ĐO ĐƯỢC, không phải suy
# đoán: ở cùng τFP = 0.10, FPR tăng từ 0.0044 (1.000 bản ghi) lên 0.0095 (4.000
# bản ghi). Còn τMERT = 0.97 hiện cho FMR 4.40%, tức ĐÃ SÁT trần 5% của §16.
# Mở rộng corpus thêm nữa thì BẮT BUỘC chạy lại EXP-01 và EXP-06 trước khi tin
# hai con số này.
#
# PHẢI khớp với `min_identity_confidence_by_match_type` trong
# configs/rules_v1.yaml (EXACT_MATCH = τFP, NEAR_MATCH = τMERT). Lệch nhau thì
# cascade nhận một khớp rồi Rule Engine vứt chính khớp đó thành UNKNOWN.
# --------------------------------------------------------------------------
FP_THRESHOLD = _get_float("FP_THRESHOLD", 0.15)
# Bộ lọc sơ bộ theo độ dài trước khi so khớp fingerprint (giây).
# 0 = TẮT (mặc định) để không âm thầm ảnh hưởng recall với truy vấn bị cắt.
FP_DURATION_WINDOW_S = _get_float("FP_DURATION_WINDOW_S", 0.0)
# Dải dò lệch thời gian khi so khớp fingerprint, tính bằng item (~0.124 s/item).
# 0 = dò toàn bộ (mặc định). Đặt 120 để tái lập đúng hành vi của pyacoustid —
# nhưng khi đó đoạn cắt từ giữa bài sẽ không khớp được (xem EXP-01).
FP_MAX_ALIGN_OFFSET = _get_int("FP_MAX_ALIGN_OFFSET", 0)
MERT_THRESHOLD = _get_float("MERT_THRESHOLD", 0.97)

# COVER_THRESHOLD = 0.97 — hiệu chỉnh bằng EXP-07 (Cover/Version Identification),
# xem experiments/results/exp07_cover.json: tại 0.97 thì Precision = 1.0 và
# False Match Rate = 0% trên tập held-out (điểm cao nhất của một bài ngoài CSDL
# chỉ đạt 0.9419, tức còn biên an toàn thật).
#
# CHƯA ĐƯỢC NỐI VÀO CASCADE, và không phải vì ngại: tầng này cần descriptor
# chroma của toàn bộ reference, mà muốn có thì phải có AUDIO — repo hiện không
# còn audio của 2750 bản ghi nào (`audio_path` trỏ tới `audio/...` đã mất).
# Không có reference thì không có gì để tìm. Xem README mục EXP-07.
COVER_THRESHOLD = _get_float("COVER_THRESHOLD", 0.97)
COVER_ENABLED = _get("COVER_ENABLED", "true").lower() in ("true", "1", "yes")
COVER_INDEX_PATH = _path("COVER_INDEX_PATH", "data/processed/cover_descriptors.npy")
COVER_ID_MAP_PATH = _path("COVER_ID_MAP_PATH", "data/processed/cover_id_map.json")
TOP_K = _get_int("TOP_K", 5)

# --------------------------------------------------------------------------
# Model & xử lý audio
# --------------------------------------------------------------------------
MERT_MODEL = _get("MERT_MODEL", "m-a-p/MERT-v1-95M")
MERT_MODEL_VERSION = _get("MERT_MODEL_VERSION", "MERT-v1-95M")
MERT_SAMPLE_RATE = _get_int("MERT_SAMPLE_RATE", 24000)
MERT_MAX_DURATION = _get_float("MERT_MAX_DURATION", 30.0)
EMBEDDING_DIM = _get_int("EMBEDDING_DIM", 768)

MAX_UPLOAD_MB = _get_int("MAX_UPLOAD_MB", 100)
ALLOWED_EXTENSIONS = tuple(
    e.strip().lower()
    for e in _get(
        "ALLOWED_EXTENSIONS", ".mp3,.wav,.flac,.m4a,.aac,.ogg,.opus,.mp4,.mkv,.mov,.webm,.avi"
    ).split(",")
    if e.strip()
)
