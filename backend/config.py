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
# Không còn mật khẩu mặc định viết cứng trong mã nguồn. Thiếu biến này thì kết nối
# thất bại ngay, main.py ghi lỗi rõ lúc khởi động và /health báo DEGRADED — thay vì
# âm thầm thử một mật khẩu mặc định công khai. Không ném lỗi lúc import: pytest và
# các script không cần CSDL vẫn phải import được backend.
DATABASE_URL = _get(
    "DATABASE_URL",
    "postgresql://postgres@localhost:5432/music_rights_ai",
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
# HIỆU CHỈNH TRÊN CORPUS FMA MEDIUM THẬT: 24.375 bản ghi có fingerprint và có
# descriptor cover, 1.900 truy vấn biến đổi từ 100 bài nguồn. Bộ ngưỡng đầu tiên
# (0.70 / 0.95) hiệu chỉnh trên đúng 2 file audio -> 36 truy vấn, và mỗi lần đo
# lại trên corpus lớn hơn đều cho thấy nó lệch CÓ HỆ THỐNG về phía lạc quan: tập
# nhỏ hiếm khi chứa cặp gây nhầm.
#
# FP_THRESHOLD = 0.30 — hiệu chỉnh bằng EXP-01 (Fingerprint Baseline),
# xem experiments/results/exp01_fingerprint_baseline.json:
#   τ = 0.05 -> P 0.9880 / R 0.6053 / F1 0.7507 / FPR 0.0484
#   τ = 0.15 -> P 0.9972 / R 0.5642 / F1 0.7207 / FPR 0.0163
#   τ = 0.30 -> P 0.9981 / R 0.5558 / F1 0.7140 / FPR 0.0021   <- đang dùng
#   τ = 0.95 -> P 1.0000 / R 0.4679 / F1 0.6375 / FPR 0.0
#   Tiêu chí chọn, tường minh chứ không theo cảm tính: F1 cao nhất trong nhóm
#   giữ FPR <= 0.005. KHÔNG lấy τ = 0.95 dù ở đó FPR = 0: "0 lần nhận nhầm trên
#   1.900 truy vấn" không phân biệt được với 0.0011 (đúng 2 truy vấn) — khoảng
#   tin cậy 95% của 0/1.900 vẫn kéo tới ~0.0016 — mà cái giá là 8,8 điểm recall.
#   Recall trần của tầng này chỉ ~0.58: dịch cao độ và đổi tốc độ (8/19 phép biến
#   đổi) Chromaprint không bắt được bài nào, đó là việc của tầng MERT và Cover.
#
# MERT_THRESHOLD = 0.98 — hiệu chỉnh bằng EXP-06 (Unknown Track Detection) trên
# 48.750 vector / 24.375 bản ghi; mỗi vector vừa làm truy vấn known (leave-one-out)
# vừa làm truy vấn unknown (bỏ cả bản ghi khỏi reference).
# Xem experiments/results/exp06_unknown_detection.json:
#   τ = 0.90 -> False Match Rate 83.59%, true accept 0.8715
#   τ = 0.96 -> False Match Rate 11.79%, true accept 0.6857
#   τ = 0.97 -> False Match Rate  6.10%   (KHÔNG đạt ngưỡng <=5% của §16)
#   τ = 0.98 -> False Match Rate  2.65%, unknown recall 0.9735, true accept 0.3812  <- đang dùng
#   τ = 1.00 -> False Match Rate  0.43%, nhưng true accept 0.0 -> vô dụng
#   Cái giá của 0.98 là thật: chỉ 38% truy vấn known được nhận. Nhưng ở 0.97 thì cứ
#   16 bài lạ có 1 bài bị gán cho một bản ghi có sẵn, và trong hệ thống bản quyền
#   một kết luận SAI về quyền tốn kém hơn nhiều so với một lần bỏ sót — bỏ sót còn
#   được tầng Cover xử lý tiếp, nhận nhầm thì ra thẳng khuyến nghị sai.
#
# NGƯỠNG PHỤ THUỘC QUY MÔ REFERENCE. Đây là quan hệ ĐO ĐƯỢC, không phải suy
# đoán: ở cùng τFP = 0.10, FPR tăng từ 0.0044 (1.000 bản ghi) lên 0.0095 (4.000)
# rồi 0.0226 (24.375). Tầng Cover cũng vậy — chấm trên 100 bài nguồn thì τ = 0.71
# đã đạt §16, nhưng chính vùng ngưỡng đó trên chỉ mục 24.375 bài cho FMR 17%.
# Còn τMERT thì ở CÙNG τ = 0.97: FMR 4,40% trên 8.000 vector nhưng 6,10% trên
# 48.750 vector — phải nâng lên 0.98 mới giữ được §16.
# Mở rộng corpus thêm nữa thì BẮT BUỘC chạy lại EXP-01, EXP-06 và EXP-07 trước
# khi tin ba con số này.
#
# PHẢI khớp với `min_identity_confidence_by_match_type` trong
# configs/rules_v1.yaml (EXACT_MATCH = τFP, NEAR_MATCH = τMERT,
# COVER_MATCH = τCover). Lệch nhau thì cascade nhận một khớp rồi Rule Engine vứt
# chính khớp đó thành UNKNOWN.
# --------------------------------------------------------------------------
FP_THRESHOLD = _get_float("FP_THRESHOLD", 0.30)
# Bộ lọc sơ bộ theo độ dài trước khi so khớp fingerprint (giây).
# 0 = TẮT (mặc định) để không âm thầm ảnh hưởng recall với truy vấn bị cắt.
FP_DURATION_WINDOW_S = _get_float("FP_DURATION_WINDOW_S", 0.0)
# Dải dò lệch thời gian khi so khớp fingerprint, tính bằng item (~0.124 s/item).
# 0 = dò toàn bộ (mặc định). Đặt 120 để tái lập đúng hành vi của pyacoustid —
# nhưng khi đó đoạn cắt từ giữa bài sẽ không khớp được (xem EXP-01).
FP_MAX_ALIGN_OFFSET = _get_int("FP_MAX_ALIGN_OFFSET", 0)
MERT_THRESHOLD = _get_float("MERT_THRESHOLD", 0.98)

# COVER_THRESHOLD = 0.90 — hiệu chỉnh bằng EXP-07, đo ĐÚNG điều kiện server:
# truy vấn là 30 giây đầu của file, tìm trên toàn bộ chỉ mục 24.375 bài
# (experiments/results/exp07_cover.json -> metrics.runtime_protocol):
#   τ = 0.80 -> P 0.9846 / R 0.7732 / FMR 5.7%   (trượt trần 5% của §16)
#   τ = 0.85 -> P 0.9908 / R 0.7405 / FMR 3.3%
#   τ = 0.90 -> P 0.9946 / R 0.6837 / FMR 0.4%   <- đang dùng
#   30 mẫu nhiễu trắng/hồng/nâu đạt cao nhất 0.7278, nên τ phải trên mức đó.
#   Siết FMR <= 0.005 thay vì dừng ở trần 5% của §16 vì Cover là tầng CUỐI:
#   nhận nhầm ở đây ra thẳng một kết luận về quyền, không còn tầng nào đỡ.
#
# Đã nối vào cascade (STAGE_3_COVER). Chỉ mục dựng bằng
# scripts/build_cover_index.py cho mọi bản ghi có audio thật.
COVER_THRESHOLD = _get_float("COVER_THRESHOLD", 0.90)
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
