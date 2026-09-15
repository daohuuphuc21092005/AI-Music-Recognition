"""
Hạ tầng dùng chung cho các thí nghiệm EXP-01..EXP-08.

CLAUDE.md §14 yêu cầu mỗi lần chạy phải ghi lại: experiment_id, git_commit,
dataset_version, split_version, model, parameters, threshold, metrics, date.
Không ghi mỗi chữ "MERT" rồi thôi.

Kết quả lưu ở experiments/results/<experiment_id>.json (CSV/JSON là đủ cho MVP,
chưa cần MLflow).
"""
import io
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from backend import config

RESULTS_DIR = os.path.join(config.BASE_DIR, "experiments", "results")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.BASE_DIR, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=config.BASE_DIR, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return bool(out)
    except Exception:
        return True


def dataset_fingerprint() -> dict:
    """Nhận dạng phiên bản dữ liệu bằng kích thước + thời điểm sửa của các file master."""
    info = {}
    for name in ("metadata_master.csv", "rights_master.csv",
                 "fingerprints_master.csv", "embeddings_master.csv"):
        path = os.path.join(config.DATA_DIR, name)
        if os.path.exists(path):
            stat = os.stat(path)
            info[name] = {
                "bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
    return info


def save_result(experiment_id: str, params: dict, metrics: dict,
                notes: str = "", extra: dict = None) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    record = {
        "experiment_id": experiment_id,
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "dataset_version": dataset_fingerprint(),
        "model": {
            "name": config.MERT_MODEL,
            "version": config.MERT_MODEL_VERSION,
            "embedding_dim": config.EMBEDDING_DIM,
            "pooling": "mean",
            "sample_rate": config.MERT_SAMPLE_RATE,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "parameters": params,
        "metrics": metrics,
        "notes": notes,
        **(extra or {}),
    }

    path = os.path.join(RESULTS_DIR, f"{experiment_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=str)
    return path


# --------------------------------------------------------------------------
# Nạp embedding
# --------------------------------------------------------------------------
def load_embedding_matrix():
    """
    Trả (ma trận (N,768) đã chuẩn hoá L2, danh sách recording_id, danh sách segment).

    Đọc thẳng từ bản đồ ID + FAISS index để chắc chắn khớp với thứ tự đang chạy
    thật trong API.
    """
    import faiss

    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.FAISS_ID_MAP_PATH, encoding="utf-8") as f:
        id_map = json.load(f)

    recording_ids = id_map["recording_ids"]
    segments = id_map.get("segments", [])
    if index.ntotal != len(recording_ids):
        raise RuntimeError(
            f"Index ({index.ntotal}) lệch bản đồ ID ({len(recording_ids)}). "
            f"Chạy lại scripts/rebuild_faiss_index.py"
        )

    matrix = index.reconstruct_n(0, index.ntotal).astype("float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
    return matrix, np.asarray(recording_ids), segments


def similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity toàn phần. Với vector đã chuẩn hoá L2 thì tích vô hướng
    chính là cosine, và kết quả TRÙNG KHỚP với IndexFlatIP mà API đang dùng
    (FAISS IndexFlat là tìm kiếm chính xác, không xấp xỉ).
    """
    return matrix @ matrix.T


# --------------------------------------------------------------------------
# Giao thức chấm điểm ở MỨC CỬA SỔ THỜI GIAN (dùng chung cho EXP-03 và EXP-07)
#
# Reference database chỉ còn VECTOR chứ không còn audio gốc, nên mọi thí nghiệm
# cần tính lại đặc trưng từ tín hiệu (mean+std pooling, chroma/CQT...) đều phải
# tự dựng lấy reference từ số audio thật còn giữ được.
#
# Hằng số và hàm nằm ở ĐÂY chứ không nằm trong từng thí nghiệm, vì EXP-07 so sánh
# trực tiếp với số liệu EXP-03 — phép so sánh đó chỉ hợp lệ khi hai bên dùng ĐÚNG
# cùng một bộ cửa sổ. Sao chép logic sang hai file là cách chắc chắn nhất để hai
# bên lệch nhau mà không ai biết.
# --------------------------------------------------------------------------
WINDOW_S = 15.0
REF_HOP_S = 10.0          # đúng quy ước của scripts/build_embeddings.py
QUERY_HOP_S = 5.0         # lấy dày hơn để có đủ mẫu thống kê
CROP_START_S = 30.0       # phải khớp scripts/augment_audio.py
OVERLAP_MIN = 0.5         # tỉ lệ chồng lấn tối thiểu để tính là nhãn đúng

# Hệ số co giãn trục thời gian: thời điểm t trong truy vấn ứng với t * factor
# trong bản gốc. librosa.effects.time_stretch(rate=r) cho file dài gấp 1/r.
TEMPO_FACTORS = {
    "tempo_0_90": 0.90, "tempo_0_95": 0.95,
    "tempo_1_05": 1.05, "tempo_1_10": 1.10,
}

# Số bán cung mà mỗi phép biến đổi dịch cao độ lên (âm là dịch xuống).
PITCH_STEPS = {
    "pitch_plus_1": 1, "pitch_plus_2": 2,
    "pitch_minus_1": -1, "pitch_minus_2": -2,
}

QUERY_MANIFEST = os.path.join(config.BASE_DIR, "data", "test_queries", "manifest.csv")


def windows(duration: float, hop: float, window_s: float = WINDOW_S):
    """Sinh (start, end) cho các cửa sổ. Clip ngắn hơn -> một cửa sổ duy nhất."""
    if duration < window_s:
        return [(0.0, duration)] if duration >= 1.0 else []
    out, start = [], 0.0
    while start + window_s <= duration + 1e-6:
        out.append((start, start + window_s))
        start += hop
    return out


def overlap_ratio(a_start, a_end, b_start, b_end) -> float:
    """Tỉ lệ chồng lấn so với khoảng NGẮN HƠN trong hai khoảng."""
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    shorter = min(a_end - a_start, b_end - b_start)
    return inter / shorter if shorter > 0 else 0.0


def query_crop_start(row: dict) -> float:
    """
    Điểm cắt thực tế của một truy vấn, ĐỌC TỪ MANIFEST chứ không giả định.

    `augment_audio.py` cắt từ giây CROP_START_S, nhưng với bản ghi ngắn hơn thế
    (clip 30 giây của FMA-small chẳng hạn) nó buộc phải lùi về giây 0. Dùng hằng
    số cho mọi truy vấn là cách chắc chắn để lệch nhãn đúng 30 giây mà không ai
    nhận ra. Manifest cũ chưa có cột này thì mới rơi về hằng số.
    """
    try:
        return float((row or {}).get("crop_start_s"))
    except (TypeError, ValueError):
        return CROP_START_S


def query_tempo_factor(row: dict) -> float:
    """Hệ số co giãn thời gian của truy vấn, ưu tiên giá trị ghi trong manifest."""
    try:
        factor = float((row or {}).get("tempo_factor"))
        if factor > 0:
            return factor
    except (TypeError, ValueError):
        pass
    return TEMPO_FACTORS.get((row or {}).get("transformation"), 1.0)


def query_pitch_steps(row: dict) -> int:
    """Lượng dịch cao độ (bán cung) của truy vấn, ưu tiên giá trị trong manifest."""
    try:
        return int((row or {}).get("pitch_steps"))
    except (TypeError, ValueError):
        return PITCH_STEPS.get((row or {}).get("transformation"), 0)


def load_query_manifest() -> list:
    """Đọc manifest tập truy vấn biến đổi. Ném lỗi rõ ràng nếu chưa dựng."""
    import csv

    if not os.path.exists(QUERY_MANIFEST):
        raise FileNotFoundError(
            f"Chưa có tập truy vấn: {QUERY_MANIFEST}. "
            f"Chạy: python scripts/augment_audio.py --from-db"
        )
    with open(QUERY_MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_source_audio(manifest: list) -> dict:
    """
    {source_file: đường dẫn tuyệt đối} cho các file audio gốc còn tồn tại.

    `source_file` được ghi theo cùng quy ước với `recordings.audio_path` (tương
    đối so với AUDIO_ROOT hoặc repo), nên phải tra qua config.resolve_audio_path
    — corpus thật nằm ngoài repo và thường ở ổ đĩa khác.
    """
    sources = {}
    for row in manifest:
        src = row["source_file"]
        if src in sources:
            continue
        path = config.resolve_audio_path(src)
        if not path:
            # Vài file audio nhỏ đi kèm repo vẫn nằm ở test_audio/
            fallback = os.path.join(config.BASE_DIR, "test_audio",
                                    os.path.basename(src))
            path = fallback if os.path.exists(fallback) else ""
        if path:
            sources[src] = path
    return sources


# --------------------------------------------------------------------------
# Checkpoint cho thí nghiệm dài
# --------------------------------------------------------------------------
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, ".checkpoints")


class Checkpoint:
    """
    Ghi kết quả từng đơn vị công việc ra JSONL để chạy lại được sau khi bị ngắt.

    Vì sao cần: EXP-03/04/08 chạy hàng giờ trên hàng nghìn truy vấn. Trên máy
    RAM hạn chế, tiến trình nền có thể bị hệ điều hành (hoặc trình quản lý tác
    vụ) giết giữa chừng — không có checkpoint thì mất sạch và không bao giờ chạy
    xong. Phép tính KHÔNG đổi: mỗi đơn vị vẫn được tính đúng một lần, chỉ là
    không tính lại thứ đã có.

    An toàn về tính đúng đắn: dòng đầu file là CHỮ KÝ cấu hình (ngưỡng, số truy
    vấn, quy ước cửa sổ...). Chữ ký khác đi thì checkpoint cũ bị **vứt bỏ** chứ
    không dùng lẫn — trộn kết quả của hai cấu hình khác nhau vào một bảng còn tệ
    hơn là chạy lại từ đầu.
    """

    def __init__(self, experiment_id: str, signature: dict, enabled: bool = True,
                 keys_only: bool = False):
        """
        `keys_only=True`: chỉ nạp danh sách KHOÁ đã xong, không giữ nội dung.

        Cần cho những công việc mà mỗi bản ghi rất nặng — `build_embeddings` lưu
        vector 768 chiều dạng text, tới bản ghi thứ 3.000 thì việc nạp lại toàn
        bộ checkpoint đã ngốn vài trăm MB và chính nó làm tiến trình bị giết,
        khiến mỗi lượt chạy lại chỉ tiến được vài chục file. Với chế độ này, nội
        dung chỉ được đọc lúc kết thúc, và đọc theo dòng qua `iter_records()`.
        """
        self.path = os.path.join(CHECKPOINT_DIR, f"{experiment_id}.jsonl")
        self.signature = signature
        self.enabled = enabled
        self.keys_only = keys_only
        self._records = []
        self._keys = set()
        self._handle = None
        if enabled:
            self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        stale = False
        try:
            with io.open(self.path, encoding="utf-8") as f:
                header = json.loads(f.readline() or "{}")
                if header.get("signature") != self.signature:
                    # Chỉ ĐÁNH DẤU ở đây, xoá sau khi ra khỏi khối `with`:
                    # trên Windows không xoá được file đang mở.
                    stale = True
                for line in (f if not stale else []):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        # Dòng cuối có thể đứt giữa chừng nếu bị giết lúc đang ghi
                        break
                    self._keys.add(entry["key"])
                    if not self.keys_only:
                        self._records.append(entry["record"])
        except Exception as e:
            print(f"  không đọc được checkpoint ({type(e).__name__}) -> chạy lại từ đầu")
            self._records, self._keys = [], set()
            return

        if stale:
            print("  checkpoint cũ có cấu hình khác -> bỏ đi, chạy lại từ đầu")
            os.remove(self.path)
            self._records, self._keys = [], set()
            return

        done = len(self._keys)
        if done:
            print(f"  tiếp tục từ checkpoint: đã có {done} kết quả")

    def _open(self):
        if self._handle is not None:
            return self._handle
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        fresh = not os.path.exists(self.path)
        self._handle = io.open(self.path, "a", encoding="utf-8", newline="\n")
        if fresh:
            self._handle.write(json.dumps({"signature": self.signature},
                                          ensure_ascii=False) + "\n")
            self._handle.flush()
        return self._handle

    def has(self, key: str) -> bool:
        return key in self._keys

    def add(self, key: str, record) -> None:
        """Ghi NGAY xuống đĩa: bị giết bất cứ lúc nào cũng không mất dòng đã in."""
        self._keys.add(key)
        if not self.keys_only:
            self._records.append(record)
        if not self.enabled:
            return
        handle = self._open()
        handle.write(json.dumps({"key": key, "record": record},
                                ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    @property
    def records(self) -> list:
        return self._records

    def iter_records(self):
        """
        Đọc lại nội dung TỪNG DÒNG từ đĩa, không nạp cả file vào RAM.

        Dùng ở cuối công việc khi `keys_only=True`: lúc đó mới cần nội dung, và
        đọc theo dòng giữ bộ nhớ phẳng bất kể checkpoint lớn tới đâu.
        """
        if not os.path.exists(self.path):
            return
        with io.open(self.path, encoding="utf-8") as f:
            f.readline()                      # bỏ dòng chữ ký
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    break                     # dòng cuối đứt giữa chừng
                yield entry["record"]

    def close(self, remove: bool = False) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if remove and os.path.exists(self.path):
            os.remove(self.path)


def protocol_params(reference_windows: int, query_windows: int) -> dict:
    """Bộ tham số giao thức, ghi vào kết quả để đối chiếu giữa các thí nghiệm."""
    return {
        "window_s": WINDOW_S,
        "reference_hop_s": REF_HOP_S,
        "query_hop_s": QUERY_HOP_S,
        # Điểm cắt lấy theo TỪNG DÒNG manifest, không phải một hằng số chung
        "crop_start_source": "manifest",
        "overlap_min": OVERLAP_MIN,
        "reference_windows": reference_windows,
        "query_windows": query_windows,
    }


# --------------------------------------------------------------------------
# In bảng kết quả
# --------------------------------------------------------------------------
def print_table(title: str, rows: list, headers: list) -> None:
    print(f"\n{title}")
    widths = [max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) if rows
              else len(str(headers[i])) for i in range(len(headers))]
    line = " | ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))
