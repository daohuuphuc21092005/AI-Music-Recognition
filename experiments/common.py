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


def load_query_manifest(sources: int = None) -> list:
    """
    Đọc manifest tập truy vấn biến đổi. Ném lỗi rõ ràng nếu chưa dựng.

    `sources`: chỉ giữ N bài nguồn đầu tiên theo thứ tự manifest. Manifest được ghi
    nối (`augment_audio.py --append`, vd. 10 bài CC0 cho EXP-08), nên không giới hạn
    thì EXP-03/07 chạy lại sẽ lệch bộ truy vấn với EXP-01/04 đã chạy trước đó.
    """
    import csv

    if not os.path.exists(QUERY_MANIFEST):
        raise FileNotFoundError(
            f"Chưa có tập truy vấn: {QUERY_MANIFEST}. "
            f"Chạy: python scripts/augment_audio.py --from-db"
        )
    with open(QUERY_MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if sources:
        keep = set(list(dict.fromkeys(r["source_recording_id"] for r in rows))[:sources])
        rows = [r for r in rows if r["source_recording_id"] in keep]
    return rows


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
# Giao thức held-out: bài nguồn phải VẮNG MẶT THẬT khỏi reference
# --------------------------------------------------------------------------
NEIGHBOUR_CACHE = os.path.join(RESULTS_DIR, ".cache", "fingerprint_neighbours.json")
# Chỉ lưu láng giềng có điểm từ mức này trở lên. Hai bài không liên quan cho điểm
# Chromaprint (nguyên clip với nguyên clip) cỡ 0.005–0.05.
NEIGHBOUR_FLOOR = 0.05
# Đo trên 100 bài nguồn x 24.375 fingerprint (2026-09-16): ba cặp 0.267 / 0.186 /
# 0.158 đều CÙNG nghệ sĩ và cùng album hoặc cùng bài (bản 12" và 7" của "Digg It!");
# cặp cao nhất tiếp theo chỉ 0.077 và là bài không liên quan. 0.10 nằm giữa khoảng
# trống đó. Quy mô reference đổi thì dò lại (đệm tự mất hiệu lực theo chữ ký bảng).
NEAR_DUPLICATE_MIN_SCORE = 0.10


def _fingerprint_rows() -> list:
    from sqlalchemy import text

    from backend.database.session import SessionLocal

    db = SessionLocal()
    try:
        rows = db.execute(text("SELECT recording_id, fingerprint FROM fingerprints")).fetchall()
    finally:
        db.close()
    return [(str(rec_id), str(fingerprint)) for rec_id, fingerprint in rows]


def fingerprint_neighbours(source_ids, rows: list = None) -> dict:
    """
    {source_id: [[recording_id, điểm], ...]}: các bản ghi KHÁC có điểm Chromaprint
    so nguyên clip với nguyên clip ≥ NEIGHBOUR_FLOOR, giảm dần.

    Mỗi bài nguồn phải dò toàn bộ bảng fingerprints (~4 giây ở 24.375 bản ghi), nên
    kết quả được lưu đệm theo chữ ký nội dung của bảng; chỉ bài chưa có mới phải tính.
    """
    import hashlib

    from backend.services.fingerprint_service import _decode_array, match_decoded

    rows = rows if rows is not None else _fingerprint_rows()
    digest = hashlib.md5()
    for rec_id, fingerprint in sorted(rows):
        digest.update(f"{rec_id}:{fingerprint}\n".encode("utf-8"))
    signature = {"fingerprints": len(rows), "digest": digest.hexdigest(),
                 "floor": NEIGHBOUR_FLOOR}

    cache = {}
    if os.path.exists(NEIGHBOUR_CACHE):
        try:
            with open(NEIGHBOUR_CACHE, encoding="utf-8") as f:
                stored = json.load(f)
            if stored.get("signature") == signature:
                cache = stored.get("neighbours", {})
        except (OSError, ValueError):
            cache = {}

    def flush():
        os.makedirs(os.path.dirname(NEIGHBOUR_CACHE), exist_ok=True)
        tmp = NEIGHBOUR_CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"signature": signature, "neighbours": cache}, f)
        os.replace(tmp, NEIGHBOUR_CACHE)

    sources = list(dict.fromkeys(str(s) for s in source_ids))
    missing = [s for s in sources if s not in cache]
    if missing:
        decoded = []
        for rec_id, fingerprint in rows:
            try:
                decoded.append((rec_id, _decode_array(fingerprint)))
            except Exception:
                continue
        vector_of = dict(decoded)
        print(f"Dò láng giềng fingerprint cho {len(missing)} bài nguồn "
              f"x {len(decoded)} bản ghi...")
        for position, source in enumerate(missing, start=1):
            query = vector_of.get(source)
            found = []
            if query is not None:
                for rec_id, vector in decoded:
                    if rec_id == source:
                        continue
                    score = match_decoded(query, vector)
                    if score >= NEIGHBOUR_FLOOR:
                        found.append([rec_id, round(float(score), 4)])
            cache[source] = sorted(found, key=lambda item: -item[1])
            if position % 10 == 0 or position == len(missing):
                flush()
                print(f"  {position}/{len(missing)}")
    return {s: cache.get(s, []) for s in sources}


def overlay_partners(manifest: list) -> dict:
    """
    {source_recording_id: recording_id của bài bị TRỘN CHỒNG trong truy vấn audio_overlay}.

    Bài trộn chồng cũng nằm trong reference, nên khi chỉ giữ lại bài nguồn mà hệ
    thống nhận ra bài trộn chồng thì đó là nhận ĐÚNG, không phải nhận nhầm. Trên
    EXP-01 ở τFP 0.30, 2 trong 4 "nhận nhầm" held-out là đúng trường hợp này
    (điểm 0.93 và 0.90).

    Dòng sinh sau có cột `overlay_source_recording_id`. Dòng cũ thì suy ra theo quy
    tắc của scripts/augment_audio.py: nguồn thứ i trộn nguồn thứ i+1 (vòng tròn)
    theo thứ tự xuất hiện, CHỈ xét các nguồn chưa có cột tường minh — nhờ vậy manifest
    ghi nối (`--append`) vẫn suy đúng phần cũ. Quy tắc này lệch nếu lượt sinh cũ từng
    BỎ QUA một nguồn quá ngắn, vì bài bị bỏ qua vẫn được đem trộn cho nguồn trước nó.
    """
    overlay_rows = [row for row in manifest if row.get("transformation") == "audio_overlay"]
    explicit = {row["source_recording_id"]: row["overlay_source_recording_id"]
                for row in overlay_rows if row.get("overlay_source_recording_id")}
    legacy = list(dict.fromkeys(row["source_recording_id"] for row in overlay_rows
                                if row["source_recording_id"] not in explicit))
    derived = ({source: legacy[(i + 1) % len(legacy)] for i, source in enumerate(legacy)}
               if len(legacy) >= 2 else {})
    return {**derived, **explicit}


class HeldOutProtocol:
    """
    Tập bản ghi phải loại khỏi reference để một truy vấn thật sự là "bài ngoài CSDL".

    Ba lớp, lớp nào cũng đã gặp trong dữ liệu thật:
      1. Fingerprint y hệt: corpus có bản thu bị nhập trùng dưới nhiều recording_id.
      2. Gần trùng: điểm Chromaprint nguyên clip ≥ `min_score`. Ví dụ
         "Digg It! (12\" FunkMixx)" và "Digg It! (7\" FunkEdit)" cùng nghệ sĩ: điểm
         nguyên clip 0.158, còn đoạn cắt 10–15 giây khớp tới 0.35 > τFP.
      3. Truy vấn audio_overlay: bài bị trộn chồng, kèm cả lớp 1–2 của nó.

    Không loại đủ thì held-out đếm cả những lần hệ thống nhận ĐÚNG âm thanh đang có
    trong CSDL là "nhận nhầm", và False Match Rate bị thổi phồng.
    """

    def __init__(self, manifest: list, min_score: float = NEAR_DUPLICATE_MIN_SCORE,
                 rows: list = None):
        rows = rows if rows is not None else _fingerprint_rows()
        by_fingerprint = {}
        for rec_id, fingerprint in rows:
            by_fingerprint.setdefault(fingerprint, set()).add(rec_id)
        self.exact = {rec: members for members in by_fingerprint.values() for rec in members}
        self.overlay = overlay_partners(manifest)
        self.min_score = min_score
        self.sources = list(dict.fromkeys(row["source_recording_id"] for row in manifest))
        self.neighbours = fingerprint_neighbours(
            list(dict.fromkeys(self.sources + list(self.overlay.values()))), rows)

    def exact_class(self, rec_id) -> set:
        rec_id = str(rec_id)
        return set(self.exact.get(rec_id, {rec_id}))

    def same_audio(self, rec_id) -> set:
        members = self.exact_class(rec_id)
        for other, score in self.neighbours.get(str(rec_id), []):
            if score >= self.min_score:
                members |= self.exact_class(other)
        return members

    def exclusions(self, row: dict) -> set:
        source = row["source_recording_id"]
        excluded = self.same_audio(source)
        partner = self.overlay.get(source)
        if row.get("transformation") == "audio_overlay" and partner:
            excluded |= self.same_audio(partner)
        return excluded

    def describe(self) -> dict:
        """Ghi vào `parameters` của kết quả: người đọc phải biết đã loại những gì."""
        near = {source: [[rec, score] for rec, score in self.neighbours.get(source, [])
                         if score >= self.min_score]
                for source in self.sources}
        return {
            "near_duplicate_min_score": self.min_score,
            "neighbour_floor": NEIGHBOUR_FLOOR,
            "sources": len(self.sources),
            "sources_with_exact_duplicates": sum(
                1 for s in self.sources if len(self.exact_class(s)) > 1),
            "sources_with_near_duplicates": sum(1 for s in self.sources if near[s]),
            "near_duplicates": {s: pairs for s, pairs in near.items() if pairs},
            "overlay_pairs": len(self.overlay),
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
