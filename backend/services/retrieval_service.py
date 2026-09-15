"""
Tầng 2 của phễu cascade: tìm kiếm vector (FAISS) trên embedding MERT.

Điểm cần lưu ý: mỗi bản ghi được chia thành nhiều segment 15s, nên index chứa
NHIỀU vector cho CÙNG một recording (2.619 vector / 114 recording). Nếu lấy
Top-K thẳng từ FAISS thì "Top-5" rất dễ là 5 đoạn của cùng một bài. Vì vậy
`search_recordings()` gộp điểm theo recording_id (lấy max qua các segment) rồi
mới xếp hạng — đây mới là Top-K bản ghi khác nhau.
"""
import json
import os
from dataclasses import dataclass, field

import faiss
import numpy as np


@dataclass
class VectorIndex:
    """Index FAISS kèm bản đồ vị trí vector -> recording_id/segment."""
    index: object
    recording_ids: list
    segments: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    @property
    def n_recordings(self) -> int:
        return len(set(self.recording_ids))


def build_faiss_index(embeddings_matrix: np.ndarray):
    """
    Khởi tạo chỉ mục FAISS dùng Cosine Similarity (IndexFlatIP trên vector đã
    chuẩn hoá L2). Input: ma trận (N, 768).
    """
    if len(embeddings_matrix) == 0:
        return None

    dimension = embeddings_matrix.shape[1]
    # IndexFlatIP = Inner Product. Vector đã chuẩn hoá L2 nên đây chính là Cosine.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix.astype("float32"))
    return index


def load_index(index_path: str, id_map_path: str) -> VectorIndex:
    """
    Nạp index + bản đồ ID. Ném lỗi rõ ràng nếu thiếu/hỏng/lệch số lượng —
    tuyệt đối KHÔNG thay bằng dữ liệu giả (CLAUDE.md §2).
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Không tìm thấy FAISS index: {index_path}. "
            f"Chạy: python scripts/rebuild_faiss_index.py"
        )
    if not os.path.exists(id_map_path):
        raise FileNotFoundError(
            f"Không tìm thấy bản đồ ID: {id_map_path}. "
            f"Chạy: python scripts/rebuild_faiss_index.py"
        )

    index = faiss.read_index(index_path)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    recording_ids = id_map.get("recording_ids", [])
    if index.ntotal != len(recording_ids):
        raise ValueError(
            f"Index và bản đồ ID lệch nhau: index có {index.ntotal} vector nhưng "
            f"bản đồ có {len(recording_ids)} ID. Chạy lại scripts/rebuild_faiss_index.py"
        )

    return VectorIndex(
        index=index,
        recording_ids=recording_ids,
        segments=id_map.get("segments", []),
        meta={k: v for k, v in id_map.items() if k not in ("recording_ids", "segments")},
    )


def search_top_k(index, query_vector: np.ndarray, top_k: int = 5):
    """Tìm Top-K VECTOR gần nhất. Trả (scores, indices) ở mức segment."""
    if query_vector is None or index is None:
        return [], []

    if len(query_vector.shape) == 1:
        query_vector = np.expand_dims(query_vector, axis=0)

    top_k = min(top_k, int(index.ntotal))
    if top_k <= 0:
        return [], []

    scores, indices = index.search(query_vector.astype("float32"), top_k)
    return scores[0].tolist(), indices[0].tolist()


def search_recordings(vector_index: VectorIndex, query_vector: np.ndarray,
                      top_k: int = 5, oversample: int = 20):
    """
    Trả Top-K RECORDING khác nhau, điểm của mỗi recording = similarity cao nhất
    trong các segment của nó.

    `oversample`: lấy dư segment từ FAISS để sau khi gộp vẫn đủ K recording.
    """
    if vector_index is None or vector_index.index is None or query_vector is None:
        return [], []

    n_probe = min(vector_index.ntotal, max(top_k * oversample, 50))
    scores, indices = search_top_k(vector_index.index, query_vector, top_k=n_probe)

    best = {}
    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(vector_index.recording_ids):
            continue
        rec_id = vector_index.recording_ids[idx]
        segment = vector_index.segments[idx] if idx < len(vector_index.segments) else None
        entry = best.get(rec_id)
        if entry is None:
            best[rec_id] = {
                "recording_id": rec_id,
                "similarity_score": float(score),
                "best_segment": segment,
                "segments_hit": 1,
            }
        else:
            entry["segments_hit"] += 1
            if score > entry["similarity_score"]:
                entry["similarity_score"] = float(score)
                entry["best_segment"] = segment

    ranked = sorted(best.values(), key=lambda c: c["similarity_score"], reverse=True)
    return ranked[:top_k], {"segments_probed": n_probe, "segments_returned": len(indices)}
