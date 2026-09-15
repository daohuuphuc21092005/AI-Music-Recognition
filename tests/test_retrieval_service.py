"""Kiểm thử tầng tìm kiếm vector: FAISS thuần và phần gộp theo recording."""
import numpy as np

from backend.services.retrieval_service import (
    VectorIndex,
    build_faiss_index,
    search_recordings,
    search_top_k,
)


def make_index(n: int = 10, dim: int = 768):
    matrix = np.random.randn(n, dim).astype("float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return build_faiss_index(matrix), matrix


def test_faiss_search():
    index, matrix = make_index()
    assert index is not None
    assert index.ntotal == 10

    scores, indices = search_top_k(index, matrix[0], top_k=3)

    assert indices[0] == 0
    assert abs(scores[0] - 1.0) < 1e-4


def test_search_top_k_handles_none():
    assert search_top_k(None, None) == ([], [])


def test_search_recordings_gop_theo_ban_ghi():
    """
    Nhiều segment của cùng một bài phải gộp thành MỘT ứng viên.

    Đây chính là lỗi của bản cũ: Top-5 trả về 5 đoạn của cùng một bài hát.
    """
    index, matrix = make_index(n=6)
    vector_index = VectorIndex(
        index=index,
        # 6 vector nhưng chỉ thuộc 2 bản ghi
        recording_ids=["rec-A", "rec-A", "rec-A", "rec-B", "rec-B", "rec-B"],
        segments=[[0, 15], [15, 30], [30, 45], [0, 15], [15, 30], [30, 45]],
    )

    candidates, stats = search_recordings(vector_index, matrix[1], top_k=5)

    assert len(candidates) == 2, "Phai gop ve dung 2 recording khac nhau"
    assert [c["recording_id"] for c in candidates] == ["rec-A", "rec-B"]
    assert candidates[0]["similarity_score"] == max(
        c["similarity_score"] for c in candidates
    )
    assert candidates[0]["segments_hit"] == 3
    assert candidates[0]["best_segment"] == [15, 30], "Phai giu segment khop nhat"
    assert stats["segments_probed"] <= vector_index.ntotal


def test_index_da_nap_khop_ban_do_id(vector_index):
    """Index thật trên đĩa phải khớp số lượng với bản đồ ID."""
    assert vector_index.ntotal == len(vector_index.recording_ids)
    assert vector_index.n_recordings > 0
    assert vector_index.meta.get("dimension") == 768
