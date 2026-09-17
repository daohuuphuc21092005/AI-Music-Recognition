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


def test_loai_tru_cho_ket_qua_y_het_index_da_loc():
    """
    exclude_recording_ids phải tương đương TUYỆT ĐỐI với việc tìm trên một index
    không chứa các bản ghi đó — đây là điều kiện để held-out của EXP-08 đo đúng
    mà không phải dựng lại FAISS cho từng bài nguồn.
    """
    rng = np.random.default_rng(7)
    n_recordings, per_recording, dim = 120, 2, 32
    matrix = rng.standard_normal((n_recordings * per_recording, dim)).astype("float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    recording_ids = [f"rec_{i // per_recording}" for i in range(len(matrix))]
    full = VectorIndex(index=build_faiss_index(matrix), recording_ids=recording_ids,
                       segments=list(range(len(matrix))))

    for trial in range(20):
        query = matrix[trial * 7 % len(matrix)] + 0.05 * rng.standard_normal(dim).astype("float32")
        query /= np.linalg.norm(query)
        # Loại đúng những bài GẦN NHẤT — trường hợp dễ rò nhất
        nearest, _ = search_recordings(full, query, top_k=8, oversample=1)
        excluded = frozenset(c["recording_id"] for c in nearest[: 1 + trial % 6])

        keep = [i for i, rec in enumerate(recording_ids) if rec not in excluded]
        filtered = VectorIndex(index=build_faiss_index(matrix[keep]),
                               recording_ids=[recording_ids[i] for i in keep],
                               segments=keep)

        for top_k, oversample in ((5, 20), (3, 1)):
            got, _ = search_recordings(full, query, top_k=top_k, oversample=oversample,
                                       exclude_recording_ids=excluded)
            want, _ = search_recordings(filtered, query, top_k=top_k, oversample=oversample)
            assert [c["recording_id"] for c in got] == [c["recording_id"] for c in want]
            assert [round(c["similarity_score"], 5) for c in got] == \
                [round(c["similarity_score"], 5) for c in want]
            assert not excluded & {c["recording_id"] for c in got}


def test_loai_tru_het_thi_tra_rong():
    index, matrix = make_index(n=4, dim=16)
    vector_index = VectorIndex(index=index, recording_ids=["a", "a", "b", "b"])
    got, _ = search_recordings(vector_index, matrix[0], top_k=5,
                               exclude_recording_ids=frozenset({"a", "b"}))
    assert got == []
