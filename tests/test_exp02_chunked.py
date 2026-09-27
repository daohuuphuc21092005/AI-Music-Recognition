"""
EXP-02 tính theo khối phải cho ĐÚNG số liệu của cách tính N×N cũ.

Cách cũ không chạy nổi ở 48.750 vector (~8,9 GiB), nên cách mới chỉ đáng tin khi
đối chiếu được với nó trên dữ liệu nhỏ — kể cả khi ranh giới khối cắt ngang các
segment của cùng một bản ghi.
"""
import numpy as np
import pytest

from experiments.exp02_mert.run import K_VALUES, retrieval_metrics


def naive_metrics(matrix, recording_ids, tau):
    """Bản sao logic N×N của EXP-02 trước khi chuyển sang tính theo khối."""
    sims = matrix @ matrix.T
    n = len(recording_ids)
    counts = {}
    for rec in recording_ids:
        counts[rec] = counts.get(rec, 0) + 1

    hits = {k: 0 for k in K_VALUES}
    rr, ap = [], []
    for i in range(n):
        true_rec = recording_ids[i]
        if counts[true_rec] < 2:
            continue
        scores = sims[i].copy()
        scores[i] = -np.inf
        ranked, seen = [], set()
        for idx in np.argsort(-scores):
            if not np.isfinite(scores[idx]) or recording_ids[idx] in seen:
                continue
            seen.add(recording_ids[idx])
            ranked.append(recording_ids[idx])
        rank = ranked.index(true_rec) + 1
        for k in K_VALUES:
            hits[k] += int(rank <= k)
        rr.append(1.0 / rank)

        found, precision_sum = 0, 0.0
        for position, idx in enumerate(np.argsort(-scores), start=1):
            if recording_ids[idx] == true_rec and np.isfinite(scores[idx]):
                found += 1
                precision_sum += found / position
        ap.append(precision_sum / (counts[true_rec] - 1))

    same = recording_ids[:, None] == recording_ids[None, :]
    upper = np.triu(np.ones_like(same, dtype=bool), k=1)
    diff_scores = sims[(~same) & upper]
    return {
        **{f"recall@{k}": round(hits[k] / len(rr), 4) for k in K_VALUES},
        "mrr": round(float(np.mean(rr)), 4),
        "map": round(float(np.mean(ap)), 4),
        "queries_evaluated": len(rr),
        "same_scores": np.sort(sims[same & upper]),
        "diff_mean": float(diff_scores.mean()),
        "diff_max": float(diff_scores.max()),
        "diff_above_tau": int((diff_scores >= tau).sum()),
        "diff_p95": float(np.percentile(diff_scores, 95)),
    }


@pytest.mark.parametrize("chunk_rows", [1, 7, 64, 1000])
def test_tinh_theo_khoi_trung_voi_cach_tinh_nxn(chunk_rows):
    rng = np.random.default_rng(11)
    n_recordings, dim = 60, 24
    centers = rng.standard_normal((n_recordings, dim))
    rows, ids = [], []
    for rec in range(n_recordings):
        # Số segment khác nhau, có cả bản ghi chỉ 1 segment (không chấm được)
        for _ in range(1 + rec % 3):
            rows.append(centers[rec] + 0.6 * rng.standard_normal(dim))
            ids.append(f"rec_{rec:03d}")
    order = rng.permutation(len(rows))  # segment của cùng bài KHÔNG nằm liền nhau
    matrix = np.asarray(rows, dtype=np.float32)[order]
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    recording_ids = np.asarray(ids)[order]
    tau = 0.5

    got = retrieval_metrics(matrix, recording_ids, tau, chunk_rows=chunk_rows)
    want = naive_metrics(matrix, recording_ids, tau)

    for key in [f"recall@{k}" for k in K_VALUES] + ["mrr", "map", "queries_evaluated"]:
        assert got[key] == want[key], key

    dist = got["similarity_distribution"]
    assert dist["same_recording"]["count"] == len(want["same_scores"])
    assert dist["different_recording"]["mean"] == pytest.approx(want["diff_mean"], abs=1e-4)
    assert dist["different_recording"]["max"] == pytest.approx(want["diff_max"], abs=1e-4)
    assert got["current_threshold_flags"]["different_recording_pairs_above_tau"] == \
        want["diff_above_tau"]
    resolution = dist["different_recording"]["percentile_resolution"]
    assert abs(dist["different_recording"]["p95"] - want["diff_p95"]) <= resolution + 1e-4
