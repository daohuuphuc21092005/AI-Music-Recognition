"""
EXP-02 — MERT Retrieval (§15).

Câu hỏi: với một đoạn nhạc 15s làm truy vấn, hệ thống có tìm lại đúng BẢN GHI
gốc của nó trong reference database không?

Thiết kế:
  - Truy vấn  : từng vector segment trong index.
  - Tham chiếu: toàn bộ index, TRỪ chính vector truy vấn (leave-one-out).
  - Nhãn đúng : mọi segment khác thuộc CÙNG recording_id.
  - Xếp hạng ở MỨC BẢN GHI: gộp điểm theo recording (lấy max qua các segment),
    đúng như `retrieval_service.search_recordings` đang làm trong API.

Metrics: Recall@1/5/10, MRR, mAP (§15).

Ngoài ra thí nghiệm còn dựng phân bố similarity của cặp CÙNG bản ghi và cặp
KHÁC bản ghi — căn cứ định lượng để chọn τMERT thay vì đoán.

Tính theo KHỐI: bản cũ dựng ma trận N×N, ở 48.750 vector là ~8,9 GiB và không
chạy nổi. Ở đây mỗi lượt chỉ giữ `CHUNK_ROWS` hàng; phân bố của cặp KHÁC bản ghi
(~1,2 tỉ cặp) được tích luỹ bằng histogram nên phân vị có sai số bằng một bin
(HIST_BINS trên [-1, 1] -> 1e-4), còn trung bình/độ lệch chuẩn/max/tỉ lệ vượt τ
là giá trị chính xác.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from backend import config
from experiments.common import (
    load_embedding_matrix,
    print_table,
    save_result,
)

EXPERIMENT_ID = "exp02_mert_retrieval"
K_VALUES = (1, 5, 10)
CHUNK_ROWS = 1024
HIST_BINS = 20000
# Cặp KHÁC bản ghi mà cosine gần như tuyệt đối: dấu hiệu cùng audio nhập dưới hai id
DUPLICATE_COSINE = 0.999


def _histogram_percentile(hist: np.ndarray, q: float) -> float:
    """Phân vị q (0–100) từ histogram trên [-1, 1]; trả MÉP PHẢI của bin chứa nó."""
    cumulative = np.cumsum(hist)
    if cumulative[-1] == 0:
        return float("nan")
    position = int(np.searchsorted(cumulative, q / 100.0 * cumulative[-1], side="left"))
    return -1.0 + 2.0 * (position + 1) / len(hist)


def retrieval_metrics(matrix: np.ndarray, recording_ids, tau: float,
                      chunk_rows: int = CHUNK_ROWS, hist_bins: int = HIST_BINS) -> dict:
    """
    Recall@K / MRR / mAP leave-one-out ở mức bản ghi và phân bố similarity.
    Hàm thuần (không đọc đĩa) để test đối chiếu được với cách tính N×N.
    """
    recording_ids = np.asarray(recording_ids)
    n = len(recording_ids)
    uniques, codes = np.unique(recording_ids, return_inverse=True)
    counts = np.bincount(codes)

    # Gom cột theo bản ghi để lấy max qua các segment bằng một phép reduceat
    perm = np.argsort(codes, kind="stable")
    sorted_codes = codes[perm]
    starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])

    hits = {k: 0 for k in K_VALUES}
    reciprocal_ranks, average_precisions, top1_scores = [], [], []

    same_scores = []
    diff_hist = np.zeros(hist_bins, dtype=np.int64)
    diff_count, diff_sum, diff_sq = 0, 0.0, 0.0
    diff_max = -np.inf
    diff_above_tau, diff_duplicates = 0, 0

    columns = np.arange(n)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        rows = np.arange(stop - start)
        query_codes = codes[start:stop]
        scores = matrix[start:stop] @ matrix.T

        # --- phân bố cặp (chỉ tam giác trên, mỗi cặp đếm một lần) -----------
        upper = columns[None, :] > np.arange(start, stop)[:, None]
        same = query_codes[:, None] == codes[None, :]
        same_scores.append(scores[upper & same])
        diff = scores[upper & ~same]
        if diff.size:
            diff64 = diff.astype(np.float64)
            diff_count += diff.size
            diff_sum += float(diff64.sum())
            diff_sq += float((diff64 * diff64).sum())
            diff_max = max(diff_max, float(diff.max()))
            diff_above_tau += int((diff >= tau).sum())
            diff_duplicates += int((diff >= DUPLICATE_COSINE).sum())
            bins = np.clip(((diff + 1.0) / 2.0 * hist_bins).astype(np.int64), 0, hist_bins - 1)
            diff_hist += np.bincount(bins, minlength=hist_bins)
        del upper, same, diff

        # --- xếp hạng leave-one-out ------------------------------------------
        scores[rows, np.arange(start, stop)] = -np.inf
        record_max = np.maximum.reduceat(scores[:, perm], starts, axis=1)
        true_max = record_max[rows, query_codes]
        ranks = 1 + (record_max > true_max[:, None]).sum(axis=1)
        top1 = record_max.max(axis=1)

        for row in rows:
            if counts[query_codes[row]] < 2:
                # Bản ghi chỉ có 1 segment -> sau leave-one-out không còn nhãn đúng
                continue
            rank = int(ranks[row])
            for k in K_VALUES:
                hits[k] += int(rank <= k)
            reciprocal_ranks.append(1.0 / rank)
            top1_scores.append(float(top1[row]))

            # mAP mức segment: vị trí của segment đúng thứ j = 1 + số điểm lớn hơn nó
            row_scores = scores[row]
            relevant = np.sort(row_scores[codes == query_codes[row]])[::-1]
            relevant = relevant[np.isfinite(relevant)]
            positions = 1 + (row_scores[None, :] > relevant[:, None]).sum(axis=1)
            average_precisions.append(
                float(np.mean(np.arange(1, len(relevant) + 1) / positions)))

    evaluated = len(reciprocal_ranks)
    same_scores = np.concatenate(same_scores) if same_scores else np.empty(0, dtype=np.float32)

    metrics = {f"recall@{k}": round(hits[k] / evaluated, 4) if evaluated else 0.0
               for k in K_VALUES}
    metrics["mrr"] = round(float(np.mean(reciprocal_ranks)), 4) if evaluated else 0.0
    metrics["map"] = round(float(np.mean(average_precisions)), 4) if evaluated else 0.0
    metrics["queries_evaluated"] = evaluated

    diff_mean = diff_sum / diff_count if diff_count else float("nan")
    diff_std = (max(diff_sq / diff_count - diff_mean ** 2, 0.0) ** 0.5
                if diff_count else float("nan"))
    diff_p99 = _histogram_percentile(diff_hist, 99)
    distribution = {
        "same_recording": {
            "count": int(same_scores.size),
            "mean": round(float(same_scores.mean()), 4) if same_scores.size else None,
            "std": round(float(same_scores.std()), 4) if same_scores.size else None,
            "p05": round(float(np.percentile(same_scores, 5)), 4) if same_scores.size else None,
            "p50": round(float(np.percentile(same_scores, 50)), 4) if same_scores.size else None,
            "p95": round(float(np.percentile(same_scores, 95)), 4) if same_scores.size else None,
        },
        "different_recording": {
            "count": int(diff_count),
            "mean": round(diff_mean, 4),
            "std": round(diff_std, 4),
            "p50": round(_histogram_percentile(diff_hist, 50), 4),
            "p95": round(_histogram_percentile(diff_hist, 95), 4),
            "p99": round(diff_p99, 4),
            "max": round(diff_max, 4) if diff_count else None,
            "percentile_resolution": round(2.0 / hist_bins, 6),
            f"pairs_at_or_above_{DUPLICATE_COSINE}": int(diff_duplicates),
        },
        "overlap_same_below_diff_p99": (
            round(float((same_scores < diff_p99).mean()), 4) if same_scores.size else None),
    }
    metrics["similarity_distribution"] = distribution
    metrics["current_threshold_flags"] = {
        "tau_mert_in_config": tau,
        # Ghi cả SỐ CẶP: làm tròn 4 chữ số thì 1.000 cặp trên 1,2 tỉ hiện ra là 0.0
        "different_recording_pairs_above_tau": int(diff_above_tau),
        "share_of_different_recording_pairs_above_tau": (
            round(diff_above_tau / diff_count, 8) if diff_count else None),
        "share_of_same_recording_pairs_above_tau": (
            round(float((same_scores >= tau).mean()), 4) if same_scores.size else None),
    }
    metrics["reference_vectors"] = int(n)
    metrics["reference_recordings"] = int(len(uniques))
    return metrics


def main() -> int:
    matrix, recording_ids, _segments = load_embedding_matrix()
    n = len(recording_ids)
    unique_recordings = len(set(recording_ids.tolist()))
    print(f"Reference: {n} vector / {unique_recordings} bản ghi "
          f"(tính theo khối {CHUNK_ROWS} hàng)")

    metrics = retrieval_metrics(matrix, recording_ids, config.MERT_THRESHOLD)
    distribution = metrics["similarity_distribution"]
    evaluated = metrics["queries_evaluated"]

    print_table(
        "EXP-02 — MERT Retrieval (leave-one-out, xếp hạng ở mức bản ghi)",
        [[f"Recall@{k}", metrics[f'recall@{k}']] for k in K_VALUES]
        + [["MRR", metrics["mrr"]], ["mAP", metrics["map"]],
           ["Số truy vấn", evaluated]],
        ["Metric", "Giá trị"],
    )

    print_table(
        "Phân bố similarity",
        [
            ["Cùng bản ghi", distribution["same_recording"]["count"],
             distribution["same_recording"]["mean"],
             distribution["same_recording"]["p05"],
             distribution["same_recording"]["p50"],
             distribution["same_recording"]["p95"]],
            ["Khác bản ghi", distribution["different_recording"]["count"],
             distribution["different_recording"]["mean"], "-",
             distribution["different_recording"]["p50"],
             distribution["different_recording"]["p95"]],
        ],
        ["Cặp", "Số lượng", "Trung bình", "P05", "P50", "P95"],
    )

    flags = metrics["current_threshold_flags"]
    print(f"\nτMERT hiện tại trong config = {flags['tau_mert_in_config']}")
    print(f"  - {flags['different_recording_pairs_above_tau']} cặp KHÁC bản ghi vẫn vượt "
          f"ngưỡng này ({flags['share_of_different_recording_pairs_above_tau']})")
    print(f"  - {flags['share_of_same_recording_pairs_above_tau'] * 100:.2f}% "
          f"cặp CÙNG bản ghi vượt ngưỡng này")

    path = save_result(
        EXPERIMENT_ID,
        params={
            "protocol": "leave-one-out segment query, recording-level ranking",
            "k_values": list(K_VALUES),
            "reference_vectors": n,
            "reference_recordings": unique_recordings,
            "pooling": "mean",
            "similarity": "cosine (IndexFlatIP on L2-normalized vectors)",
            "chunk_rows": CHUNK_ROWS,
            "different_pair_histogram_bins": HIST_BINS,
            "tau_mert": config.MERT_THRESHOLD,
        },
        metrics=metrics,
        notes=(
            f"Reference gồm {n} vector của {unique_recordings} bản ghi có audio thật. "
            "Truy vấn là chính các segment trong index theo giao thức leave-one-out, "
            "nên con số này là CẬN TRÊN của hiệu năng thực tế — truy vấn đã qua biến "
            "đổi được đo ở EXP-03, EXP-04 và EXP-05. Phân vị của cặp KHÁC bản ghi lấy "
            "từ histogram (sai số một bin); các chỉ số còn lại là chính xác."
        ),
    )
    print(f"\n💾 Đã lưu kết quả: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
