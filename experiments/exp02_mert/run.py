"""
EXP-02 — MERT Retrieval (§15).

Câu hỏi: với một đoạn nhạc 15s làm truy vấn, hệ thống có tìm lại đúng BẢN GHI
gốc của nó trong reference database không?

Thiết kế:
  - Truy vấn  : từng vector segment trong index (2.000 vector / 1.000 bản ghi).
  - Tham chiếu: toàn bộ index, TRỪ chính vector truy vấn (leave-one-out).
  - Nhãn đúng : mọi segment khác thuộc CÙNG recording_id.
  - Xếp hạng ở MỨC BẢN GHI: gộp điểm theo recording (lấy max qua các segment),
    đúng như `retrieval_service.search_recordings` đang làm trong API.

Metrics: Recall@1/5/10, MRR, mAP (§15).

Ngoài ra thí nghiệm còn dựng phân bố similarity của cặp CÙNG bản ghi và cặp
KHÁC bản ghi — đây là căn cứ định lượng để chọn τMERT thay vì đoán.

Lưu ý phạm vi: reference gồm 1.000 bản ghi có audio thật. 922 bản ghi còn lại
trong CSDL là metadata-only (ba nhóm không lấy được từ nguồn mở) nên cố ý không
có embedding — tầng 2 chỉ tìm được trong 1.000 bản ghi, đó là thiết kế chứ không
phải thiếu sót.
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
    similarity_matrix,
)

EXPERIMENT_ID = "exp02_mert_retrieval"
K_VALUES = (1, 5, 10)


def rank_recordings(sims: np.ndarray, recording_ids: np.ndarray, query_index: int):
    """
    Xếp hạng các BẢN GHI theo similarity cao nhất trong số segment của nó,
    sau khi loại bỏ chính vector truy vấn (leave-one-out).
    """
    scores = sims.copy()
    scores[query_index] = -np.inf  # bỏ chính nó

    order = np.argsort(-scores)
    ranked, seen = [], set()
    for idx in order:
        if not np.isfinite(scores[idx]):
            continue
        rec = recording_ids[idx]
        if rec in seen:
            continue
        seen.add(rec)
        ranked.append((rec, float(scores[idx])))
    return ranked


def main() -> int:
    matrix, recording_ids, _segments = load_embedding_matrix()
    n = len(recording_ids)
    unique_recordings = len(set(recording_ids.tolist()))
    print(f"Reference: {n} vector / {unique_recordings} bản ghi")

    sims = similarity_matrix(matrix)

    hits = {k: 0 for k in K_VALUES}
    reciprocal_ranks, average_precisions, top1_scores = [], [], []
    evaluated = 0

    # Đếm số segment của mỗi bản ghi để tính mAP ở mức segment
    counts = {}
    for rec in recording_ids:
        counts[rec] = counts.get(rec, 0) + 1

    for i in range(n):
        true_rec = recording_ids[i]
        if counts[true_rec] < 2:
            # Bản ghi chỉ có đúng 1 segment -> sau leave-one-out không còn nhãn
            # đúng nào trong reference, không thể chấm điểm.
            continue
        evaluated += 1

        ranked = rank_recordings(sims[i], recording_ids, i)
        rec_order = [rec for rec, _ in ranked]
        rank = rec_order.index(true_rec) + 1 if true_rec in rec_order else None

        for k in K_VALUES:
            if rank is not None and rank <= k:
                hits[k] += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        top1_scores.append(ranked[0][1] if ranked else 0.0)

        # mAP ở mức segment: mọi segment khác của cùng bản ghi đều là nhãn đúng
        scores = sims[i].copy()
        scores[i] = -np.inf
        seg_order = np.argsort(-scores)
        relevant_total = counts[true_rec] - 1
        found, precision_sum = 0, 0.0
        for position, idx in enumerate(seg_order, start=1):
            if not np.isfinite(scores[idx]):
                continue
            if recording_ids[idx] == true_rec:
                found += 1
                precision_sum += found / position
                if found == relevant_total:
                    break
        average_precisions.append(precision_sum / relevant_total if relevant_total else 0.0)

    metrics = {f"recall@{k}": round(hits[k] / evaluated, 4) for k in K_VALUES}
    metrics["mrr"] = round(float(np.mean(reciprocal_ranks)), 4)
    metrics["map"] = round(float(np.mean(average_precisions)), 4)
    metrics["queries_evaluated"] = evaluated

    # --- Phân bố similarity: cùng bản ghi vs khác bản ghi -------------------
    same_mask = recording_ids[:, None] == recording_ids[None, :]
    np.fill_diagonal(same_mask, False)
    upper = np.triu(np.ones_like(same_mask, dtype=bool), k=1)

    same_scores = sims[same_mask & upper]
    diff_scores = sims[(~same_mask) & upper]

    distribution = {
        "same_recording": {
            "count": int(same_scores.size),
            "mean": round(float(same_scores.mean()), 4),
            "std": round(float(same_scores.std()), 4),
            "p05": round(float(np.percentile(same_scores, 5)), 4),
            "p50": round(float(np.percentile(same_scores, 50)), 4),
            "p95": round(float(np.percentile(same_scores, 95)), 4),
        },
        "different_recording": {
            "count": int(diff_scores.size),
            "mean": round(float(diff_scores.mean()), 4),
            "std": round(float(diff_scores.std()), 4),
            "p50": round(float(np.percentile(diff_scores, 50)), 4),
            "p95": round(float(np.percentile(diff_scores, 95)), 4),
            "p99": round(float(np.percentile(diff_scores, 99)), 4),
            "max": round(float(diff_scores.max()), 4),
        },
    }

    # Hai phân bố chồng lấn bao nhiêu? Đây là thứ quyết định ngưỡng có dùng được không.
    overlap = float((same_scores < np.percentile(diff_scores, 99)).mean())
    distribution["overlap_same_below_diff_p99"] = round(overlap, 4)

    metrics["similarity_distribution"] = distribution
    metrics["current_threshold_flags"] = {
        "tau_mert_in_config": config.MERT_THRESHOLD,
        "share_of_different_recording_pairs_above_tau": round(
            float((diff_scores >= config.MERT_THRESHOLD).mean()), 4),
        "share_of_same_recording_pairs_above_tau": round(
            float((same_scores >= config.MERT_THRESHOLD).mean()), 4),
    }

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
    print(f"  - {flags['share_of_different_recording_pairs_above_tau']*100:.2f}% "
          f"cặp KHÁC bản ghi vẫn vượt ngưỡng này (nguồn dương tính giả)")
    print(f"  - {flags['share_of_same_recording_pairs_above_tau']*100:.2f}% "
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
        },
        metrics=metrics,
        notes=(
            "Reference gồm 2.000 vector của 1.000 bản ghi có audio thật (922 bản "
            "ghi metadata-only không có embedding, đúng thiết kế). Truy vấn là "
            "chính các segment trong index theo giao thức leave-one-out, nên con "
            "số này là CẬN TRÊN của hiệu năng thực tế — truy vấn đã qua biến đổi "
            "được đo ở EXP-03 và EXP-04."
        ),
    )
    print(f"\n💾 Đã lưu kết quả: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
