"""
EXP-06 — Unknown Track Detection (§15, thí nghiệm BẮT BUỘC).

Câu hỏi: khi truy vấn là bài KHÔNG có trong reference database, hệ thống có
dám nói "không biết" hay lại gán bừa cho một bài nào đó?

Thiết kế (split theo recording_id đúng như §2 yêu cầu):
  - Nhóm UNKNOWN: lần lượt loại BỎ HOÀN TOÀN mọi segment của một bản ghi khỏi
    reference, rồi dùng chính các segment đó làm truy vấn. Đáp án đúng là
    "không có trong DB" -> hệ thống phải trả UNKNOWN.
  - Nhóm KNOWN  : leave-one-out như EXP-02, bản ghi gốc vẫn nằm trong reference.

Sau đó quét ngưỡng τMERT và đo:
  - False Match Rate : tỉ lệ truy vấn UNKNOWN bị gán nhầm cho một bản ghi
  - Unknown Recall   : tỉ lệ truy vấn UNKNOWN được từ chối đúng
  - Unknown Precision: trong các truy vấn bị hệ thống từ chối, bao nhiêu % thực sự là UNKNOWN
  - True Accept Rate : tỉ lệ truy vấn KNOWN được nhận đúng bản ghi

Ngưỡng nghiệm thu nội bộ (§16): False Match Rate ≤ 5%.
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

EXPERIMENT_ID = "exp06_unknown_detection"
THRESHOLDS = [round(t, 3) for t in np.arange(0.70, 1.001, 0.01)]
TARGET_FMR = [0.05, 0.01]  # §16: False Match Rate ≤ 5%


def main() -> int:
    matrix, recording_ids, _segments = load_embedding_matrix()
    n = len(recording_ids)
    unique = sorted(set(recording_ids.tolist()))
    print(f"Reference: {n} vector / {len(unique)} bản ghi")

    sims = similarity_matrix(matrix)

    # --- Nhóm KNOWN: leave-one-out -----------------------------------------
    known_top1, known_correct = [], []
    for i in range(n):
        scores = sims[i].copy()
        scores[i] = -np.inf
        best = int(np.argmax(scores))
        known_top1.append(float(scores[best]))
        known_correct.append(recording_ids[best] == recording_ids[i])
    known_top1 = np.asarray(known_top1)
    known_correct = np.asarray(known_correct)

    # --- Nhóm UNKNOWN: loại bỏ toàn bộ bản ghi khỏi reference --------------
    unknown_top1 = []
    for rec in unique:
        rows = np.where(recording_ids == rec)[0]
        mask = recording_ids == rec          # cột cần loại
        for i in rows:
            scores = sims[i].copy()
            scores[mask] = -np.inf           # bỏ CẢ bản ghi, không chỉ chính nó
            unknown_top1.append(float(np.max(scores)))
    unknown_top1 = np.asarray(unknown_top1)

    print(f"Truy vấn KNOWN  : {known_top1.size}")
    print(f"Truy vấn UNKNOWN: {unknown_top1.size}")

    # --- Quét ngưỡng --------------------------------------------------------
    sweep = []
    for tau in THRESHOLDS:
        accepted_unknown = unknown_top1 >= tau
        false_match_rate = float(accepted_unknown.mean())
        unknown_recall = 1.0 - false_match_rate

        accepted_known = known_top1 >= tau
        true_accept_rate = float((accepted_known & known_correct).mean())
        # Nhận nhầm sang bản ghi khác dù bài CÓ trong DB
        wrong_accept_rate = float((accepted_known & ~known_correct).mean())

        rejected_total = int((~accepted_unknown).sum() + (~accepted_known).sum())
        unknown_precision = (float((~accepted_unknown).sum() / rejected_total)
                             if rejected_total else 0.0)

        sweep.append({
            "threshold": tau,
            "false_match_rate": round(false_match_rate, 4),
            "unknown_recall": round(unknown_recall, 4),
            "unknown_precision": round(unknown_precision, 4),
            "true_accept_rate": round(true_accept_rate, 4),
            "wrong_accept_rate": round(wrong_accept_rate, 4),
        })

    recommended = {}
    for target in TARGET_FMR:
        feasible = [row for row in sweep if row["false_match_rate"] <= target]
        if feasible:
            # Trong các ngưỡng đạt mục tiêu FMR, chọn ngưỡng giữ được nhiều
            # truy vấn đúng nhất
            best = max(feasible, key=lambda r: r["true_accept_rate"])
            recommended[f"fmr<={target}"] = best
        else:
            recommended[f"fmr<={target}"] = None

    print_table(
        "EXP-06 — Quét ngưỡng τMERT",
        [[r["threshold"], r["false_match_rate"], r["unknown_recall"],
          r["true_accept_rate"], r["wrong_accept_rate"]]
         for r in sweep if round(r["threshold"] * 100) % 2 == 0],
        ["τMERT", "False Match Rate", "Unknown Recall", "True Accept", "Wrong Accept"],
    )

    print("\nNgưỡng khuyến nghị:")
    for key, row in recommended.items():
        if row:
            print(f"  {key:12} -> τMERT = {row['threshold']} "
                  f"(FMR={row['false_match_rate']}, "
                  f"nhận đúng {row['true_accept_rate']*100:.1f}% truy vấn known)")
        else:
            print(f"  {key:12} -> KHÔNG ngưỡng nào đạt được mục tiêu này")

    current = next(r for r in sweep if abs(r["threshold"] - config.MERT_THRESHOLD) < 1e-9) \
        if any(abs(r["threshold"] - config.MERT_THRESHOLD) < 1e-9 for r in sweep) else None
    if current:
        print(f"\nτMERT hiện tại trong config = {config.MERT_THRESHOLD}: "
              f"FMR = {current['false_match_rate']*100:.1f}% "
              f"({'ĐẠT' if current['false_match_rate'] <= 0.05 else 'KHÔNG ĐẠT'} "
              f"ngưỡng nghiệm thu §16 là ≤ 5%)")

    metrics = {
        "sweep": sweep,
        "recommended": recommended,
        "current_threshold": config.MERT_THRESHOLD,
        "current_threshold_row": current,
        "known_queries": int(known_top1.size),
        "unknown_queries": int(unknown_top1.size),
        "known_top1_mean": round(float(known_top1.mean()), 4),
        "unknown_top1_mean": round(float(unknown_top1.mean()), 4),
        "unknown_top1_p95": round(float(np.percentile(unknown_top1, 95)), 4),
        "unknown_top1_max": round(float(unknown_top1.max()), 4),
    }

    path = save_result(
        EXPERIMENT_ID,
        params={
            "protocol": "leave-one-recording-out (split theo recording_id, §2)",
            "thresholds": THRESHOLDS,
            "target_false_match_rate": TARGET_FMR,
            "reference_vectors": n,
            "reference_recordings": len(unique),
        },
        metrics=metrics,
        notes=(
            "Truy vấn UNKNOWN ở đây là segment của chính reference set sau khi "
            "loại bản ghi đó ra, chưa phải audio ngoài tập. Reference chỉ 114 "
            "bản ghi nên số liệu mang tính chỉ báo. τMERT khuyến nghị cần kiểm "
            "lại khi reference mở rộng: FMR tăng theo số bản ghi trong index."
        ),
    )
    print(f"\n💾 Đã lưu kết quả: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
