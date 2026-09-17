"""
EXP-01 — Fingerprint Baseline (§15).

Chạy Chromaprint trên tập truy vấn đã biến đổi (`data/test_queries/manifest.csv`)
và đo Precision / Recall / F1 / False Positive Rate / Latency.

Hai lượt đo:
  1. Lượt CHÍNH: reference đầy đủ. Truy vấn đúng khi bản ghi trả về trùng bản
     ghi nguồn VÀ điểm >= τFP.
  2. Lượt HELD-OUT: loại bản ghi nguồn khỏi reference, tức là giả lập "bài này
     không có trong CSDL". Mọi lần hệ thống vẫn nhận một bản ghi nào đó đều là
     DƯƠNG TÍNH GIẢ. Loại theo `experiments.common.HeldOutProtocol`: bản trùng
     fingerprint, bản gần trùng, và với audio_overlay cả bài bị trộn chồng — thiếu
     những thứ đó thì nhận ĐÚNG âm thanh còn trong CSDL cũng bị đếm là nhận nhầm.
     Kết quả ghi cả `held_out_score_exact_only` (chỉ loại bản trùng hệt, giao
     thức cũ) để thấy rõ phần chênh.

Ngoài ra script quét τFP để đề xuất ngưỡng dựa trên số liệu thay vì phỏng đoán.

PHẠM VI: truy vấn sinh từ các bản ghi nguồn × 19 phép biến đổi, chấm trên toàn
bộ fingerprint của reference database.

Về LỚP TƯƠNG ĐƯƠNG fingerprint: cơ chế này ban đầu sinh ra để chống chế cho dữ
liệu mô phỏng hỏng (một fingerprint bị gán cho 38 recording_id). Trên corpus thật
nó vẫn CẦN, nhưng vì lý do chính đáng: FMA có bản thu trùng nhau nằm ở các shard
khác nhau — cùng tiêu đề, cùng nghệ sĩ, cùng thời lượng — và Chromaprint cho chúng
cùng fingerprint là ĐÚNG vì chúng đúng là cùng audio. Chấm theo lớp tương đương là
cách xử lý trung thực cho trùng lặp thật, không phải nới lỏng nhãn.
"""
import csv
import os
import sys
import time
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sqlalchemy import text

from backend import config
from backend.database.session import SessionLocal
from backend.services.fingerprint_service import (
    _decode_array,
    extract_query_fingerprint,
    match_decoded,
)
from experiments.common import Checkpoint, HeldOutProtocol, print_table, save_result

EXPERIMENT_ID = "exp01_fingerprint_baseline"
MANIFEST = os.path.join(config.BASE_DIR, "data", "test_queries", "manifest.csv")
THRESHOLD_SWEEP = [round(t, 2) for t in np.arange(0.05, 1.001, 0.05)]


def load_reference(db):
    """
    Trả (reference, lớp tương đương).

    Dữ liệu mô phỏng có fingerprint TRÙNG LẶP: chuỗi của test.mp3 được gán cho
    38 recording_id khác nhau. Khi hai bản ghi có fingerprint y hệt thì về mặt
    âm thanh KHÔNG THỂ phân biệt được — chấm "sai" vì chọn nhầm một trong số đó
    là chấm sai phương pháp. Vì vậy nhãn đúng được định nghĩa theo LỚP TƯƠNG
    ĐƯƠNG (tập các recording có cùng fingerprint), và lượt held-out loại bỏ cả
    lớp chứ không chỉ một bản ghi.
    """
    rows = db.execute(text(
        "SELECT recording_id, fingerprint FROM fingerprints"
    )).fetchall()

    reference, classes = [], {}
    for rec_id, fingerprint in rows:
        try:
            reference.append((str(rec_id), _decode_array(fingerprint)))
        except Exception:
            continue
        classes.setdefault(str(fingerprint), set()).add(str(rec_id))

    equivalence = {}
    for members in classes.values():
        for rec_id in members:
            equivalence[rec_id] = members
    return reference, equivalence


def score_all(query_vec, reference) -> np.ndarray:
    """Điểm của truy vấn với MỌI reference — tính một lần, dùng cho cả lượt chính lẫn held-out."""
    return np.fromiter((match_decoded(query_vec, vector) for _, vector in reference),
                       dtype=np.float64, count=len(reference))


def main() -> int:
    if not os.path.exists(MANIFEST):
        print(f"❌ Chưa có tập truy vấn: {MANIFEST}\n"
              f"   Chạy: python scripts/augment_audio.py --from-db")
        return 1

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))

    db = SessionLocal()
    try:
        reference, equivalence = load_reference(db)
    finally:
        db.close()

    ambiguous = sum(1 for members in equivalence.values() if len(members) > 1)
    print(f"Reference: {len(reference)} fingerprint | Truy vấn: {len(queries)}")
    print(f"Bản ghi có fingerprint trùng với bản ghi khác: {ambiguous}/{len(equivalence)}")
    held_out = HeldOutProtocol(queries)

    # Checkpoint: mỗi truy vấn phải dò qua TOÀN BỘ reference, nên chi phí tăng
    # theo tích số truy vấn × quy mô reference — hàng giờ ở quy mô hàng chục nghìn
    # fingerprint, quá dài để chấp nhận mất trắng khi bị ngắt.
    checkpoint = Checkpoint(EXPERIMENT_ID, {
        "queries": len(queries),
        "reference_fingerprints": len(reference),
        "fp_max_align_offset": config.FP_MAX_ALIGN_OFFSET,
        "threshold_sweep": list(THRESHOLD_SWEEP),
        "held_out_protocol": "exact+near_duplicates+overlay",
        "near_duplicate_min_score": held_out.min_score,
    })
    reference_ids = np.array([rec_id for rec_id, _ in reference])

    for query in queries:
        path = os.path.join(config.BASE_DIR, query["path"])
        if not os.path.exists(path):
            print(f"⏭️  thiếu file {query['path']}")
            continue
        if checkpoint.has(query["path"]):
            continue

        start = time.perf_counter()
        try:
            _duration, fingerprint = extract_query_fingerprint(path)
            query_vec = _decode_array(fingerprint)
        except Exception as e:
            print(f"⏭️  {query['transformation']}: không tạo được fingerprint ({e})")
            continue

        scores = score_all(query_vec, reference)
        best = int(np.argmax(scores))
        score = float(scores[best])
        predicted = str(reference_ids[best]) if score > 0.0 else None
        latency_ms = (time.perf_counter() - start) * 1000

        true_class = equivalence.get(query["source_recording_id"],
                                     {query["source_recording_id"]})

        # Lượt held-out trên cùng bộ điểm. Bản "exact_only" là giao thức cũ, giữ
        # lại để thấy phần nhận nhầm nào thực ra là âm thanh vẫn còn trong CSDL.
        outside = ~np.isin(reference_ids, list(true_class))
        held_score_exact = float(scores[outside].max()) if outside.any() else 0.0
        outside = ~np.isin(reference_ids, list(held_out.exclusions(query)))
        held_score = float(scores[outside].max()) if outside.any() else 0.0

        checkpoint.add(query["path"], {
            "transformation": query["transformation"],
            "source_recording_id": query["source_recording_id"],
            "true_class_size": len(true_class),
            "predicted": predicted,
            "score": round(float(score), 4),
            "held_out_score": round(float(held_score), 4),
            "held_out_score_exact_only": round(float(held_score_exact), 4),
            "correct_recording": predicted in true_class,
            "exact_recording": predicted == query["source_recording_id"],
            "latency_ms": round(latency_ms, 1),
        })

    results = checkpoint.records
    if not results:
        print("Không có truy vấn nào chạy được.")
        return 1

    # --- Quét ngưỡng --------------------------------------------------------
    sweep = []
    for tau in THRESHOLD_SWEEP:
        accepted = [r for r in results if r["score"] >= tau]
        correct = [r for r in accepted if r["correct_recording"]]
        false_accept = [r for r in results if r["held_out_score"] >= tau]
        false_accept_exact = [r for r in results
                              if r.get("held_out_score_exact_only", r["held_out_score"]) >= tau]

        precision = len(correct) / len(accepted) if accepted else 0.0
        recall = len(correct) / len(results)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        sweep.append({
            "threshold": tau,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(len(false_accept) / len(results), 4),
            "false_positive_rate_exact_only": round(len(false_accept_exact) / len(results), 4),
            "accepted": len(accepted),
        })

    at_current = next(r for r in sweep
                      if abs(r["threshold"] - round(config.FP_THRESHOLD, 2)) < 1e-9)
    best_f1 = max(sweep, key=lambda r: (r["f1"], -r["false_positive_rate"]))
    zero_fp = [r for r in sweep if r["false_positive_rate"] == 0.0]
    lowest_zero_fp = min(zero_fp, key=lambda r: r["threshold"]) if zero_fp else None

    # --- Bảng theo từng phép biến đổi (đây cũng là dữ liệu cho EXP-05) -----
    by_transformation = defaultdict(list)
    for r in results:
        by_transformation[r["transformation"]].append(r)

    transformation_rows = []
    for name, rows in sorted(by_transformation.items()):
        scores = [r["score"] for r in rows]
        matched = sum(1 for r in rows
                      if r["correct_recording"] and r["score"] >= config.FP_THRESHOLD)
        transformation_rows.append([
            name, len(rows), round(float(np.mean(scores)), 4),
            round(float(np.min(scores)), 4), f"{matched}/{len(rows)}",
            round(float(np.mean([r['latency_ms'] for r in rows])), 1),
        ])

    print_table(
        "EXP-01 — Chromaprint theo từng phép biến đổi "
        f"(τFP hiện tại = {config.FP_THRESHOLD})",
        transformation_rows,
        ["Biến đổi", "N", "Điểm TB", "Điểm min", "Khớp đúng", "Latency (ms)"],
    )

    print_table(
        "EXP-01 — Quét ngưỡng τFP",
        [[r["threshold"], r["precision"], r["recall"], r["f1"],
          r["false_positive_rate"]] for r in sweep],
        ["τFP", "Precision", "Recall", "F1", "FPR (held-out)"],
    )

    print(f"\nτFP hiện tại {config.FP_THRESHOLD}: "
          f"P={at_current['precision']} R={at_current['recall']} "
          f"F1={at_current['f1']} FPR={at_current['false_positive_rate']}")
    print(f"F1 cao nhất tại τFP = {best_f1['threshold']} (F1={best_f1['f1']})")
    if lowest_zero_fp:
        print(f"τFP thấp nhất mà FPR = 0: {lowest_zero_fp['threshold']} "
              f"(Recall={lowest_zero_fp['recall']})")

    metrics = {
        "at_current_threshold": at_current,
        "best_f1": best_f1,
        "lowest_threshold_with_zero_fpr": lowest_zero_fp,
        "sweep": sweep,
        "per_transformation": {
            name: {
                "n": len(rows),
                "mean_score": round(float(np.mean([r["score"] for r in rows])), 4),
                "min_score": round(float(np.min([r["score"] for r in rows])), 4),
                "matched_at_current_tau": sum(
                    1 for r in rows
                    if r["correct_recording"] and r["score"] >= config.FP_THRESHOLD),
                "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in rows])), 1),
                "mean_held_out_score": round(
                    float(np.mean([r["held_out_score"] for r in rows])), 4),
            }
            for name, rows in sorted(by_transformation.items())
        },
        "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in results])), 1),
        "queries": len(results),
    }

    path = save_result(
        EXPERIMENT_ID,
        params={
            "reference_fingerprints": len(reference),
            "threshold_sweep": THRESHOLD_SWEEP,
            "current_tau_fp": config.FP_THRESHOLD,
            "fpr_protocol": ("held-out: loại bản trùng fingerprint, bản gần trùng "
                             "và (với audio_overlay) bài bị trộn chồng khỏi reference; "
                             "false_positive_rate_exact_only = chỉ loại bản trùng hệt"),
            "held_out_exclusions": held_out.describe(),
            "ground_truth": ("lớp tương đương fingerprint — gộp các bản thu trùng "
                             "nhau có thật trong corpus nguồn"),
            "matcher": "vectorized Chromaprint bit-error (khớp acoustid tham chiếu)",
        },
        metrics=metrics,
        notes=(
            f"Reference {len(reference)} fingerprint; {len(results)} truy vấn từ "
            f"{len({r['source_recording_id'] for r in results})} bài nguồn x "
            f"{len(by_transformation)} phép biến đổi. Ở τFP = {config.FP_THRESHOLD}, "
            "biến đổi KHÔNG khớp nổi truy vấn nào: "
            + (", ".join(name for name, rows in sorted(by_transformation.items())
                         if not any(r["correct_recording"]
                                    and r["score"] >= config.FP_THRESHOLD for r in rows))
               or "không có")
            + ". Matcher dò "
            + ("TOÀN BỘ offset" if not config.FP_MAX_ALIGN_OFFSET
               else f"offset ±{config.FP_MAX_ALIGN_OFFSET} item")
            + f" (FP_MAX_ALIGN_OFFSET={config.FP_MAX_ALIGN_OFFSET})."
        ),
        extra={"raw_results": results},
    )
    checkpoint.close(remove=True)
    print(f"\n💾 Đã lưu kết quả: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
