"""
EXP-10 — Multi-window Audio Scanning Evaluation.

Mục tiêu:
  1. So sánh Recall nhận diện giữa chế độ đơn cửa sổ (first 30s) và đa cửa sổ (multi-window)
     trên các truy vấn dài, bị cắt lệch hoặc ghép đoạn (composite / padded offset).
  2. Đo lường rủi ro tăng False Match Rate (FMR) trên tập held-out (UNKNOWN) khi số lượng
     cửa sổ K tăng từ 1 đến SCAN_MAX_WINDOWS (hiệu ứng FMR inflation do K phép thử giả thuyết).

Cách chạy:
    python experiments/exp10_multiwindow/run.py --scan-mode multi --windows 8 --budget 60.0
    python experiments/exp10_multiwindow/run.py --scan-mode first
    python experiments/exp10_multiwindow/run.py --held-out --windows 8
"""
import argparse
import csv
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from backend import config
from backend.services.analysis_pipeline import analyze_audio
from experiments.common import Checkpoint, HeldOutProtocol, print_table, save_result

EXPERIMENT_ID = "exp10_multiwindow"
MANIFEST = os.path.join(config.BASE_DIR, "data", "test_queries", "manifest.csv")


def load_queries(manifest_path: str, max_sources: int = None, transformations: list = None) -> list:
    if not os.path.exists(manifest_path):
        print(f"❌ Không tìm thấy manifest: {manifest_path}")
        print("   Hãy chạy: python scripts/augment_audio.py --from-db --composite")
        return []

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if transformations:
        rows = [r for r in rows if r.get("transformation") in transformations]

    if max_sources:
        sources_seen = set()
        filtered = []
        for r in rows:
            src = r.get("source_recording_id")
            if src not in sources_seen and len(sources_seen) >= max_sources:
                continue
            sources_seen.add(src)
            filtered.append(r)
        rows = filtered

    return rows


def run_evaluation(args):
    queries = load_queries(MANIFEST, max_sources=args.sources)
    if not queries:
        return 1

    # Cấu hình quét
    original_mode = getattr(config, "SCAN_MODE", "multi")
    original_max = getattr(config, "SCAN_MAX_WINDOWS", 8)
    original_budget = getattr(config, "SCAN_TIME_BUDGET_S", 60.0)

    config.SCAN_MODE = args.scan_mode
    config.SCAN_MAX_WINDOWS = args.windows
    config.SCAN_TIME_BUDGET_S = args.budget

    print(f"\n=======================================================")
    print(f"EXP-10: Multi-window Audio Scanning Evaluation")
    print(f"Mode: {config.SCAN_MODE} | Max Windows: {config.SCAN_MAX_WINDOWS} | Budget: {config.SCAN_TIME_BUDGET_S}s")
    print(f"Held-out Mode: {args.held_out} | Total Queries: {len(queries)}")
    print(f"=======================================================\n")

    checkpoint = Checkpoint(f"{EXPERIMENT_ID}_{args.scan_mode}_w{args.windows}", {
        "scan_mode": args.scan_mode,
        "windows": args.windows,
        "budget": args.budget,
        "held_out": args.held_out,
    })

    results = []
    total = len(queries)

    for i, q in enumerate(queries, 1):
        query_id = q["query_id"]
        if checkpoint.has(query_id):
            continue

        raw_path = q["path"]
        path = config.resolve_audio_path(raw_path)
        if not path or not os.path.exists(path):
            continue

        source_rec_id = q.get("source_recording_id")
        transformation = q.get("transformation", "")
        expected_match = q.get("expected_match_type", "EXACT_MATCH")

        exclude_ids = [source_rec_id] if args.held_out else []

        start_t = time.perf_counter()
        try:
            assessment = analyze_audio(
                audio_path=path,
                usage_context={"platform": "youtube", "commercial_use": False, "monetization": False},
                skip_media_check=True,
                exclude_recording_ids=exclude_ids,
            )
            elapsed = time.perf_counter() - start_t
            match_type = assessment.match_type
            matched_id = assessment.matched_recording_id
            evidence = assessment.evidence or {}
            windows_info = evidence.get("windows", {})
            windows_scanned = windows_info.get("windows_scanned", 1)
            stopped_early = windows_info.get("stopped_early", False)
            windows_skipped = windows_info.get("windows_skipped", False)
        except Exception as e:
            print(f"❌ Lỗi xử lý {path}: {e}")
            continue

        is_correct = False
        is_false_match = False

        if args.held_out:
            # Nhãn đúng của held_out là UNKNOWN
            if match_type == "UNKNOWN":
                is_correct = True
            else:
                is_false_match = True
        else:
            if match_type != "UNKNOWN" and str(matched_id) == str(source_rec_id):
                is_correct = True

        rec_result = {
            "query_id": query_id,
            "transformation": transformation,
            "match_type": match_type,
            "matched_id": matched_id,
            "source_id": source_rec_id,
            "is_correct": is_correct,
            "is_false_match": is_false_match,
            "windows_scanned": windows_scanned,
            "stopped_early": stopped_early,
            "windows_skipped": windows_skipped,
            "elapsed_s": round(elapsed, 3),
        }
        results.append(rec_result)
        checkpoint.add(query_id, rec_result)

        if i % 10 == 0 or i == total:
            print(f"[{i}/{total}] {transformation}: match={match_type}, windows={windows_scanned}, t={elapsed:.2f}s")

    # Restore configs
    config.SCAN_MODE = original_mode
    config.SCAN_MAX_WINDOWS = original_max
    config.SCAN_TIME_BUDGET_S = original_budget

    all_records = checkpoint.records
    checkpoint.close()

    # Tổng kết
    total_samples = len(all_records)
    if total_samples == 0:
        print("Không có kết quả nào.")
        return 0

    avg_latency = np.mean([r["elapsed_s"] for r in all_records])
    avg_windows = np.mean([r["windows_scanned"] for r in all_records])
    early_stops = sum(1 for r in all_records if r["stopped_early"])
    time_budget_hits = sum(1 for r in all_records if r["windows_skipped"])

    summary = {
        "scan_mode": args.scan_mode,
        "max_windows": args.windows,
        "time_budget_s": args.budget,
        "total_queries": total_samples,
        "avg_latency_s": round(float(avg_latency), 3),
        "avg_windows_scanned": round(float(avg_windows), 2),
        "early_stops": early_stops,
        "time_budget_hits": time_budget_hits,
    }

    if args.held_out:
        fmr = sum(1 for r in all_records if r["is_false_match"]) / total_samples
        summary["false_match_rate"] = round(float(fmr), 4)
        print(f"\n--- HELD-OUT RESULTS (FMR) ---")
        print(f"Total Unknown Queries: {total_samples}")
        print(f"False Match Rate (FMR): {fmr * 100:.2f}% (Target <= 5%)")
        print(f"Avg Windows Scanned: {avg_windows:.2f}")
    else:
        accuracy = sum(1 for r in all_records if r["is_correct"]) / total_samples
        summary["accuracy"] = round(float(accuracy), 4)

        # Accuracy per transformation
        by_trans = {}
        for r in all_records:
            t = r["transformation"]
            by_trans.setdefault(t, []).append(r["is_correct"])

        trans_stats = {t: round(float(np.mean(vals)), 4) for t, vals in by_trans.items()}
        summary["accuracy_by_transformation"] = trans_stats

        print(f"\n--- RECALL / ACCURACY RESULTS ---")
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")
        print(f"Avg Latency: {avg_latency:.2f}s | Avg Windows: {avg_windows:.2f}")
        print(f"Early Stops: {early_stops} | Budget Limits Hit: {time_budget_hits}")
        print("\nAccuracy by Transformation:")
        for t, acc in trans_stats.items():
            print(f"  - {t:30s}: {acc * 100:.1f}%")

    out_file = f"exp10_{args.scan_mode}_w{args.windows}{'_heldout' if args.held_out else ''}.json"
    save_result(EXPERIMENT_ID, summary, out_file)
    print(f"\n💾 Kết quả lưu tại: experiments/results/{EXPERIMENT_ID}/{out_file}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="EXP-10 Multi-window Audio Scanning")
    parser.add_argument("--scan-mode", default="multi", choices=["first", "multi"],
                        help="Chế độ quét: 'first' (30s đầu) hoặc 'multi' (sliding windows)")
    parser.add_argument("--windows", type=int, default=8,
                        help="Số cửa sổ tối đa (SCAN_MAX_WINDOWS)")
    parser.add_argument("--budget", type=float, default=60.0,
                        help="Giới hạn thời gian quét (SCAN_TIME_BUDGET_S)")
    parser.add_argument("--sources", type=int, default=None,
                        help="Giới hạn số bản ghi nguồn truy vấn")
    parser.add_argument("--held-out", action="store_true", default=False,
                        help="Chạy trên tập held-out để đo FMR")
    args = parser.parse_args()
    return run_evaluation(args)


if __name__ == "__main__":
    sys.exit(main())
