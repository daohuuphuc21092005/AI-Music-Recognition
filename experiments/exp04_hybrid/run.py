"""
EXP-04 — Fingerprint vs MERT vs Hybrid Cascade (§15: thí nghiệm CHÍNH của đồ án).

Chạy CÙNG một tập truy vấn qua ba hệ thống và so sánh:

  A. Chromaprint đơn thuần   : chỉ tầng 1
  B. MERT đơn thuần          : chỉ tầng 2 (bỏ qua fingerprint)
  C. Cascade (Chromaprint -> MERT): kiến trúc thật của hệ thống

Câu hỏi cần trả lời (§20): Fingerprint thất bại ở đâu? MERT cứu được ở đâu?
Cascade có tốt hơn cả hai không, và trả giá bằng bao nhiêu độ trễ?

Nhãn đúng dùng LỚP TƯƠNG ĐƯƠNG fingerprint (xem EXP-01): corpus FMA có một số
bản thu trùng nhau nằm ở các shard khác nhau, và cho chúng cùng fingerprint là
đúng vì chúng đúng là cùng audio. Đòi hệ thống phân biệt hai file giống hệt nhau
mới tính là đúng thì đó là một yêu cầu vô nghĩa.

PHẠM VI: 1.140 truy vấn từ 60 bản ghi nguồn, reference 1.000 fingerprint và
2.000 vector MERT của 1.000 bản ghi.
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
from backend.services.embedding_service import EmbeddingAudioError, extract_mert_embedding
from backend.services.fingerprint_service import (
    _decode_array,
    extract_query_fingerprint,
    match_decoded,
)
from backend.services.retrieval_service import load_index, search_recordings
from experiments.common import Checkpoint, print_table, save_result

EXPERIMENT_ID = "exp04_hybrid_cascade"
MANIFEST = os.path.join(config.BASE_DIR, "data", "test_queries", "manifest.csv")


def load_reference(db):
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


def fingerprint_stage(query_path, reference):
    start = time.perf_counter()
    _duration, fingerprint = extract_query_fingerprint(query_path)
    query_vec = _decode_array(fingerprint)

    best_id, best_score = None, 0.0
    for rec_id, vector in reference:
        score = match_decoded(query_vec, vector)
        if score > best_score:
            best_score, best_id = score, rec_id

    latency = (time.perf_counter() - start) * 1000
    accepted = best_score >= config.FP_THRESHOLD
    return {
        "recording_id": best_id if accepted else None,
        "score": float(best_score),
        "accepted": accepted,
        "latency_ms": latency,
    }


def mert_stage(query_path, vector_index):
    start = time.perf_counter()
    try:
        vector = extract_mert_embedding(query_path)
    except EmbeddingAudioError:
        # File truy vấn không giải mã được: tính là không nhận diện được. Lỗi MODEL
        # thì để nổi lên — lặng lẽ đếm nó thành "trượt" sẽ làm sai số liệu thí nghiệm.
        return {"recording_id": None, "score": 0.0, "accepted": False,
                "latency_ms": (time.perf_counter() - start) * 1000}

    candidates, _stats = search_recordings(vector_index, vector, top_k=config.TOP_K)
    latency = (time.perf_counter() - start) * 1000

    best = candidates[0] if candidates else None
    score = best["similarity_score"] if best else 0.0
    accepted = score >= config.MERT_THRESHOLD
    return {
        "recording_id": best["recording_id"] if (best and accepted) else None,
        "score": float(score),
        "accepted": accepted,
        "latency_ms": latency,
        "top_k": [c["recording_id"] for c in candidates],
    }


def summarize(name, rows, total):
    accepted = [r for r in rows if r["accepted"]]
    correct = [r for r in accepted if r["correct"]]
    wrong = [r for r in accepted if not r["correct"]]

    precision = len(correct) / len(accepted) if accepted else 0.0
    recall = len(correct) / total
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "system": name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accepted": len(accepted),
        "correct": len(correct),
        "wrong_accept": len(wrong),
        "abstained": total - len(accepted),
        "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in rows])), 1),
    }


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

    vector_index = load_index(config.FAISS_INDEX_PATH, config.FAISS_ID_MAP_PATH)
    print(f"Reference: {len(reference)} fingerprint | "
          f"{vector_index.ntotal} vector / {vector_index.n_recordings} bản ghi có embedding")
    print(f"Truy vấn: {len(queries)}")
    print(f"τFP = {config.FP_THRESHOLD} | τMERT = {config.MERT_THRESHOLD}\n")

    # Checkpoint: thí nghiệm này chạy hàng chục phút trên 1.140 truy vấn và có
    # thể bị giết giữa chừng khi máy thiếu RAM. Chữ ký gồm mọi thứ ảnh hưởng kết
    # quả, nên đổi ngưỡng hay đổi tập truy vấn là checkpoint cũ tự bị vứt.
    checkpoint = Checkpoint(EXPERIMENT_ID, {
        "tau_fp": config.FP_THRESHOLD,
        "tau_mert": config.MERT_THRESHOLD,
        "top_k": config.TOP_K,
        "fp_max_align_offset": config.FP_MAX_ALIGN_OFFSET,
        "queries": len(queries),
        "reference_fingerprints": len(reference),
        "reference_vectors": vector_index.ntotal,
    })

    for index, query in enumerate(queries, start=1):
        path = os.path.join(config.BASE_DIR, query["path"])
        if not os.path.exists(path):
            continue
        if checkpoint.has(query["path"]):
            continue

        true_class = equivalence.get(query["source_recording_id"],
                                     {query["source_recording_id"]})

        fp = fingerprint_stage(path, reference)
        mert = mert_stage(path, vector_index)

        # Cascade: tầng 1 khớp thì dừng, không thì mới chạy tầng 2
        if fp["accepted"]:
            cascade = {**fp, "stage": "STAGE_1_CHROMAPRINT"}
        else:
            cascade = {**mert,
                       "stage": "STAGE_2_MERT_RETRIEVAL",
                       "latency_ms": fp["latency_ms"] + mert["latency_ms"]}

        for result in (fp, mert, cascade):
            result["correct"] = result["recording_id"] in true_class

        checkpoint.add(query["path"], {
            "transformation": query["transformation"],
            "source_recording_id": query["source_recording_id"],
            "fingerprint": fp,
            "mert": mert,
            "cascade": cascade,
        })
        print(f"  [{index:2}/{len(queries)}] {query['transformation']:18} "
              f"FP={fp['score']:.3f}{'✓' if fp['correct'] else ' '}  "
              f"MERT={mert['score']:.3f}{'✓' if mert['correct'] else ' '}  "
              f"-> {cascade['stage'].replace('STAGE_', 'S')}"
              f"{'✓' if cascade['correct'] else '✗'}")

    per_query = checkpoint.records
    total = len(per_query)
    summaries = [
        summarize("A. Chromaprint", [q["fingerprint"] for q in per_query], total),
        summarize("B. MERT", [q["mert"] for q in per_query], total),
        summarize("C. Cascade", [q["cascade"] for q in per_query], total),
    ]

    print_table(
        "EXP-04 — So sánh ba hệ thống",
        [[s["system"], s["precision"], s["recall"], s["f1"], s["correct"],
          s["wrong_accept"], s["abstained"], s["mean_latency_ms"]]
         for s in summaries],
        ["Hệ thống", "Precision", "Recall", "F1", "Đúng", "Nhận sai",
         "Từ chối", "Latency TB (ms)"],
    )

    # --- Bảng robustness: hệ nào cứu được phép biến đổi nào (dữ liệu EXP-05) --
    by_transformation = defaultdict(list)
    for q in per_query:
        by_transformation[q["transformation"]].append(q)

    robustness_rows = []
    for name, rows in sorted(by_transformation.items()):
        fp_ok = sum(1 for r in rows if r["fingerprint"]["correct"])
        mert_ok = sum(1 for r in rows if r["mert"]["correct"])
        cascade_ok = sum(1 for r in rows if r["cascade"]["correct"])
        robustness_rows.append([
            name, len(rows), f"{fp_ok}/{len(rows)}", f"{mert_ok}/{len(rows)}",
            f"{cascade_ok}/{len(rows)}",
            "MERT cứu" if mert_ok > fp_ok else ("—" if fp_ok == mert_ok else "FP tốt hơn"),
        ])

    print_table(
        "EXP-04/05 — Hiệu năng theo từng phép biến đổi",
        robustness_rows,
        ["Biến đổi", "N", "Chromaprint", "MERT", "Cascade", "Nhận xét"],
    )

    rescued = [name for name, rows in sorted(by_transformation.items())
               if sum(1 for r in rows if r["mert"]["correct"])
               > sum(1 for r in rows if r["fingerprint"]["correct"])]
    print(f"\nMERT cứu được các phép biến đổi mà Chromaprint bó tay: "
          f"{', '.join(rescued) if rescued else 'không có'}")

    metrics = {
        "systems": summaries,
        "per_transformation": {
            name: {
                "n": len(rows),
                "fingerprint_correct": sum(1 for r in rows if r["fingerprint"]["correct"]),
                "mert_correct": sum(1 for r in rows if r["mert"]["correct"]),
                "cascade_correct": sum(1 for r in rows if r["cascade"]["correct"]),
                "fingerprint_mean_score": round(float(np.mean(
                    [r["fingerprint"]["score"] for r in rows])), 4),
                "mert_mean_score": round(float(np.mean(
                    [r["mert"]["score"] for r in rows])), 4),
            }
            for name, rows in sorted(by_transformation.items())
        },
        "transformations_rescued_by_mert": rescued,
        "cascade_stage_distribution": {
            stage: sum(1 for q in per_query if q["cascade"]["stage"] == stage)
            for stage in ("STAGE_1_CHROMAPRINT", "STAGE_2_MERT_RETRIEVAL")
        },
    }

    path = save_result(
        EXPERIMENT_ID,
        params={
            "tau_fp": config.FP_THRESHOLD,
            "tau_mert": config.MERT_THRESHOLD,
            "fp_max_align_offset": config.FP_MAX_ALIGN_OFFSET,
            "reference_fingerprints": len(reference),
            "reference_embedding_vectors": vector_index.ntotal,
            "reference_embedding_recordings": vector_index.n_recordings,
            "queries": total,
            "ground_truth": ("lớp tương đương fingerprint — gộp các bản thu trùng "
                             "nhau có thật trong corpus nguồn"),
        },
        metrics=metrics,
        notes=(
            "1.140 truy vấn từ 60 bản ghi nguồn, đủ để kết luận thống kê ở mức "
            "60 mẫu cho mỗi phép biến đổi. MERT tìm trong 1.000 bản ghi có "
            "embedding; 922 bản ghi metadata-only không có audio nên cố ý không "
            "có embedding. Ngưỡng dùng ở đây đã hiệu chỉnh trên chính corpus này "
            "(τFP từ EXP-01, τMERT từ EXP-06)."
        ),
        extra={"raw_results": per_query},
    )
    # Đã lưu kết quả đầy đủ -> checkpoint hết nhiệm vụ
    checkpoint.close(remove=True)
    print(f"\n💾 Đã lưu kết quả: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
