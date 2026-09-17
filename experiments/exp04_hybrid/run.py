"""
EXP-04 — Fingerprint vs MERT vs Hybrid Cascade (§15: thí nghiệm CHÍNH của đồ án).

Chạy CÙNG một tập truy vấn qua năm hệ thống và so sánh:

  A. Chromaprint đơn thuần        : chỉ tầng 1
  B. MERT đơn thuần               : chỉ tầng 2 (bỏ qua fingerprint)
  C. Cover (chroma + OTI) đơn thuần: chỉ tầng 3, tìm trên toàn bộ chỉ mục cover
  D. Cascade Chromaprint -> MERT  : để thấy riêng phần đóng góp của tầng Cover
  E. Cascade đầy đủ               : ĐÚNG như cascade_service chạy production —
                                    Chromaprint -> MERT -> Cover (khi COVER_ENABLED)

Câu hỏi cần trả lời (§20): Fingerprint thất bại ở đâu? MERT cứu được ở đâu? Cover
cứu thêm được gì? Cascade có tốt hơn từng tầng không, và trả giá bằng bao nhiêu độ trễ?

Bản trước chỉ có A/B/D, trong khi production đã có tầng Cover — tức "hệ thống
thật" trong bảng không còn là hệ thống đang chạy.

Nhãn đúng dùng LỚP TƯƠNG ĐƯƠNG fingerprint (xem EXP-01): corpus FMA có một số
bản thu trùng nhau nằm ở các shard khác nhau, và cho chúng cùng fingerprint là
đúng vì chúng đúng là cùng audio. Đòi hệ thống phân biệt hai file giống hệt nhau
mới tính là đúng thì đó là một yêu cầu vô nghĩa.

Quy mô truy vấn/reference được in ra lúc chạy và ghi vào `parameters` của kết quả.
`--sources N` chỉ chạy N bài nguồn đầu tiên của manifest.
"""
import argparse
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
from backend.services.cover_service import identify_cover, load_cover_index
from backend.services.embedding_service import EmbeddingAudioError, extract_mert_embedding
from backend.services.fingerprint_service import _decode_array, search_fingerprint
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


def fingerprint_stage(query_path, db):
    """
    Tầng 1 đúng như cascade_service gọi (search_fingerprint): lọc ứng viên theo hash
    trùng, quay về quét đầy đủ khi sát ngưỡng, ngưỡng nâng theo độ dài truy vấn.
    EXP-01 mới là nơi quét đầy đủ để hiệu chỉnh τFP.
    """
    start = time.perf_counter()
    result = search_fingerprint(db, query_path)
    latency = (time.perf_counter() - start) * 1000
    accepted = result["match_type"] == "EXACT_MATCH"
    return {
        "recording_id": result["recording_id"] if accepted else None,
        "score": float(result["fingerprint_score"]),
        "accepted": accepted,
        "latency_ms": latency,
        "full_scan": bool(result["prefilter"]["full_scan"]),
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


def cover_stage(query_path, cover_index):
    """Tầng 3 đúng như cascade_service gọi: 30 giây đầu, tìm trên toàn bộ chỉ mục."""
    start = time.perf_counter()
    result = identify_cover(audio_path=query_path, cover_index=cover_index,
                            top_k=config.TOP_K)
    latency = (time.perf_counter() - start) * 1000

    top = result.get("top_candidate") or {}
    accepted = bool(result.get("matched"))
    return {
        "recording_id": top.get("recording_id") if accepted else None,
        "score": float(top.get("similarity_score") or 0.0),
        "accepted": accepted,
        "latency_ms": latency,
        "oti": top.get("oti"),
        "tempo_factor": top.get("tempo_factor"),
        "top_k": [c["recording_id"] for c in result.get("candidates") or []],
    }


def cascade_decision(fp, mert, cover, with_cover: bool) -> dict:
    """Thứ tự dừng sớm của cascade_service.process_music_query."""
    if fp["accepted"]:
        return {**fp, "stage": "STAGE_1_CHROMAPRINT"}
    if mert["accepted"] or not with_cover:
        return {**mert, "stage": "STAGE_2_MERT_RETRIEVAL",
                "latency_ms": fp["latency_ms"] + mert["latency_ms"]}
    return {**cover, "stage": "STAGE_3_COVER",
            "latency_ms": fp["latency_ms"] + mert["latency_ms"] + cover["latency_ms"]}


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=None,
                        help="Chỉ chạy N bài nguồn đầu tiên của manifest")
    args = parser.parse_args()

    if not os.path.exists(MANIFEST):
        print(f"❌ Chưa có tập truy vấn: {MANIFEST}\n"
              f"   Chạy: python scripts/augment_audio.py --from-db")
        return 1

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))
    if args.sources:
        keep = set(list(dict.fromkeys(q["source_recording_id"] for q in queries))[:args.sources])
        queries = [q for q in queries if q["source_recording_id"] in keep]

    db = SessionLocal()
    reference, equivalence = load_reference(db)

    vector_index = load_index(config.FAISS_INDEX_PATH, config.FAISS_ID_MAP_PATH)
    cover_index = load_cover_index()
    if cover_index is None or not cover_index.n_items:
        print("❌ Chưa có chỉ mục cover. Chạy: python scripts/build_cover_index.py")
        return 1
    print(f"Reference: {len(reference)} fingerprint | "
          f"{vector_index.ntotal} vector / {vector_index.n_recordings} bản ghi có embedding | "
          f"chỉ mục cover {cover_index.n_items} bài")
    print(f"Truy vấn: {len(queries)} từ "
          f"{len({q['source_recording_id'] for q in queries})} bài nguồn")
    print(f"τFP = {config.FP_THRESHOLD} | τMERT = {config.MERT_THRESHOLD} | "
          f"τCover = {config.COVER_THRESHOLD} (production bật Cover: {config.COVER_ENABLED})\n")

    # Checkpoint: thí nghiệm này chạy hàng chục phút trên 1.140 truy vấn và có
    # thể bị giết giữa chừng khi máy thiếu RAM. Chữ ký gồm mọi thứ ảnh hưởng kết
    # quả, nên đổi ngưỡng hay đổi tập truy vấn là checkpoint cũ tự bị vứt.
    checkpoint = Checkpoint(EXPERIMENT_ID, {
        "tau_fp": config.FP_THRESHOLD,
        "tau_mert": config.MERT_THRESHOLD,
        "tau_cover": config.COVER_THRESHOLD,
        "cover_enabled": config.COVER_ENABLED,
        "cover_tempo_factors": list(config.COVER_TEMPO_FACTORS),
        "fp_prefilter": [config.FP_PREFILTER_TOP_K, config.FP_PREFILTER_BAND,
                         config.FP_PREFILTER_MIN_QUERY_S],
        "top_k": config.TOP_K,
        "fp_max_align_offset": config.FP_MAX_ALIGN_OFFSET,
        "queries": len(queries),
        "reference_fingerprints": len(reference),
        "reference_vectors": vector_index.ntotal,
        "cover_index_items": cover_index.n_items,
    })

    for index, query in enumerate(queries, start=1):
        path = os.path.join(config.BASE_DIR, query["path"])
        if not os.path.exists(path):
            continue
        if checkpoint.has(query["path"]):
            continue

        true_class = equivalence.get(query["source_recording_id"],
                                     {query["source_recording_id"]})

        fp = fingerprint_stage(path, db)
        mert = mert_stage(path, vector_index)
        cover = cover_stage(path, cover_index)

        cascade_no_cover = cascade_decision(fp, mert, cover, with_cover=False)
        cascade = cascade_decision(fp, mert, cover, with_cover=config.COVER_ENABLED)

        for result in (fp, mert, cover, cascade_no_cover, cascade):
            result["correct"] = result["recording_id"] in true_class

        checkpoint.add(query["path"], {
            "transformation": query["transformation"],
            "source_recording_id": query["source_recording_id"],
            "true_class": sorted(true_class),
            "fingerprint": fp,
            "mert": mert,
            "cover": cover,
            "cascade_no_cover": cascade_no_cover,
            "cascade": cascade,
        })
        print(f"  [{index:2}/{len(queries)}] {query['transformation']:18} "
              f"FP={fp['score']:.3f}{'✓' if fp['correct'] else ' '}  "
              f"MERT={mert['score']:.3f}{'✓' if mert['correct'] else ' '}  "
              f"Cover={cover['score']:.3f}{'✓' if cover['correct'] else ' '}  "
              f"-> {cascade['stage'].replace('STAGE_', 'S')}"
              f"{'✓' if cascade['correct'] else '✗'}")

    db.close()
    per_query = checkpoint.records
    total = len(per_query)
    summaries = [
        summarize("A. Chromaprint", [q["fingerprint"] for q in per_query], total),
        summarize("B. MERT", [q["mert"] for q in per_query], total),
        summarize("C. Cover (chroma/OTI)", [q["cover"] for q in per_query], total),
        summarize("D. Cascade FP -> MERT", [q["cascade_no_cover"] for q in per_query], total),
        summarize("E. Cascade production", [q["cascade"] for q in per_query], total),
    ]

    print_table(
        "EXP-04 — So sánh năm hệ thống",
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
        cover_ok = sum(1 for r in rows if r["cover"]["correct"])
        no_cover_ok = sum(1 for r in rows if r["cascade_no_cover"]["correct"])
        cascade_ok = sum(1 for r in rows if r["cascade"]["correct"])
        robustness_rows.append([
            name, len(rows), f"{fp_ok}/{len(rows)}", f"{mert_ok}/{len(rows)}",
            f"{cover_ok}/{len(rows)}", f"{no_cover_ok}/{len(rows)}",
            f"{cascade_ok}/{len(rows)}",
            "Cover cứu" if cascade_ok > no_cover_ok else (
                "MERT cứu" if mert_ok > fp_ok else ("—" if fp_ok == mert_ok else "FP tốt hơn")),
        ])

    print_table(
        "EXP-04/05 — Hiệu năng theo từng phép biến đổi",
        robustness_rows,
        ["Biến đổi", "N", "Chromaprint", "MERT", "Cover", "FP->MERT", "Production",
         "Nhận xét"],
    )

    # Recall@K ở quy mô toàn chỉ mục với truy vấn ĐÃ BIẾN ĐỔI — đúng tiêu chí
    # "Robust retrieval Recall@5 >= 0.80" của §13. Tính trên top-K bất kể ngưỡng:
    # đây là khả năng TÌM THẤY, còn nhận/từ chối là việc của τ.
    #
    # §13 chấm ở CẤP HỆ THỐNG (quyết định của chủ dự án, 2026-09-17): hợp top-K của
    # các tầng truy xuất (MERT ∪ Cover) — cả hai danh sách đều hiện trong evidence
    # cho người thẩm định. MERT đơn lẻ vẫn được báo cạnh bên, không bị giấu.
    def recall_at_k(stages: tuple, rows: list) -> float:
        hits = sum(1 for q in rows
                   if set().union(*(q[s].get("top_k") or [] for s in stages))
                   & set(q.get("true_class") or [q["source_recording_id"]]))
        return round(hits / len(rows), 4) if rows else 0.0

    retrieval_recall = {
        f"{name}_recall@{config.TOP_K}": {
            "overall": recall_at_k(stages, per_query),
            "per_transformation": {t: recall_at_k(stages, rows)
                                   for t, rows in sorted(by_transformation.items())},
        }
        for name, stages in (("mert", ("mert",)), ("cover", ("cover",)),
                             ("system", ("mert", "cover")))
    }
    print(f"\nRecall@{config.TOP_K} trên toàn chỉ mục (truy vấn đã biến đổi, bỏ qua ngưỡng): "
          f"hệ thống (MERT ∪ Cover) {retrieval_recall[f'system_recall@{config.TOP_K}']['overall']}"
          f" | MERT {retrieval_recall[f'mert_recall@{config.TOP_K}']['overall']}"
          f" | Cover {retrieval_recall[f'cover_recall@{config.TOP_K}']['overall']}"
          f" (§13: Recall@5 >= 0.80, chấm cấp hệ thống)")

    rescued = [name for name, rows in sorted(by_transformation.items())
               if sum(1 for r in rows if r["mert"]["correct"])
               > sum(1 for r in rows if r["fingerprint"]["correct"])]
    rescued_by_cover = [name for name, rows in sorted(by_transformation.items())
                        if sum(1 for r in rows if r["cascade"]["correct"])
                        > sum(1 for r in rows if r["cascade_no_cover"]["correct"])]
    print(f"\nMERT cứu được các phép biến đổi mà Chromaprint bó tay: "
          f"{', '.join(rescued) if rescued else 'không có'}")
    print(f"Tầng Cover cứu thêm được (production so với FP -> MERT): "
          f"{', '.join(rescued_by_cover) if rescued_by_cover else 'không có'}")

    metrics = {
        "systems": summaries,
        "retrieval_recall_at_k": retrieval_recall,
        "per_transformation": {
            name: {
                "n": len(rows),
                "fingerprint_correct": sum(1 for r in rows if r["fingerprint"]["correct"]),
                "mert_correct": sum(1 for r in rows if r["mert"]["correct"]),
                "cover_correct": sum(1 for r in rows if r["cover"]["correct"]),
                "cascade_no_cover_correct": sum(
                    1 for r in rows if r["cascade_no_cover"]["correct"]),
                "cascade_correct": sum(1 for r in rows if r["cascade"]["correct"]),
                "cascade_wrong_accept": sum(
                    1 for r in rows if r["cascade"]["accepted"] and not r["cascade"]["correct"]),
                "fingerprint_mean_score": round(float(np.mean(
                    [r["fingerprint"]["score"] for r in rows])), 4),
                "mert_mean_score": round(float(np.mean(
                    [r["mert"]["score"] for r in rows])), 4),
                "cover_mean_score": round(float(np.mean(
                    [r["cover"]["score"] for r in rows])), 4),
            }
            for name, rows in sorted(by_transformation.items())
        },
        "transformations_rescued_by_mert": rescued,
        "transformations_rescued_by_cover": rescued_by_cover,
        "fingerprint_full_scans": sum(1 for q in per_query
                                      if q["fingerprint"].get("full_scan")),
        "cascade_stage_distribution": {
            stage: sum(1 for q in per_query if q["cascade"]["stage"] == stage)
            for stage in ("STAGE_1_CHROMAPRINT", "STAGE_2_MERT_RETRIEVAL", "STAGE_3_COVER")
        },
    }

    path = save_result(
        EXPERIMENT_ID,
        params={
            "tau_fp": config.FP_THRESHOLD,
            "tau_mert": config.MERT_THRESHOLD,
            "tau_cover": config.COVER_THRESHOLD,
            "cover_enabled_in_production": config.COVER_ENABLED,
            "cover_tempo_factors": list(config.COVER_TEMPO_FACTORS),
            "fp_prefilter": {"top_k": config.FP_PREFILTER_TOP_K,
                             "band": config.FP_PREFILTER_BAND,
                             "min_query_s": config.FP_PREFILTER_MIN_QUERY_S},
            "fp_max_align_offset": config.FP_MAX_ALIGN_OFFSET,
            "reference_fingerprints": len(reference),
            "reference_embedding_vectors": vector_index.ntotal,
            "reference_embedding_recordings": vector_index.n_recordings,
            "cover_index_items": cover_index.n_items,
            "queries": total,
            "source_recordings": len({q["source_recording_id"] for q in per_query}),
            "sources_selected": args.sources,
            "production_cascade": ("Chromaprint -> MERT -> Cover" if config.COVER_ENABLED
                                   else "Chromaprint -> MERT"),
            "ground_truth": ("lớp tương đương fingerprint — gộp các bản thu trùng "
                             "nhau có thật trong corpus nguồn"),
        },
        metrics=metrics,
        notes=(
            f"{total} truy vấn từ {len({q['source_recording_id'] for q in per_query})} "
            f"bài nguồn. MERT tìm trong {vector_index.n_recordings} bản ghi có embedding, "
            f"Cover trong {cover_index.n_items} bài của chỉ mục cover. Ngưỡng dùng ở đây "
            "đã hiệu chỉnh trên chính corpus này (τFP từ EXP-01, τMERT từ EXP-06, τCover "
            "từ EXP-07). Nhãn đúng là lớp trùng fingerprint của bài nguồn: với "
            "audio_overlay, nhận ra bài bị trộn chồng vẫn bị tính là SAI."
        ),
        extra={"raw_results": per_query},
    )
    # Đã lưu kết quả đầy đủ -> checkpoint hết nhiệm vụ
    checkpoint.close(remove=True)
    print(f"\n💾 Đã lưu kết quả: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
