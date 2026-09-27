"""
EXP-08 — End-to-End System (§15): chạy TRỌN pipeline và chấm điểm ở đầu ra cuối
cùng, tức mức RỦI RO mà người dùng thực sự nhìn thấy.

    audio -> cascade (Chromaprint -> MERT) -> tra cứu rights -> Rule Engine
          -> LOW / CONDITIONAL / HIGH / UNKNOWN

Đây là thí nghiệm duy nhất đo được thứ mà EXP-01..EXP-06 không đo: **sai sót ở
khâu NHẬN DIỆN lan sang khâu ĐÁNH GIÁ QUYỀN như thế nào**. Nhận nhầm bài không
chỉ là một dòng sai trong bảng retrieval — nó kéo theo cả một bộ giấy phép khác,
và có thể lật hẳn mức rủi ro.

Thiết kế:
  - Truy vấn : các file biến đổi trong `data/test_queries/manifest.csv`
                (`--sources N` để chỉ lấy N bài nguồn đầu tiên).
  - Hai điều kiện cho MỖI truy vấn:
      * `known`    : reference đầy đủ. Nhãn đúng = quyết định của Rule Engine khi
                     ĐỊNH DANH HOÀN HẢO (oracle: rights của đúng bản ghi nguồn,
                     identity_confidence = 1.0).
      * `held_out` : bài nguồn VẮNG MẶT khỏi cả ba tầng (fingerprint, FAISS, chỉ mục
                     Cover), kèm bản trùng/gần trùng và bài bị trộn chồng
                     (`experiments.common.HeldOutProtocol`). Nhãn đúng = UNKNOWN.
    Việc loại bỏ đi qua tham số `exclude_recording_ids` của
    `cascade_service.process_music_query` — đúng đường chạy production, không
    sửa CSDL, không dựng lại index.
  - Mỗi kết quả nhận diện được đem chấm với nhiều `usage_context` khác nhau
    (nền tảng / mục đích thương mại / bật kiếm tiền), vì cùng một bản ghi có thể
    ra mức rủi ro khác nhau tuỳ ngữ cảnh sử dụng.

Metrics (§15): Macro-F1, class-wise Precision/Recall, Confusion Matrix,
Unknown Detection Rate, Latency.

Chạy trước:
    python scripts/augment_audio.py --from-db
    python init_db.py
"""
import argparse
import csv
import hashlib
import os
import sys
import time
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from backend import config
from backend.database.session import SessionLocal
from backend.services.cascade_service import process_music_query
from backend.services.decision_service import (
    RISK_LEVELS,
    evaluate_rights_and_risk,
    load_rules,
)
from backend.services.retrieval_service import load_index
from backend.services.rights_service import get_full_music_rights
from experiments.common import Checkpoint, HeldOutProtocol, print_table, save_result

EXPERIMENT_ID = "exp08_end_to_end"
MANIFEST = os.path.join(config.BASE_DIR, "data", "test_queries", "manifest.csv")

# Ngữ cảnh sử dụng: cùng một bản ghi có thể ra mức rủi ro khác nhau tuỳ mục đích.
USAGE_CONTEXTS = {
    "noncommercial": {"platform": "youtube", "commercial_use": False, "monetization": False},
    "commercial": {"platform": "youtube", "commercial_use": True, "monetization": True},
}


# --------------------------------------------------------------------------
# Rule Engine
# --------------------------------------------------------------------------
def decide(rights_lookup: dict, match_type: str, identity_confidence: float,
           context: dict) -> dict:
    """Gọi Rule Engine đúng như `analysis_pipeline.analyze_audio` vẫn gọi."""
    rights_data = (rights_lookup or {}).get("rights_and_licensing")
    return evaluate_rights_and_risk(
        rights_data=rights_data,
        match_info={"match_type": match_type,
                    "identity_confidence": identity_confidence},
        usage_context=context,
    )


def oracle_risk(rights_lookup: dict, context: dict) -> str:
    """Mức rủi ro ĐÚNG: giả định nhận diện hoàn hảo bản ghi nguồn."""
    return decide(rights_lookup, "EXACT_MATCH", 1.0, context)["risk_level"]


# --------------------------------------------------------------------------
# Chỉ số phân loại
# --------------------------------------------------------------------------
def classification_report(pairs, labels):
    """pairs = [(nhãn đúng, nhãn dự đoán)]. Trả confusion matrix + P/R/F1 từng lớp."""
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for truth, pred in pairs:
        matrix[truth][pred] += 1

    per_class, f1s = {}, []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[t][label] for t in labels if t != label)
        fn = sum(matrix[label][p] for p in labels if p != label)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        # Macro-F1 chỉ tính trên lớp CÓ MẶT trong nhãn đúng — lớp support = 0
        # mà cộng vào sẽ kéo tụt chỉ số một cách vô nghĩa.
        if support:
            f1s.append(f1)

    return {
        "confusion_matrix": matrix,
        "per_class": per_class,
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "accuracy": round(
            sum(matrix[l][l] for l in labels) / max(1, len(pairs)), 4),
        "labels_present_in_truth": [l for l in labels if per_class[l]["support"]],
        "n": len(pairs),
    }


def percentiles(values):
    if not values:
        return {}
    return {
        "mean": round(float(np.mean(values)), 1),
        "p50": round(float(np.percentile(values, 50)), 1),
        "p95": round(float(np.percentile(values, 95)), 1),
        "max": round(float(np.max(values)), 1),
    }


# --------------------------------------------------------------------------
def select_queries(queries: list, sources: int = None, per_license: int = None,
                   license_of=None) -> list:
    """
    Chọn bài nguồn theo thứ tự manifest (tất định, không ngẫu nhiên):
      - `per_license`: tối đa N bài cho MỖI loại giấy phép. Mẫu ngẫu nhiên gần như
        không có CC0 (117/24.375 bài) — nhánh duy nhất ra LOW — nên chọn phân tầng
        mới cho lớp đó có mẫu mà không phải chạy cả manifest.
      - `sources`: N bài đầu tiên.
    """
    order = list(dict.fromkeys(q["source_recording_id"] for q in queries))
    if per_license:
        taken, keep = Counter(), set()
        for source in order:
            license_type = license_of(source)
            if taken[license_type] < per_license:
                taken[license_type] += 1
                keep.add(source)
    elif sources:
        keep = set(order[:sources])
    else:
        return queries
    return [q for q in queries if q["source_recording_id"] in keep]


def source_license(db, recording_id: str):
    rights = (get_full_music_rights(recording_id, db) or {}).get("rights_and_licensing") or {}
    return rights.get("license_type") or "KHÔNG CÓ"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=None,
                        help="Chỉ chạy N bài nguồn đầu tiên của manifest")
    parser.add_argument("--per-license", type=int, default=None,
                        help="Tối đa N bài nguồn cho mỗi loại giấy phép")
    args = parser.parse_args()

    if not os.path.exists(MANIFEST):
        print(f"Chưa có tập truy vấn: {MANIFEST}\n"
              f"   Chạy: python scripts/augment_audio.py --from-db")
        return 1

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    db = SessionLocal()
    full_index = load_index(config.FAISS_INDEX_PATH, config.FAISS_ID_MAP_PATH)

    try:
        queries = select_queries(manifest, args.sources, args.per_license,
                                 license_of=lambda rec: source_license(db, rec))
        # Dựng trên CẢ manifest để cặp bài trộn chồng không phụ thuộc --sources
        held_out = HeldOutProtocol(manifest)

        print(f"Reference: {full_index.ntotal} vector / "
              f"{full_index.n_recordings} bản ghi có embedding")
        print(f"Truy vấn: {len(queries)} file x 2 điều kiện x "
              f"{len(USAGE_CONTEXTS)} ngữ cảnh sử dụng")
        print(f"τFP = {config.FP_THRESHOLD} | τMERT = {config.MERT_THRESHOLD} | "
              f"τCover = {config.COVER_THRESHOLD} (bật: {config.COVER_ENABLED})\n")

        # Cache oracle rights theo bản ghi nguồn
        rights_cache = {}

        # Checkpoint theo FILE truy vấn: mỗi file sinh ra 2 điều kiện x N ngữ
        # cảnh, và cả nhóm đó được ghi cùng lúc nên không bao giờ lưu nửa vời.
        # Chữ ký phải gồm CẢ cổng identity của Rule Engine, không chỉ ngưỡng
        # trong .env: EXP-08 là thí nghiệm duy nhất chạy qua Rule Engine, nên
        # sửa rules_v1.yaml mà giữ nguyên .env vẫn làm kết quả đổi hoàn toàn.
        # Thiếu nó thì checkpoint cũ sẽ được dùng lại một cách âm thầm và bảng
        # cuối cùng trộn kết quả của hai cấu hình khác nhau.
        gate = load_rules()["identity_gate"]
        checkpoint = Checkpoint(EXPERIMENT_ID, {
            "tau_fp": config.FP_THRESHOLD,
            "tau_mert": config.MERT_THRESHOLD,
            "tau_cover": config.COVER_THRESHOLD,
            "cover_enabled": config.COVER_ENABLED,
            "cover_tempo_factors": list(config.COVER_TEMPO_FACTORS),
            "fp_prefilter": [config.FP_PREFILTER_TOP_K, config.FP_PREFILTER_BAND,
                             config.FP_PREFILTER_MIN_QUERY_S],
            "top_k": config.TOP_K,
            "queries": len(queries),
            # Cùng số truy vấn nhưng khác bài nguồn thì checkpoint cũ trộn vào bảng
            "query_set": hashlib.md5("|".join(sorted(q["path"] for q in queries))
                                     .encode("utf-8")).hexdigest(),
            "held_out_protocol": "exclude_recording_ids+near_duplicates+overlay",
            "near_duplicate_min_score": held_out.min_score,
            "usage_contexts": sorted(USAGE_CONTEXTS),
            "reference_vectors": full_index.ntotal,
            "identity_gate": gate.get("min_identity_confidence_by_match_type"),
            "identity_gate_default": gate.get("min_identity_confidence"),
        })

        for position, query in enumerate(queries, start=1):
            path = os.path.join(config.BASE_DIR, query["path"])
            if not os.path.exists(path):
                print(f"  thiếu file: {path}")
                continue
            if checkpoint.has(query["path"]):
                continue

            batch = []

            true_rec = query["source_recording_id"]
            klass = held_out.exact_class(true_rec)
            excluded = held_out.exclusions(query)

            if true_rec not in rights_cache:
                rights_cache[true_rec] = get_full_music_rights(true_rec, db)
            true_rights = rights_cache[true_rec]

            for condition in ("known", "held_out"):
                start = time.perf_counter()
                cascade = process_music_query(
                    audio_path=path, db=db, vector_index=full_index,
                    top_k=config.TOP_K,
                    exclude_recording_ids=excluded if condition == "held_out" else None,
                )
                identify_ms = (time.perf_counter() - start) * 1000

                predicted_rec = cascade.get("recording_id")
                start = time.perf_counter()
                predicted_rights = (
                    get_full_music_rights(str(predicted_rec), db) if predicted_rec else None
                )
                rights_ms = (time.perf_counter() - start) * 1000

                timings = (cascade.get("evidence") or {}).get("timings_ms", {})
                for context_name, context in USAGE_CONTEXTS.items():
                    start = time.perf_counter()
                    decision = decide(
                        predicted_rights, cascade.get("match_type"),
                        float(cascade.get("identity_confidence") or 0.0), context,
                    )
                    decision_ms = (time.perf_counter() - start) * 1000

                    truth = ("UNKNOWN" if condition == "held_out"
                             else oracle_risk(true_rights, context))

                    batch.append({
                        "transformation": query["transformation"],
                        "condition": condition,
                        "usage_context": context_name,
                        "true_recording_id": true_rec,
                        "predicted_recording_id": str(predicted_rec) if predicted_rec else None,
                        "identified_correctly": (str(predicted_rec) in klass
                                                 if predicted_rec else False),
                        "same_equivalence_class_but_other_id": bool(
                            predicted_rec and str(predicted_rec) in klass
                            and str(predicted_rec) != true_rec),
                        # Chỉ có nghĩa ở held_out: phải luôn False, nếu True là lỗ rò
                        "predicted_excluded_recording": bool(
                            condition == "held_out" and predicted_rec
                            and str(predicted_rec) in excluded),
                        "excluded_recordings": len(excluded) if condition == "held_out" else 0,
                        "match_type": cascade.get("match_type"),
                        "pipeline_stage": cascade.get("pipeline_stage"),
                        "identity_confidence": round(
                            float(cascade.get("identity_confidence") or 0.0), 4),
                        "true_risk": truth,
                        "true_license": ((true_rights or {}).get("rights_and_licensing")
                                         or {}).get("license_type"),
                        "predicted_risk": decision["risk_level"],
                        "predicted_category": decision["category"],
                        "rule_id": decision["evidence"]["rule_id"],
                        "decision_confidence": decision["decision_confidence"],
                        "latency_ms": round(identify_ms + rights_ms + decision_ms, 2),
                        "identify_ms": round(identify_ms, 2),
                        "rights_ms": round(rights_ms, 2),
                        "decision_ms": round(decision_ms, 3),
                        "fingerprint_ms": timings.get("fingerprint_ms"),
                        "embedding_ms": timings.get("embedding_ms"),
                        "vector_search_ms": timings.get("vector_search_ms"),
                    })

                flag = "OK " if batch[-1]["true_risk"] == batch[-1]["predicted_risk"] else "SAI"
                print(f"  [{position:2}/{len(queries)}] {query['transformation']:18} "
                      f"{condition:9} {cascade.get('match_type'):12} "
                      f"-> {batch[-1]['predicted_risk']:12} "
                      f"(đúng: {batch[-1]['true_risk']}) {flag}")

            # Ghi cả nhóm của file này xuống đĩa một lần
            checkpoint.add(query["path"], batch)

    finally:
        db.close()

    # Gộp các nhóm theo file thành một danh sách phẳng
    records = [row for batch in checkpoint.records for row in batch]

    if not records:
        print("Không chạy được truy vấn nào.")
        return 1

    # ----------------------------------------------------------------------
    # Tổng hợp
    # ----------------------------------------------------------------------
    labels = list(RISK_LEVELS)
    overall = classification_report(
        [(r["true_risk"], r["predicted_risk"]) for r in records], labels)

    by_condition = {
        condition: classification_report(
            [(r["true_risk"], r["predicted_risk"]) for r in records
             if r["condition"] == condition], labels)
        for condition in ("known", "held_out")
    }
    by_context = {
        name: classification_report(
            [(r["true_risk"], r["predicted_risk"]) for r in records
             if r["usage_context"] == name], labels)
        for name in USAGE_CONTEXTS
    }

    held = [r for r in records if r["condition"] == "held_out"]
    known = [r for r in records if r["condition"] == "known"]
    unknown_detected = sum(1 for r in held if r["match_type"] == "UNKNOWN")
    false_match = sum(1 for r in held if r["predicted_recording_id"])

    leaks = sum(1 for r in held if r.get("predicted_excluded_recording"))
    if leaks:
        raise RuntimeError(
            f"Lỗ rò held-out: {leaks} truy vấn trả về đúng bản ghi đã bị loại. "
            f"Kết quả không dùng được.")

    identification = {
        "known_identified_correctly": round(
            sum(1 for r in known if r["identified_correctly"]) / len(known), 4),
        "unknown_detection_rate": round(unknown_detected / len(held), 4),
        "false_match_rate_on_held_out": round(false_match / len(held), 4),
        # Chỉ xét điều kiện `known`: ở `held_out` thì KHÔNG nhận diện được mới là
        # hành vi đúng, gộp chung vào sẽ thổi phồng con số này.
        "risk_correct_when_identification_failed_on_known": round(
            sum(1 for r in known
                if not r["identified_correctly"] and r["true_risk"] == r["predicted_risk"])
            / max(1, sum(1 for r in known if not r["identified_correctly"])), 4),
        "known_identification_failures": sum(
            1 for r in known if not r["identified_correctly"]),
        "resolved_to_other_id_in_same_fingerprint_class": sum(
            1 for r in records if r["same_equivalence_class_but_other_id"]),
    }

    latency = {
        "total_ms": percentiles([r["latency_ms"] for r in records]),
        "identification_ms": percentiles([r["identify_ms"] for r in records]),
        "rights_lookup_ms": percentiles([r["rights_ms"] for r in records]),
        "rule_engine_ms": percentiles([r["decision_ms"] for r in records]),
        "fingerprint_ms": percentiles(
            [r["fingerprint_ms"] for r in records if r["fingerprint_ms"] is not None]),
        "embedding_ms": percentiles(
            [r["embedding_ms"] for r in records if r["embedding_ms"] is not None]),
        "vector_search_ms": percentiles(
            [r["vector_search_ms"] for r in records if r["vector_search_ms"] is not None]),
    }

    # --- In bảng ----------------------------------------------------------
    print_table(
        "EXP-08 — Confusion matrix (hàng = đúng, cột = dự đoán)",
        [[truth] + [overall["confusion_matrix"][truth][pred] for pred in labels]
         for truth in labels],
        ["Đúng \\ Dự đoán"] + list(labels),
    )

    print_table(
        "EXP-08 — Hiệu năng từng lớp rủi ro",
        [[label, overall["per_class"][label]["precision"],
          overall["per_class"][label]["recall"],
          overall["per_class"][label]["f1"],
          overall["per_class"][label]["support"]] for label in labels],
        ["Mức rủi ro", "Precision", "Recall", "F1", "Số mẫu"],
    )

    print_table(
        "EXP-08 — Macro-F1 theo lát cắt",
        [["Toàn bộ", overall["macro_f1"], overall["accuracy"], overall["n"]]]
        + [[f"Điều kiện: {c}", r["macro_f1"], r["accuracy"], r["n"]]
           for c, r in by_condition.items()]
        + [[f"Ngữ cảnh: {c}", r["macro_f1"], r["accuracy"], r["n"]]
           for c, r in by_context.items()],
        ["Lát cắt", "Macro-F1", "Accuracy", "Số mẫu"],
    )

    print_table(
        "EXP-08 — Độ trễ (ms)",
        [[stage, v.get("mean"), v.get("p50"), v.get("p95"), v.get("max")]
         for stage, v in latency.items() if v],
        ["Giai đoạn", "Trung bình", "P50", "P95", "Max"],
    )

    print(f"\nNhận diện đúng khi bài CÓ trong CSDL : "
          f"{identification['known_identified_correctly'] * 100:.1f}%")
    print(f"Từ chối đúng khi bài KHÔNG có trong CSDL: "
          f"{identification['unknown_detection_rate'] * 100:.1f}% "
          f"(False Match Rate {identification['false_match_rate_on_held_out'] * 100:.2f}%, "
          f"ngưỡng §16 là ≤5%)")

    # Đọc từ bản ghi đã lưu chứ không từ rights_cache: chạy tiếp từ checkpoint thì
    # cache chỉ chứa những bài của lượt này.
    source_licenses = list({
        r["true_recording_id"]: r.get("true_license") or "KHÔNG CÓ" for r in known
    }.values())

    target = 0.80
    verdict = "ĐẠT" if overall["macro_f1"] >= target else "CHƯA ĐẠT"
    print(f"Ngưỡng nghiệm thu §16 (Macro-F1 ≥ {target}): {verdict} "
          f"— đo được {overall['macro_f1']}")

    metrics = {
        "overall": overall,
        "by_condition": by_condition,
        "by_usage_context": by_context,
        "identification": identification,
        "latency_ms": latency,
        "acceptance_16": {
            "macro_f1_target": target,
            "macro_f1_measured": overall["macro_f1"],
            "passed": bool(overall["macro_f1"] >= target),
            "false_match_rate_target": 0.05,
            "false_match_rate_measured": identification["false_match_rate_on_held_out"],
            "false_match_rate_passed": bool(
                identification["false_match_rate_on_held_out"] <= 0.05),
        },
        "per_transformation": {
            name: {
                "n": len(rows),
                "risk_correct": sum(1 for r in rows
                                    if r["true_risk"] == r["predicted_risk"]),
                "identified_correctly": sum(1 for r in rows if r["identified_correctly"]),
            }
            for name, rows in sorted(
                {t: [r for r in records if r["transformation"] == t]
                 for t in {r["transformation"] for r in records}}.items())
        },
    }

    path = save_result(
        EXPERIMENT_ID,
        params={
            "protocol": "full pipeline: cascade -> rights -> Rule Engine; nhãn đúng "
                        "sinh bằng oracle identity (điều kiện known) và UNKNOWN "
                        "(điều kiện held_out)",
            "held_out_protocol": ("exclude_recording_ids của cascade_service: bài "
                                  "nguồn vắng mặt khỏi CẢ BA tầng (fingerprint, FAISS, "
                                  "chỉ mục Cover), kèm bản trùng fingerprint, bản gần "
                                  "trùng và bài bị trộn chồng. Không sửa dữ liệu, "
                                  "không dựng lại index."),
            "held_out_exclusions": held_out.describe(),
            "sources_selected": args.sources,
            "per_license_selected": args.per_license,
            "tau_fp": config.FP_THRESHOLD,
            "tau_mert": config.MERT_THRESHOLD,
            "tau_cover": config.COVER_THRESHOLD,
            "cover_enabled": config.COVER_ENABLED,
            "cover_tempo_factors": list(config.COVER_TEMPO_FACTORS),
            "fp_prefilter": {"top_k": config.FP_PREFILTER_TOP_K,
                             "band": config.FP_PREFILTER_BAND,
                             "min_query_s": config.FP_PREFILTER_MIN_QUERY_S},
            "top_k": config.TOP_K,
            "usage_contexts": USAGE_CONTEXTS,
            "queries": len(queries),
            "evaluations": len(records),
            "reference_embedding_vectors": full_index.ntotal,
            "reference_embedding_recordings": full_index.n_recordings,
            "risk_labels": labels,
            "rules_version": "rules_v1",
        },
        metrics=metrics,
        notes=(
            f"PHẠM VI: {len({r['true_recording_id'] for r in records})} bản ghi nguồn "
            f"có audio thật; nhãn đúng (điều kiện known) theo lớp: "
            f"{dict(Counter(r['true_risk'] for r in known))}; giấy phép của bài nguồn: "
            f"{dict(Counter(source_licenses))}. Lớp nào support = 0 thì Macro-F1 ở đây "
            "KHÔNG phải bằng chứng cho nhánh đó của Rule Engine — độ phủ đủ các lớp "
            "của tầng quyết định do tests/test_decision_rules.py bảo đảm. Cái mà EXP-08 "
            "đo được và không thí nghiệm nào khác đo được là: sai sót nhận diện lan "
            "sang mức rủi ro ra sao. "
            "Trường 'resolved_to_other_id_in_same_fingerprint_class' đếm số lần hệ "
            "thống trả về một ID khác trong cùng lớp tương đương. Con số này KHÔNG "
            "phải lỗi — corpus FMA có bản thu trùng nhau thật — nhưng vẫn là rủi ro "
            "cần theo dõi: hai bản ghi cùng audio có thể mang bộ quyền khác nhau."
        ),
        extra={"raw_results": records},
    )
    checkpoint.close(remove=True)
    print(f"\nĐã lưu kết quả: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
