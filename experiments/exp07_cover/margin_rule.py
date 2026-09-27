"""
EXP-07b — Luật chấp nhận của tầng Cover theo KHOẢNG CÁCH hạng 1 – hạng 2.

Chỉ ĐO, không đổi production (quyết định của chủ dự án, 2026-09-18).

Vì sao: EXP-04 cho thấy Cover xếp đúng bài ở hạng 1 cho 400/400 truy vấn dịch cao
độ và 400/400 truy vấn đổi nhịp trên chỉ mục 24.375 bài, nhưng 116 và 82 truy vấn
bị loại vì điểm tuyệt đối < τCover 0.90. τ cao như vậy là để bài NGOÀI CSDL không bị
nhận nhầm (FMR 0,11%). Giả thuyết: bài có trong CSDL thì hạng 1 NỔI HẲN so với hạng
2, còn với bài lạ các ứng viên đầu sát nhau. Nếu đúng, luật

    chấp nhận  ⇔  s1 ≥ τ_abs   HOẶC   (s1 ≥ τ_low  VÀ  s1 − s2 ≥ δ)

nhận thêm được mà không tăng nhận nhầm. `s2` là điểm cao nhất của một bản ghi KHÁC
bản ghi hạng 1; biến thể `dedup` bỏ luôn các bản ghi trùng fingerprint với hạng 1
(corpus có bài bị nhập trùng dưới nhiều recording_id — khoảng cách giữa hai bản
trùng luôn ~0, luật khoảng cách sẽ loại oan).

Giao thức: đúng điều kiện server của EXP-07 (cắt theo COVER_TEMPO_FACTORS, toàn bộ
chỉ mục cover, HeldOutProtocol, 30 mẫu nhiễu). Chọn luật theo cùng thứ tự ràng buộc
của `pick_threshold`: không nhận mẫu nhiễu nào → FMR ≤ 0,5% → recall cao nhất. Để
khỏi tự chấm trên chính dữ liệu dùng để chọn, chia bài nguồn làm 2 nửa: chọn trên nửa
này, chấm trên nửa kia (và ngược lại) — split theo bài nguồn, không theo truy vấn (§8).

    python experiments/exp07_cover/margin_rule.py --sources 100
"""
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from backend import config
from backend.services.cover_service import (
    CHROMA_SR,
    best_scores,
    load_cover_index,
    query_descriptors,
    query_descriptors_from_file,
    query_spans,
)
from experiments.common import HeldOutProtocol, RESULTS_DIR, load_query_manifest, save_result
from experiments.exp07_cover.run import (
    FAMILIES,
    MAX_FALSE_MATCH,
    PREFERRED_FALSE_MATCH,
    RUNTIME_DURATION_S,
    noise_signals,
)

EXPERIMENT_ID = "exp07_cover_margin_rule"
CHECKPOINT = os.path.join(RESULTS_DIR, ".checkpoints", f"{EXPERIMENT_ID}.jsonl")

TAU_ABS = (config.COVER_THRESHOLD, None)            # None: bỏ hẳn nhánh điểm tuyệt đối
TAU_LOW = tuple(round(0.60 + 0.05 * i, 2) for i in range(7))    # 0.60 .. 0.90
DELTAS = tuple(round(0.01 * i, 2) for i in range(21))           # 0.00 .. 0.20
VARIANTS = ("any", "dedup")
FAMILIES_OUT = {**FAMILIES, "Chồng âm": ["audio_overlay"]}


def top_two(scores: np.ndarray, column_rec: list, columns_of: dict, group_of) -> tuple:
    """(cột hạng 1, s1, s2 khác bản ghi, s2 khác cả nhóm trùng fingerprint)."""
    top = int(scores.argmax())
    s1 = float(scores[top])
    rest = scores.copy()
    rest[columns_of[column_rec[top]]] = -np.inf
    s2_any = float(rest.max())
    for rec in group_of(column_rec[top]):
        rest[columns_of.get(rec, [])] = -np.inf
    s2_dedup = float(rest.max())
    return top, s1, s2_any, s2_dedup


def score_queries(manifest: list, index, protocol: HeldOutProtocol) -> list:
    column_rec = [str(rec) for rec in index.id_map]
    columns_of = {}
    for column, rec in enumerate(column_rec):
        columns_of.setdefault(rec, []).append(column)

    done = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                done[record["query_id"]] = record
    print(f"{len(done)} truy vấn đã có trong checkpoint")

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    records, started = [], time.time()
    with open(CHECKPOINT, "a", encoding="utf-8") as out:
        for i, row in enumerate(manifest, 1):
            if row["query_id"] in done:
                records.append(done[row["query_id"]])
                continue
            source = row["source_recording_id"]
            truth = protocol.exact_class(source)
            path = (row["path"] if os.path.isabs(row["path"])
                    else os.path.join(config.BASE_DIR, row["path"]))
            if not any(rec in columns_of for rec in truth) or not os.path.exists(path):
                continue
            descriptors, _ = query_descriptors_from_file(path, base_duration=RUNTIME_DURATION_S)
            scores = best_scores(descriptors, index.matrix)[0]

            top, s1, s2_any, s2_dedup = top_two(scores, column_rec, columns_of, protocol.exact_class)
            held = scores.copy()
            for rec in protocol.exclusions(row):
                held[columns_of.get(rec, [])] = -np.inf
            _, h1, h2_any, h2_dedup = top_two(held, column_rec, columns_of, protocol.exact_class)

            record = {
                "query_id": row["query_id"], "source": source,
                "transformation": row["transformation"],
                "known": {"s1": s1, "s2_any": s2_any, "s2_dedup": s2_dedup,
                          "correct": column_rec[top] in truth, "top_rec": column_rec[top]},
                "held_out": {"s1": h1, "s2_any": h2_any, "s2_dedup": h2_dedup},
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            records.append(record)
            if i % 100 == 0:
                rate = (time.time() - started) / max(len(records) - len(done), 1)
                print(f"  {i}/{len(manifest)} truy vấn, ~{rate:.2f} s/truy vấn", flush=True)
    return records


def score_noise(index, protocol) -> list:
    column_rec = [str(rec) for rec in index.id_map]
    columns_of = {}
    for column, rec in enumerate(column_rec):
        columns_of.setdefault(rec, []).append(column)
    longest = max(seconds for _, seconds in query_spans(RUNTIME_DURATION_S))
    out = []
    for name, signal in noise_signals(longest):
        scores = best_scores(query_descriptors(signal, CHROMA_SR, RUNTIME_DURATION_S)[0],
                             index.matrix)[0]
        _, s1, s2_any, s2_dedup = top_two(scores, column_rec, columns_of, protocol.exact_class)
        out.append({"name": name, "s1": s1, "s2_any": s2_any, "s2_dedup": s2_dedup})
    return out


def accepts(scores: dict, rule: dict) -> bool:
    s1, s2 = scores["s1"], scores[f"s2_{rule['variant']}"]
    if rule["tau_abs"] is not None and s1 >= rule["tau_abs"]:
        return True
    return s1 >= rule["tau_low"] and s1 - s2 >= rule["delta"]


def evaluate(records: list, noise: list, rule: dict) -> dict:
    accepted = correct = false_match = 0
    for r in records:
        if accepts(r["known"], rule):
            accepted += 1
            correct += int(r["known"]["correct"])
        false_match += int(accepts(r["held_out"], rule))
    n = len(records)
    precision = correct / accepted if accepted else 0.0
    recall = correct / n if n else 0.0
    return {
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "accepted": accepted, "correct": correct, "queries": n,
        "false_match_rate": round(false_match / n, 4) if n else 0.0, "false_matches": false_match,
        "noise_matches": sum(accepts(s, rule) for s in noise),
    }


def all_rules() -> list:
    rules = [{"tau_abs": config.COVER_THRESHOLD, "tau_low": 1.01, "delta": 0.0, "variant": "any",
              "name": "baseline"}]
    for variant in VARIANTS:
        for tau_abs in TAU_ABS:
            for tau_low in TAU_LOW:
                for delta in DELTAS:
                    rules.append({"tau_abs": tau_abs, "tau_low": tau_low, "delta": delta,
                                  "variant": variant})
    return rules


def pick(records: list, noise: list, rules: list) -> tuple:
    """Cùng thứ tự với EXP-07 pick_threshold: nhiễu 0 → FMR ≤ 0,5% (rồi 5%) → recall."""
    scored = [(rule, evaluate(records, noise, rule)) for rule in rules]
    for bound in (PREFERRED_FALSE_MATCH, MAX_FALSE_MATCH):
        safe = [(rule, m) for rule, m in scored
                if m["noise_matches"] == 0 and m["false_match_rate"] <= bound]
        if safe:
            # hoà recall -> precision cao hơn -> luật chặt hơn (δ lớn, τ_low cao)
            return max(safe, key=lambda item: (item[1]["recall"], item[1]["precision"],
                                               item[0]["delta"], item[0]["tau_low"])), bound
    return None, None


def per_family(records: list, rule: dict) -> dict:
    out = {}
    for family, transformations in FAMILIES_OUT.items():
        subset = [r for r in records if r["transformation"] in transformations]
        if subset:
            hit = sum(1 for r in subset if r["known"]["correct"] and accepts(r["known"], rule))
            out[family] = {"n": len(subset), "recall": round(hit / len(subset), 4)}
    return out


def margins(records: list, variant: str) -> dict:
    def stats(values):
        values = np.asarray(values)
        return ({"count": int(values.size),
                 **{f"p{q:02d}": round(float(np.percentile(values, q)), 4) for q in (5, 25, 50, 75, 95)}}
                if values.size else {"count": 0})
    key = f"s2_{variant}"
    return {
        "known_correct_below_tau": stats([r["known"]["s1"] - r["known"][key] for r in records
                                          if r["known"]["correct"]
                                          and r["known"]["s1"] < config.COVER_THRESHOLD]),
        "held_out_top1": stats([r["held_out"]["s1"] - r["held_out"][key] for r in records]),
        "held_out_top1_s1_ge_0.80": stats([r["held_out"]["s1"] - r["held_out"][key]
                                           for r in records if r["held_out"]["s1"] >= 0.80]),
    }


def error_breakdown(records: list, rule: dict, protocol: HeldOutProtocol) -> dict:
    """
    Nhận SAI bài (truy vấn có trong CSDL) và nhận NHẦM bài lạ (held-out), theo phép
    biến đổi. Với audio_overlay, "sai bài" có thể là nhận ra bài bị trộn chồng — cũng
    có mặt trong audio và trong CSDL — nên đếm riêng thay vì gộp vào lỗi.
    """
    wrong, false_match, overlay_partner = {}, {}, 0
    for r in records:
        if accepts(r["known"], rule) and not r["known"]["correct"]:
            wrong[r["transformation"]] = wrong.get(r["transformation"], 0) + 1
            partner = protocol.overlay.get(r["source"])
            if (r["transformation"] == "audio_overlay" and partner
                    and r["known"].get("top_rec") in protocol.same_audio(partner)):
                overlay_partner += 1
        if accepts(r["held_out"], rule):
            false_match[r["transformation"]] = false_match.get(r["transformation"], 0) + 1
    return {"wrong_accepts": dict(sorted(wrong.items())),
            "overlay_wrong_is_mixed_in_song": overlay_partner,
            "held_out_false_matches": dict(sorted(false_match.items()))}


def describe_rule(rule: dict) -> str:
    absolute = f"s1 ≥ {rule['tau_abs']} hoặc " if rule["tau_abs"] is not None else ""
    return (f"{absolute}(s1 ≥ {rule['tau_low']} và s1 − s2[{rule['variant']}] ≥ {rule['delta']})"
            if rule.get("name") != "baseline" else f"s1 ≥ {rule['tau_abs']} (production)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=100,
                        help="số bài nguồn đầu manifest (cùng bộ truy vấn EXP-04/07)")
    parser.add_argument("--limit", type=int, default=None,
                        help="chạy thử N truy vấn đầu; KHÔNG ghi file kết quả")
    args = parser.parse_args()

    index = load_cover_index()
    if index is None or not index.n_items:
        print("Chưa có chỉ mục cover. Chạy: python scripts/build_cover_index.py")
        return 1
    manifest = load_query_manifest(args.sources)
    if args.limit:
        manifest = manifest[:args.limit]
    protocol = HeldOutProtocol(manifest)
    print(f"{len(manifest)} truy vấn x chỉ mục {index.n_items} bài")

    records = score_queries(manifest, index, protocol)
    noise = score_noise(index, protocol)
    rules = all_rules()

    baseline = evaluate(records, noise, rules[0])
    best_rule, bound = pick(records, noise, rules[1:])

    sources = sorted({r["source"] for r in records})
    fold_of = {source: i % 2 for i, source in enumerate(sources)}
    cross = []
    for fold in (0, 1):
        tune = [r for r in records if fold_of[r["source"]] != fold]
        test = [r for r in records if fold_of[r["source"]] == fold]
        chosen, fold_bound = pick(tune, noise, rules[1:])
        cross.append({
            "fold": fold, "tune_queries": len(tune), "test_queries": len(test),
            "rule": chosen[0] if chosen else None, "bound": fold_bound,
            "test": evaluate(test, noise, chosen[0]) if chosen else None,
            "test_baseline": evaluate(test, noise, rules[0]),
        })

    # Bảng so sánh vài luật có tên. `conservative` KHÔNG do bộ chọn sinh ra: nửa 0 của
    # kiểm tra chéo tự chọn δ 0.08, nửa 1 chọn 0.05; ở 0.08 nhánh khoảng cách không
    # thêm lượt nhận nhầm held-out nào so với production — ghi rõ để người đọc biết
    # đây là lựa chọn SAU KHI thấy số, và số đọc chính vẫn là `cross_validated`.
    named = {"baseline": rules[0],
             "conservative": {"tau_abs": config.COVER_THRESHOLD, "tau_low": 0.70,
                              "delta": 0.08, "variant": "any"}}
    if best_rule:
        named["best_in_sample"] = best_rule[0]
    comparison = {
        name: {"rule_text": describe_rule(rule), **evaluate(records, noise, rule),
               "per_family_recall": per_family(records, rule),
               **error_breakdown(records, rule, protocol)}
        for name, rule in named.items()
    }

    metrics = {
        "rule_comparison": comparison,
        "baseline": {"rule": describe_rule(rules[0]), **baseline,
                     "per_family_recall": per_family(records, rules[0])},
        "best_in_sample": ({"rule": best_rule[0], "rule_text": describe_rule(best_rule[0]),
                            "fmr_bound": bound, **best_rule[1],
                            "per_family_recall": per_family(records, best_rule[0])}
                           if best_rule else None),
        "cross_validated": cross,
        "margin_distribution": {v: margins(records, v) for v in VARIANTS},
        "noise_max_s1": round(max(n["s1"] for n in noise), 4),
        "rules_evaluated": len(rules) - 1,
    }
    params = {
        "tau_cover_production": config.COVER_THRESHOLD,
        "cover_tempo_factors": list(config.COVER_TEMPO_FACTORS),
        "cover_index_items": index.n_items,
        "queries": len(records), "sources_selected": args.sources,
        "grid": {"tau_abs": list(TAU_ABS), "tau_low": list(TAU_LOW), "delta": list(DELTAS),
                 "variants": list(VARIANTS)},
        "selection": "nhiễu 0 -> FMR <= 0.005 (không có thì 0.05) -> recall -> precision -> luật chặt hơn",
        "cross_validation": "2 nửa theo bài nguồn (thứ tự sắp xếp recording_id, chẵn/lẻ)",
        "held_out_exclusions": protocol.describe(),
    }
    path = None if args.limit else save_result(EXPERIMENT_ID, params, metrics, notes=(
        "Chỉ đo, production vẫn dùng s1 >= tauCover. Số đọc chính là cross_validated.test: "
        "luật chọn trên một nửa bài nguồn, chấm trên nửa kia."))

    print(f"\nBaseline  {metrics['baseline']['rule']}: P {baseline['precision']} R {baseline['recall']} "
          f"FMR {baseline['false_match_rate']} nhiễu {baseline['noise_matches']}")
    if best_rule:
        m = best_rule[1]
        print(f"Tốt nhất  {describe_rule(best_rule[0])}: P {m['precision']} R {m['recall']} "
              f"FMR {m['false_match_rate']} nhiễu {m['noise_matches']}")
    for c in cross:
        if c["test"]:
            print(f"Chéo nửa {c['fold']}: {describe_rule(c['rule'])} -> R {c['test']['recall']} "
                  f"(baseline {c['test_baseline']['recall']}), FMR {c['test']['false_match_rate']} "
                  f"(baseline {c['test_baseline']['false_match_rate']})")
    print(f"Đã lưu {path}" if path else "Chạy thử (--limit): không ghi file kết quả")
    return 0


if __name__ == "__main__":
    sys.exit(main())
