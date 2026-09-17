"""
EXP-05 — Robustness (§15): biến đổi âm thanh ảnh hưởng thế nào tới từng tầng.

Script này KHÔNG chạy lại suy luận; nó đọc kết quả đã lưu của EXP-01 và EXP-04
rồi dựng bảng "phép biến đổi × hiệu năng" dùng cho báo cáo (§19).

Chạy trước:
    python experiments/exp01_fingerprint/run.py
    python experiments/exp04_hybrid/run.py
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.common import RESULTS_DIR, print_table, save_result

EXPERIMENT_ID = "exp05_robustness"

# Nhóm các phép biến đổi theo bản chất tác động lên tín hiệu
FAMILIES = {
    "Cắt đoạn": ["original_crop30s", "crop_15s", "crop_10s"],
    "Nén codec": ["mp3_128k", "mp3_64k", "aac_96k"],
    "Nhiễu": ["noise_snr20", "noise_snr10", "noise_snr5"],
    "Biên độ / EQ": ["gain_minus12db", "eq_lowpass_4k"],
    "Dịch cao độ": ["pitch_plus_1", "pitch_minus_1", "pitch_plus_2", "pitch_minus_2"],
    "Đổi tốc độ": ["tempo_0_90", "tempo_0_95", "tempo_1_05", "tempo_1_10"],
    "Chồng âm": ["audio_overlay"],
}
# Dưới ngưỡng này thì coi như tầng đó KHÔNG dùng được cho cả nhóm biến đổi.
# Chọn 5% chứ không phải 0: xem lý do đầy đủ ở chỗ tính điểm mù bên dưới.
BLIND_SPOT_MAX_ACCURACY = 0.05


def load(name: str):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(f"❌ Thiếu kết quả {name}. Chạy thí nghiệm đó trước.")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def query_set(result: dict) -> set:
    return {(r["source_recording_id"], r["transformation"])
            for r in result.get("raw_results") or []}


def consistency_problems(exp01: dict, exp04: dict) -> list:
    """
    EXP-05 ghép số liệu của HAI lượt chạy khác nhau vào một bảng, nên chỉ hợp lệ
    khi hai lượt dùng cùng bộ truy vấn. Từng có lúc hai file kết quả không chung
    một truy vấn nào (0/1.900) mà bảng vẫn in ra bình thường.

    τFP được phép khác: EXP-01 chạy TRƯỚC khi ghi ngưỡng đã hiệu chỉnh, EXP-04 chạy
    SAU. Mọi con số "đúng/sai" của bảng này lấy từ EXP-04 nên dùng đúng τ hiện hành.
    """
    problems = []
    q01, q04 = query_set(exp01), query_set(exp04)
    if not q04:
        problems.append("EXP-04 không lưu raw_results nên không đối chiếu được bộ truy vấn")
    elif q01 != q04:
        problems.append(
            f"bộ truy vấn khác nhau: EXP-01 {len(q01)}, EXP-04 {len(q04)}, "
            f"chung {len(q01 & q04)}")
    return problems


def main() -> int:
    exp01 = load("exp01_fingerprint_baseline")
    exp04 = load("exp04_hybrid_cascade")
    if not exp01 or not exp04:
        return 1
    problems = consistency_problems(exp01, exp04)
    if problems:
        print("❌ Không ghép được EXP-01 với EXP-04: " + "; ".join(problems)
              + ". Chạy lại cả hai trên cùng manifest và cùng ngưỡng.")
        return 1
    tau01 = exp01.get("parameters", {}).get("current_tau_fp")
    if tau01 != exp04["parameters"].get("tau_fp"):
        print(f"ℹ️  EXP-01 chạy với τFP {tau01}, EXP-04 với {exp04['parameters'].get('tau_fp')}"
              " — số đúng/sai trong bảng lấy từ EXP-04.")

    fp_detail = exp01["metrics"]["per_transformation"]
    hybrid = exp04["metrics"]["per_transformation"]
    has_cover = all("cover_correct" in v for v in hybrid.values())

    rows, family_metrics = [], {}
    for family, names in FAMILIES.items():
        present = [n for n in names if n in hybrid]
        if not present:
            continue

        n = sum(hybrid[t]["n"] for t in present)
        fp_ok = sum(hybrid[t]["fingerprint_correct"] for t in present)
        mert_ok = sum(hybrid[t]["mert_correct"] for t in present)
        cascade_ok = sum(hybrid[t]["cascade_correct"] for t in present)
        cover_ok = sum(hybrid[t].get("cover_correct", 0) for t in present)
        no_cover_ok = sum(hybrid[t].get("cascade_no_cover_correct", cascade_ok)
                          for t in present) if has_cover else cascade_ok
        fp_score = sum(hybrid[t]["fingerprint_mean_score"] * hybrid[t]["n"]
                       for t in present) / n
        mert_score = sum(hybrid[t]["mert_mean_score"] * hybrid[t]["n"]
                         for t in present) / n

        rows.append([
            family, n,
            f"{fp_ok}/{n} ({fp_ok / n * 100:.0f}%)",
            f"{mert_ok}/{n} ({mert_ok / n * 100:.0f}%)",
            *([f"{cover_ok}/{n} ({cover_ok / n * 100:.0f}%)"] if has_cover else []),
            f"{cascade_ok}/{n} ({cascade_ok / n * 100:.0f}%)",
            round(fp_score, 3), round(mert_score, 3),
        ])
        family_metrics[family] = {
            "n": n,
            "fingerprint_accuracy": round(fp_ok / n, 4),
            "mert_accuracy": round(mert_ok / n, 4),
            "cascade_accuracy": round(cascade_ok / n, 4),
            **({"cover_accuracy": round(cover_ok / n, 4),
                "cascade_no_cover_accuracy": round(no_cover_ok / n, 4)} if has_cover else {}),
            "fingerprint_mean_score": round(fp_score, 4),
            "mert_mean_score": round(mert_score, 4),
            "transformations": present,
        }

    print_table(
        "EXP-05 — Độ bền theo nhóm phép biến đổi",
        rows,
        ["Nhóm biến đổi", "N", "Chromaprint", "MERT", *(["Cover"] if has_cover else []),
         "Cascade", "Điểm FP TB", "Điểm MERT TB"],
    )

    # Nhóm nào là điểm mù của từng tầng?
    #
    # KHÔNG dùng `== 0`. Trên nhóm "Dịch cao độ", MERT nhận đúng 1/401 truy vấn
    # (accuracy 0.0025): khác 0 về mặt số học, nên phép so sánh bằng 0 kết luận
    # "MERT không có điểm mù" — trong khi đây đúng là nhóm mà CẢ HAI tầng cùng
    # thất bại, tức phát hiện chính của đóng góp C2. Đoán trúng 1 lần trên 401
    # không phải là năng lực, và ranh giới `== 0` còn phụ thuộc may rủi cỡ mẫu:
    # bớt đi đúng truy vấn đó là nhóm này lại thành "mù hoàn toàn".
    #
    # Ngưỡng dưới đây là QUY ƯỚC ĐỌC KẾT QUẢ, không phải tham số hiệu chỉnh —
    # nó không đi vào bất kỳ quyết định nào của hệ thống, chỉ quyết định khi nào
    # bảng báo cáo gọi một nhóm là điểm mù. Ghi kèm vào file kết quả để người
    # đọc biết nhãn này nghĩa là gì.
    def blind(key):
        return [f for f, m in family_metrics.items()
                if m[key] <= BLIND_SPOT_MAX_ACCURACY]

    fp_blind = blind("fingerprint_accuracy")
    mert_blind = blind("mert_accuracy")
    both_blind = [f for f in fp_blind if f in mert_blind]
    cover_blind = blind("cover_accuracy") if has_cover else []
    all_blind = [f for f in both_blind if f in cover_blind] if has_cover else both_blind

    def fmt(names, key):
        if not names:
            return "không có"
        return ", ".join(f"{f} ({family_metrics[f][key]:.2%})" for f in names)

    print("")
    print(f"Điểm mù (tầng nhận đúng <= {BLIND_SPOT_MAX_ACCURACY:.0%} cả nhóm):")
    print(f"  Chromaprint : {fmt(fp_blind, 'fingerprint_accuracy')}")
    print(f"  MERT        : {fmt(mert_blind, 'mert_accuracy')}")
    print(f"  Cả hai tầng : {fmt(both_blind, 'cascade_accuracy')}")
    if has_cover:
        print(f"  Cover       : {fmt(cover_blind, 'cover_accuracy')}")
        print(f"  Cả ba tầng  : {fmt(all_blind, 'cascade_accuracy')}")
        rescued = [f for f in both_blind
                   if family_metrics[f]["cascade_accuracy"]
                   > family_metrics[f]["cascade_no_cover_accuracy"]]
        for f in rescued:
            m = family_metrics[f]
            print(f"  -> {f}: Chromaprint và MERT cùng bó tay, tầng Cover đưa cascade từ "
                  f"{m['cascade_no_cover_accuracy']:.2%} lên {m['cascade_accuracy']:.2%}")

    if both_blind and not has_cover:
        names = ", ".join(both_blind)
        worst = ", ".join(f"{family_metrics[f]['cascade_accuracy']:.2%}" for f in both_blind)
        print("")
        print(f"⚠️  {names}: cả hai tầng đều bó tay nên CASCADE CŨNG KHÔNG CỨU ĐƯỢC "
              f"(cascade accuracy {worst}). Đây chính là khoảng trống mà module "
              f"Cover/Version Identification (EXP-07) phải xử lý — dịch cao độ "
              f"đổi cả fingerprint lẫn phân bố embedding, nên thêm dữ liệu hay "
              f"hạ ngưỡng đều không giải quyết được.")

    print("\nĐộ suy giảm điểm số so với nhóm 'Cắt đoạn' (không biến dạng tín hiệu):")
    if "Cắt đoạn" in family_metrics:
        baseline = family_metrics["Cắt đoạn"]
        for family, m in family_metrics.items():
            if family == "Cắt đoạn":
                continue
            fp_drop = m["fingerprint_mean_score"] - baseline["fingerprint_mean_score"]
            mert_drop = m["mert_mean_score"] - baseline["mert_mean_score"]
            print(f"  {family:16} Chromaprint {fp_drop:+.3f} | MERT {mert_drop:+.3f}")

    total_queries = sum(m["n"] for m in family_metrics.values())
    sizes = sorted({hybrid[t]["n"]
                    for m in family_metrics.values() for t in m["transformations"]})
    per_t = str(sizes[0]) if len(sizes) == 1 else f"{sizes[0]}-{sizes[-1]}"

    path = save_result(
        EXPERIMENT_ID,
        params={
            "source_experiments": ["exp01_fingerprint_baseline", "exp04_hybrid_cascade"],
            "families": {k: v for k, v in FAMILIES.items()},
            "tau_fp": exp04["parameters"]["tau_fp"],
            "tau_mert": exp04["parameters"]["tau_mert"],
            "tau_cover": exp04["parameters"].get("tau_cover"),
            "queries_shared_by_exp01_and_exp04": len(query_set(exp01)),
            "blind_spot_max_accuracy": BLIND_SPOT_MAX_ACCURACY,
        },
        metrics={
            "by_family": family_metrics,
            # Ghi kèm ĐỊNH NGHĨA để không ai phải đoán nhãn "điểm mù" nghĩa là gì.
            "blind_spot_definition": {
                "max_accuracy": BLIND_SPOT_MAX_ACCURACY,
                "rule": "tầng nhận đúng <= ngưỡng này trên cả nhóm thì coi là điểm mù",
            },
            "fingerprint_blind_spots": fp_blind,
            "mert_blind_spots": mert_blind,
            "both_blind_spots": both_blind,
            "cover_blind_spots": cover_blind,
            "all_tiers_blind_spots": all_blind,
            "has_cover_stage": has_cover,
            # Kèm số thật, vì danh sách tên không cho biết "mù" tới mức nào.
            "blind_spot_accuracy": {
                f: {
                    "fingerprint": family_metrics[f]["fingerprint_accuracy"],
                    "mert": family_metrics[f]["mert_accuracy"],
                    "cascade": family_metrics[f]["cascade_accuracy"],
                    **({"cover": family_metrics[f]["cover_accuracy"]} if has_cover else {}),
                }
                for f in sorted(set(fp_blind) | set(mert_blind) | set(cover_blind))
            },
            "per_transformation_fingerprint": fp_detail,
            "per_transformation_hybrid": hybrid,
        },
        notes=(
            f"Tổng hợp lại từ EXP-01 và EXP-04, không chạy lại suy luận. Quy mô "
            f"{total_queries} truy vấn, {per_t} mẫu cho mỗi phép biến đổi. "
            f"Nhãn 'điểm mù' = tầng nhận đúng <= "
            f"{BLIND_SPOT_MAX_ACCURACY:.0%} trên cả nhóm."
        ),
    )
    print(f"\n💾 Đã lưu kết quả: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
