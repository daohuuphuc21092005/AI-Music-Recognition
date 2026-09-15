"""
Đọc ngưỡng đã hiệu chỉnh từ kết quả thí nghiệm rồi ghi vào `.env` (§4, §14).

Vì sao cần bước riêng này: EXP-01, EXP-06 và EXP-07 **đo ra** ngưỡng, còn EXP-04
và EXP-08 lại **tiêu thụ** ngưỡng qua `backend/config.py`. Chạy một lượt duy nhất
nghĩa là hai thí nghiệm quan trọng nhất vẫn dùng con số cũ — đúng điều §4 cấm
("không tự chọn threshold trước khi có dữ liệu"). Quy trình đúng là hai lượt:

    python scripts/run_all_experiments.py                    # lượt 1: đo
    python scripts/apply_calibrated_thresholds.py --write    # ghi vào .env
    python scripts/run_all_experiments.py --only exp04,exp05,exp08   # lượt 2

Ngưỡng lấy từ đâu:
  * τFP    <- EXP-01. Ưu tiên ngưỡng THẤP NHẤT còn giữ FPR = 0 thay vì ngưỡng cho
             F1 cao nhất: nhận nhầm một bản thu có bản quyền tốn kém hơn nhiều so
             với việc bỏ sót rồi để tầng MERT xử lý tiếp.
  * τMERT  <- EXP-06, mục tiêu False Match Rate ≤ 5% của §16.
  * τCover <- EXP-07, `recommended_tau_cover`.

Không đoán: thiếu file kết quả hoặc thiếu khoá thì báo rõ và bỏ qua ngưỡng đó,
KHÔNG ghi một giá trị bịa vào .env.
"""
import argparse
import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config

RESULTS_DIR = os.path.join(config.BASE_DIR, "experiments", "results")
ENV_PATH = os.path.join(config.BASE_DIR, ".env")


def load(name: str):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"⏭️  chưa có {name} — bỏ qua ngưỡng lấy từ đây")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Ngưỡng nghiệm thu §16 cho tầng nhận dạng chính xác. Dùng thẳng hai con số này
# làm RÀNG BUỘC khi chọn τFP, thay vì tối đa hoá F1 rồi mới xem có đạt hay không:
# F1 cao nhất có thể rơi vào một ngưỡng nhận nhầm nhiều hơn mức dự án chấp nhận.
MIN_PRECISION = 0.95      # §16: clean exact-match Precision >= 0.95
MAX_FALSE_MATCH = 0.05    # §16: False Match Rate <= 5% (ràng buộc CỨNG)

# Ngưỡng nhận nhầm MONG MUỐN, chặt hơn §16 mười lần. Lý do không dừng ở §16: trên
# corpus thật, MỌI ngưỡng trong dải quét đều đạt §16, nên §16 một mình không phân
# biệt được gì và quy tắc "F1 cao nhất" sẽ chọn ngưỡng thấp nhất — tức là nhận
# nhầm nhiều nhất. Trong hệ thống bản quyền, hai loại lỗi KHÔNG cân xứng: nhận
# nhầm cho ra thẳng một kết luận sai về quyền, còn bỏ sót thì truy vấn vẫn rơi
# xuống tầng MERT để xử lý tiếp. Vì vậy siết tỉ lệ nhận nhầm trước, rồi mới tối
# đa hoá F1 trong phần còn lại.
PREFERRED_FALSE_MATCH = 0.005


def tau_fp() -> tuple:
    """
    (giá trị, giải thích) cho τFP, lấy từ EXP-01.

    Quy tắc chọn, theo đúng thứ tự:
      1. Có ngưỡng nào giữ FPR = 0 thì lấy ngưỡng THẤP NHẤT trong số đó — giữ
         recall cao nhất mà vẫn không nhận nhầm lần nào.
      2. Không có thì lấy F1 cao nhất TRONG SỐ các ngưỡng đạt cả hai điều kiện
         §16. Hoà F1 thì lấy ngưỡng cao hơn (nhận nhầm tốn kém hơn bỏ sót: bỏ
         sót còn được tầng MERT xử lý tiếp, nhận nhầm thì ra thẳng kết luận sai).
      3. Không ngưỡng nào đạt §16 -> KHÔNG đề xuất gì, để người chạy tự quyết.
    """
    data = load("exp01_fingerprint_baseline.json")
    if not data:
        return None, ""
    metrics = data.get("metrics", {})

    zero_fpr = metrics.get("lowest_threshold_with_zero_fpr")
    if zero_fpr:
        return zero_fpr["threshold"], (
            f"ngưỡng thấp nhất giữ FPR = 0 (P={zero_fpr.get('precision')}, "
            f"R={zero_fpr.get('recall')}, F1={zero_fpr.get('f1')})")

    sweep = metrics.get("sweep") or []
    passing = [r for r in sweep
               if r.get("precision", 0) >= MIN_PRECISION
               and r.get("false_positive_rate", 1) <= MAX_FALSE_MATCH]
    if not passing:
        print(f"⚠️  EXP-01: không ngưỡng nào đạt §16 (P>={MIN_PRECISION}, "
              f"FPR<={MAX_FALSE_MATCH}) -> giữ nguyên τFP, cần xem lại dữ liệu")
        return None, ""

    preferred = [r for r in passing
                 if r["false_positive_rate"] <= PREFERRED_FALSE_MATCH]
    pool, label = (preferred, f"FPR<={PREFERRED_FALSE_MATCH}") if preferred else (
        passing, f"§16 FPR<={MAX_FALSE_MATCH}")
    if not preferred:
        print(f"⚠️  EXP-01: không ngưỡng nào đạt FPR <= {PREFERRED_FALSE_MATCH} "
              f"-> nới về ràng buộc §16")

    # Hoà F1 thì lấy ngưỡng THẤP hơn: trong nhóm đã siết FPR, ngưỡng thấp hơn cho
    # recall cao hơn mà không phải trả thêm gì về tỉ lệ nhận nhầm.
    best = max(pool, key=lambda r: (r["f1"], -r["threshold"]))
    return best["threshold"], (
        f"F1 cao nhất trong {len(pool)}/{len(sweep)} ngưỡng đạt {label} "
        f"(P={best['precision']}, R={best['recall']}, F1={best['f1']}, "
        f"FPR={best['false_positive_rate']})")


def tau_mert() -> tuple:
    """(giá trị, giải thích) cho τMERT, lấy từ EXP-06 theo mục tiêu FMR ≤ 5%."""
    data = load("exp06_unknown_detection.json")
    if not data:
        return None, ""
    recommended = data.get("metrics", {}).get("recommended") or {}

    for key in ("fmr<=0.05", "fmr<=0.01"):
        row = recommended.get(key)
        if row:
            return row["threshold"], (
                f"{key} (FMR={row.get('false_match_rate')}, "
                f"Unknown Recall={row.get('unknown_recall')}, "
                f"True Accept={row.get('true_accept_rate')})")
    print("⚠️  EXP-06 không đạt được mục tiêu FMR nào -> giữ nguyên τMERT")
    return None, ""


def tau_cover() -> tuple:
    """(giá trị, giải thích) cho τCover, lấy từ EXP-07."""
    data = load("exp07_cover.json")
    if not data:
        return None, ""
    best = data.get("metrics", {}).get("recommended_tau_cover")
    if not best:
        print("⚠️  EXP-07 không có 'recommended_tau_cover'")
        return None, ""
    return best["threshold"], (f"P={best.get('precision')}, R={best.get('recall')}, "
                               f"F1={best.get('f1')}")


def write_env(updates: dict) -> None:
    """
    Cập nhật .env tại chỗ: sửa dòng đã có, thêm dòng còn thiếu.

    Giữ nguyên mọi dòng khác (comment, DATABASE_URL, AUDIO_ROOT...) — file này do
    người dùng sở hữu, script chỉ được chạm đúng các khoá ngưỡng.
    """
    lines = []
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, encoding="utf-8") as f:
            lines = f.read().splitlines()

    remaining = dict(updates)
    for index, line in enumerate(lines):
        match = re.match(r"\s*#?\s*([A-Z_]+)\s*=", line)
        key = match.group(1) if match else None
        if key in remaining:
            lines[index] = f"{key}={remaining.pop(key)}"

    if remaining:
        lines.append("")
        lines.append("# Ngưỡng hiệu chỉnh từ thí nghiệm (apply_calibrated_thresholds.py)")
        lines.extend(f"{k}={v}" for k, v in remaining.items())

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Ghi vào .env (mặc định chỉ hiển thị)")
    args = parser.parse_args()

    current = {
        "FP_THRESHOLD": config.FP_THRESHOLD,
        "MERT_THRESHOLD": config.MERT_THRESHOLD,
        "COVER_THRESHOLD": config.COVER_THRESHOLD,
    }
    measured = {
        "FP_THRESHOLD": tau_fp(),
        "MERT_THRESHOLD": tau_mert(),
        "COVER_THRESHOLD": tau_cover(),
    }

    print("\n" + "=" * 78)
    print("  NGƯỠNG HIỆU CHỈNH TỪ THÍ NGHIỆM")
    print("=" * 78)

    updates = {}
    for key, (value, why) in measured.items():
        old = current[key]
        if value is None:
            print(f"  {key:<16} {old}  ->  (giữ nguyên, chưa đo được)")
            continue
        mark = "=" if abs(float(value) - float(old)) < 1e-9 else "->"
        print(f"  {key:<16} {old}  {mark}  {value}")
        print(f"  {'':<16} vì {why}")
        if mark == "->":
            updates[key] = value

    if not updates:
        print("\nKhông có ngưỡng nào thay đổi.")
        return 0

    if not args.write:
        print(f"\n{len(updates)} ngưỡng sẽ đổi. Thêm --write để ghi vào {ENV_PATH}.")
        return 0

    write_env(updates)
    print(f"\n💾 Đã ghi {len(updates)} ngưỡng vào {ENV_PATH}")
    print("Bước tiếp: python scripts/run_all_experiments.py --only exp04,exp05,exp08")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
