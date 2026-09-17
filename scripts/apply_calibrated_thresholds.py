"""
Đọc ngưỡng đã hiệu chỉnh từ kết quả thí nghiệm rồi ghi vào `.env` (§4, §14).

Vì sao cần bước riêng này: EXP-01, EXP-06 và EXP-07 **đo ra** ngưỡng, còn EXP-04
và EXP-08 lại **tiêu thụ** ngưỡng qua `backend/config.py`. Chạy một lượt duy nhất
nghĩa là hai thí nghiệm quan trọng nhất vẫn dùng con số cũ — đúng điều §4 cấm
("không tự chọn threshold trước khi có dữ liệu"). `scripts/run_all_experiments.py`
gọi script này GIỮA pha hiệu chỉnh và pha tiêu thụ, rồi hỏi lại backend.config
trong một tiến trình mới và đối chiếu với identity_gate của configs/rules_v1.yaml.
Chạy tay thì:

    python scripts/apply_calibrated_thresholds.py --write    # ghi vào .env
    # sửa min_identity_confidence_by_match_type trong configs/rules_v1.yaml
    python scripts/run_all_experiments.py --only exp02,exp04,exp05,exp08,license

Ngưỡng lấy từ đâu:
  * τFP    <- EXP-01. F1 cao nhất TRONG SỐ các ngưỡng giữ tỉ lệ nhận nhầm ≤ 0.005:
             nhận nhầm một bản thu có bản quyền tốn kém hơn nhiều so với việc bỏ
             sót rồi để tầng MERT xử lý tiếp, nên siết nhận nhầm trước, tối đa hoá
             F1 sau.
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
      1. Lấy F1 cao nhất TRONG SỐ các ngưỡng đạt §16 (Precision ≥ 0.95) và giữ
         tỉ lệ nhận nhầm ≤ 0.005. Hoà F1 thì lấy ngưỡng THẤP hơn — trong nhóm đã
         siết nhận nhầm, ngưỡng thấp hơn cho recall cao hơn mà không phải trả thêm.
      2. Không ngưỡng nào đạt 0.005 -> nới về ràng buộc §16 (FPR ≤ 0.05).
      3. Không ngưỡng nào đạt §16 -> KHÔNG đề xuất gì, để người chạy tự quyết.

    Bản cũ có thêm một quy tắc đứng trước: "ngưỡng THẤP NHẤT còn giữ FPR = 0". Bỏ
    quy tắc đó vì trên corpus 24.375 bản ghi nó chọn τ = 0.95 (recall 0.4679) thay
    vì 0.30 (recall 0.5558) — đổi 8,8 điểm recall lấy một khác biệt mà dữ liệu
    KHÔNG phân biệt được: FPR đo trên 1.900 truy vấn, nên "0 lần nhận nhầm" và
    "0.0011" chỉ cách nhau 2 truy vấn, và khoảng tin cậy 95% của 0/1.900 vẫn kéo
    tới ~0.0016. Con số 0 ở đây là giới hạn của phép đo, không phải bằng chứng
    rằng rủi ro bằng 0.
    """
    data = load("exp01_fingerprint_baseline.json")
    if not data:
        return None, ""
    metrics = data.get("metrics", {})

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

    # Bỏ luôn các dòng gán TRÙNG cho cùng một khoá, không chỉ sửa dòng đầu tiên.
    # Bản cũ sửa dòng đầu rồi `pop` khỏi danh sách, nên bản sao phía dưới còn
    # nguyên — mà trình đọc .env lấy lần gán CUỐI. Hậu quả có thật: .env chứa cả
    # FP_THRESHOLD=0.3 (vừa hiệu chỉnh) lẫn FP_THRESHOLD=0.15 (cũ) và runtime chạy
    # 0.15 suốt, trong khi script vẫn báo "đã ghi" — sai một cách im lặng.
    remaining = dict(updates)
    seen, cleaned = set(), []
    for line in lines:
        match = re.match(r"\s*#?\s*([A-Z_]+)\s*=", line)
        key = match.group(1) if match else None
        if key in updates:
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(f"{key}={updates[key]}")
            remaining.pop(key, None)
        else:
            cleaned.append(line)
    lines = cleaned

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
    print("Bước tiếp: đồng bộ min_identity_confidence_by_match_type trong "
          "configs/rules_v1.yaml, rồi")
    print("  python scripts/run_all_experiments.py --only exp02,exp04,exp05,exp08,license")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
