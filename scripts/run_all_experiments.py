"""
Chạy toàn bộ 8 thí nghiệm §15 theo ĐÚNG thứ tự phụ thuộc.

Vì sao cần script này thay vì gõ 8 lệnh: thứ tự không tuỳ ý. EXP-05 không chạy
suy luận nào cả, nó đọc kết quả đã lưu của EXP-01 và EXP-04; EXP-07 đọc kết quả
EXP-03 để so sánh chroma với MERT trên cùng bộ cửa sổ. Chạy sai thứ tự thì hai
thí nghiệm đó hoặc chết, hoặc âm thầm so sánh với số liệu của lần chạy trước —
tức là đối chiếu hai điều kiện khác nhau mà tưởng là một.

Và chúng phải chạy TUẦN TỰ, không song song: mọi thí nghiệm đều đo độ trễ, chạy
chồng nhau trên cùng CPU là làm hỏng chính con số mình đang đo.

Dùng:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --only exp03,exp07
    python scripts/run_all_experiments.py --skip exp02,exp06
    python scripts/run_all_experiments.py --list
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config

# (khoá, thư mục, mô tả, phụ thuộc kết quả của thí nghiệm nào)
EXPERIMENTS = [
    ("exp01", "exp01_fingerprint", "Chromaprint baseline -> hiệu chỉnh τFP", []),
    ("exp02", "exp02_mert", "MERT retrieval: Recall@K, MRR, mAP", []),
    ("exp03", "exp03_pooling", "mean vs mean+std pooling", []),
    ("exp04", "exp04_hybrid", "FP vs MERT vs Cascade (thí nghiệm chính)", []),
    ("exp05", "exp05_robustness", "Tổng hợp độ bền theo phép biến đổi", ["exp01", "exp04"]),
    ("exp06", "exp06_unknown", "Unknown detection -> hiệu chỉnh τMERT", []),
    ("exp07", "exp07_cover", "Chroma/CQT + OTI -> hiệu chỉnh τCover", ["exp03"]),
    ("exp08", "exp08_end_to_end", "Trọn pipeline -> Macro-F1, confusion matrix", []),
]

INDEX = {key: (folder, desc, deps) for key, folder, desc, deps in EXPERIMENTS}


def parse_keys(value: str) -> list:
    keys = [k.strip() for k in (value or "").split(",") if k.strip()]
    unknown = [k for k in keys if k not in INDEX]
    if unknown:
        raise SystemExit(f"Không có thí nghiệm: {', '.join(unknown)}. "
                         f"Hợp lệ: {', '.join(INDEX)}")
    return keys


def run_one(key: str) -> tuple:
    folder, desc, _deps = INDEX[key]
    script = os.path.join(config.BASE_DIR, "experiments", folder, "run.py")

    print("\n" + "=" * 78)
    print(f"  {key.upper()} — {desc}")
    print("=" * 78, flush=True)

    start = time.perf_counter()
    # -u để output hiện ngay khi chạy nền, không bị đệm tới lúc kết thúc
    code = subprocess.run([sys.executable, "-u", script], cwd=config.BASE_DIR).returncode
    return code, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Chỉ chạy các thí nghiệm này (phân tách bằng dấu phẩy)")
    parser.add_argument("--skip", help="Bỏ qua các thí nghiệm này")
    parser.add_argument("--list", action="store_true", help="Liệt kê rồi thoát")
    parser.add_argument("--keep-going", action="store_true",
                        help="Chạy tiếp khi có thí nghiệm lỗi (mặc định dừng lại)")
    args = parser.parse_args()

    if args.list:
        for key, _folder, desc, deps in EXPERIMENTS:
            suffix = f"  (cần: {', '.join(deps)})" if deps else ""
            print(f"  {key}  {desc}{suffix}")
        return 0

    selected = parse_keys(args.only) if args.only else list(INDEX)
    for key in parse_keys(args.skip) if args.skip else []:
        if key in selected:
            selected.remove(key)

    # Giữ nguyên thứ tự chuẩn dù người dùng gõ --only theo thứ tự nào
    ordered = [key for key, _f, _d, _dep in EXPERIMENTS if key in selected]

    # Cảnh báo khi phụ thuộc bị bỏ ra ngoài lượt chạy: kết quả cũ vẫn dùng được,
    # nhưng người chạy phải biết mình đang đối chiếu với dữ liệu của lần trước.
    for key in ordered:
        missing = [d for d in INDEX[key][2] if d not in ordered]
        for dep in missing:
            path = os.path.join(config.BASE_DIR, "experiments", "results")
            print(f"⚠️  {key} đọc kết quả của {dep} nhưng {dep} không nằm trong lượt "
                  f"chạy này — nó sẽ dùng file cũ trong {path}")

    print(f"Sẽ chạy {len(ordered)} thí nghiệm, tuần tự: {', '.join(ordered)}")
    print(f"τFP = {config.FP_THRESHOLD} | τMERT = {config.MERT_THRESHOLD} | "
          f"τCover = {config.COVER_THRESHOLD}")

    results, total_start = [], time.perf_counter()
    for key in ordered:
        code, seconds = run_one(key)
        results.append((key, code, seconds))
        if code != 0 and not args.keep_going:
            print(f"\n❌ {key} thất bại (mã {code}). Dừng lại — các thí nghiệm sau "
                  f"có thể đọc kết quả của nó. Dùng --keep-going để chạy tiếp.")
            break

    print("\n" + "=" * 78)
    print("  TỔNG KẾT")
    print("=" * 78)
    for key, code, seconds in results:
        status = "OK  " if code == 0 else f"LỖI {code}"
        print(f"  {status}  {key}  {seconds / 60:6.1f} phút   {INDEX[key][1]}")
    print(f"\nTổng thời gian: {(time.perf_counter() - total_start) / 60:.1f} phút")

    failed = [k for k, c, _ in results if c != 0]
    if failed:
        print(f"Thất bại: {', '.join(failed)}")
        return 1
    print(f"Kết quả lưu ở experiments/results/ ({len(results)} file JSON đã cập nhật)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
