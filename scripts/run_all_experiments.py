"""
Chạy toàn bộ thí nghiệm §15 theo ĐÚNG thứ tự phụ thuộc.

Thứ tự không tuỳ ý, và có hai loại phụ thuộc:

1. Phụ thuộc KẾT QUẢ: EXP-05 không chạy suy luận, nó đọc kết quả đã lưu của EXP-01
   và EXP-04; EXP-07 đọc EXP-03 để so chroma với MERT trên cùng bộ cửa sổ.
2. Phụ thuộc NGƯỠNG: EXP-01 / EXP-06 / EXP-07 ĐO ra τFP / τMERT / τCover, còn
   EXP-02 / EXP-04 / EXP-08 TIÊU THỤ ngưỡng qua `backend/config.py`. Chạy chung một
   lượt thì các thí nghiệm tiêu thụ dùng ngưỡng CŨ.

Nên mỗi lượt chạy đủ đi qua bốn pha:

    hiệu chỉnh  : exp01, exp06, exp03, exp07
    ghi ngưỡng  : scripts/apply_calibrated_thresholds.py --write
    kiểm tra    : hỏi lại backend.config trong một TIẾN TRÌNH MỚI, đối chiếu với
                  identity_gate của configs/rules_v1.yaml — lệch thì dừng
    tiêu thụ    : exp02, exp04, exp05, exp08, rồi sweep license classifier

Bước kiểm tra tồn tại vì một lỗi có thật: `.env` từng có khoá trùng nên script ghi
ngưỡng báo "đã ghi" mà runtime vẫn chạy ngưỡng cũ suốt một ngày. Và rules_v1.yaml
phải đổi CÙNG LÚC với .env (EXACT_MATCH = τFP, NEAR_MATCH = τMERT, COVER_MATCH =
τCover), nếu không cascade nhận một khớp rồi Rule Engine vứt chính khớp đó.

Chạy TUẦN TỰ, không song song: mọi thí nghiệm đều đo độ trễ, chạy chồng nhau trên
cùng máy là làm hỏng chính con số mình đang đo (và máy này không đủ RAM).

Dùng:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --only exp04,exp05
    python scripts/run_all_experiments.py --skip exp02,exp06
    python scripts/run_all_experiments.py --no-apply     # không ghi ngưỡng mới
    python scripts/run_all_experiments.py --list
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config

CALIBRATE, CONSUME = "hiệu chỉnh", "tiêu thụ"

# (khoá, lệnh, mô tả, pha, phụ thuộc kết quả của bước nào)
STEPS = [
    ("exp01", ["experiments/exp01_fingerprint/run.py"],
     "Chromaprint baseline -> đo τFP", CALIBRATE, []),
    ("exp06", ["experiments/exp06_unknown/run.py"],
     "Unknown detection -> đo τMERT", CALIBRATE, []),
    ("exp03", ["experiments/exp03_pooling/run.py"],
     "mean vs mean+std pooling", CALIBRATE, []),
    ("exp07", ["experiments/exp07_cover/run.py"],
     "Chroma/CQT + OTI -> đo τCover", CALIBRATE, ["exp03"]),
    ("exp02", ["experiments/exp02_mert/run.py"],
     "MERT retrieval: Recall@K, MRR, mAP", CONSUME, []),
    ("exp04", ["experiments/exp04_hybrid/run.py"],
     "FP vs MERT vs Cover vs Cascade (thí nghiệm chính)", CONSUME, []),
    ("exp05", ["experiments/exp05_robustness/run.py"],
     "Tổng hợp độ bền theo phép biến đổi", CONSUME, ["exp01", "exp04"]),
    ("exp08", ["experiments/exp08_end_to_end/run.py"],
     "Trọn pipeline -> Macro-F1, confusion matrix", CONSUME, []),
    ("license", ["scripts/train_license_classifier.py", "--sweep"],
     "Sweep độ phức tạp license classifier", CONSUME, []),
]

INDEX = {key: (command, desc, phase, deps) for key, command, desc, phase, deps in STEPS}


def parse_keys(value: str) -> list:
    keys = [k.strip() for k in (value or "").split(",") if k.strip()]
    unknown = [k for k in keys if k not in INDEX]
    if unknown:
        raise SystemExit(f"Không có bước: {', '.join(unknown)}. "
                         f"Hợp lệ: {', '.join(INDEX)}")
    return keys


def run_command(title: str, command: list) -> tuple:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78, flush=True)
    start = time.perf_counter()
    # -u để output hiện ngay khi chạy nền, không bị đệm tới lúc kết thúc
    code = subprocess.run([sys.executable, "-u", *command], cwd=config.BASE_DIR).returncode
    return code, time.perf_counter() - start


def runtime_thresholds() -> dict:
    """Ngưỡng mà một tiến trình MỚI thực sự đọc được — không tin tiến trình hiện tại."""
    probe = ("import json; from backend import config; print(json.dumps({"
             "'EXACT_MATCH': config.FP_THRESHOLD, 'NEAR_MATCH': config.MERT_THRESHOLD, "
             "'COVER_MATCH': config.COVER_THRESHOLD}))")
    out = subprocess.run([sys.executable, "-c", probe], cwd=config.BASE_DIR,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def gate_thresholds() -> dict:
    from backend.services.decision_service import load_rules

    return dict(load_rules()["identity_gate"].get("min_identity_confidence_by_match_type") or {})


def threshold_mismatches(runtime: dict, gate: dict) -> list:
    return [f"{key}: config {runtime[key]} ≠ rules_v1.yaml {gate.get(key)}"
            for key in runtime
            if gate.get(key) is None or abs(float(runtime[key]) - float(gate[key])) > 1e-9]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Chỉ chạy các bước này (phân tách bằng dấu phẩy)")
    parser.add_argument("--skip", help="Bỏ qua các bước này")
    parser.add_argument("--no-apply", action="store_true",
                        help="Không ghi ngưỡng mới vào .env giữa hai pha")
    parser.add_argument("--list", action="store_true", help="Liệt kê rồi thoát")
    parser.add_argument("--keep-going", action="store_true",
                        help="Chạy tiếp khi có bước lỗi (mặc định dừng lại)")
    args = parser.parse_args()

    if args.list:
        for key, _command, desc, phase, deps in STEPS:
            suffix = f"  (cần: {', '.join(deps)})" if deps else ""
            print(f"  [{phase}] {key:8} {desc}{suffix}")
        return 0

    selected = parse_keys(args.only) if args.only else list(INDEX)
    for key in parse_keys(args.skip) if args.skip else []:
        if key in selected:
            selected.remove(key)
    # Giữ nguyên thứ tự chuẩn dù người dùng gõ --only theo thứ tự nào
    ordered = [key for key, *_ in STEPS if key in selected]

    for key in ordered:
        for dep in INDEX[key][3]:
            if dep not in ordered:
                print(f"⚠️  {key} đọc kết quả của {dep} nhưng {dep} không nằm trong lượt "
                      f"chạy này — sẽ dùng file cũ trong experiments/results/")

    calibrate = [k for k in ordered if INDEX[k][2] == CALIBRATE]
    consume = [k for k in ordered if INDEX[k][2] == CONSUME]
    print(f"Pha hiệu chỉnh: {', '.join(calibrate) or '(không)'}")
    print(f"Pha tiêu thụ  : {', '.join(consume) or '(không)'}")
    print(f"τFP = {config.FP_THRESHOLD} | τMERT = {config.MERT_THRESHOLD} | "
          f"τCover = {config.COVER_THRESHOLD}")

    results, total_start = [], time.perf_counter()

    def run_steps(keys) -> bool:
        for key in keys:
            command, desc, _phase, _deps = INDEX[key]
            code, seconds = run_command(f"{key.upper()} — {desc}", command)
            results.append((key, code, seconds))
            if code != 0 and not args.keep_going:
                print(f"\n❌ {key} thất bại (mã {code}). Dừng lại — các bước sau có thể "
                      f"đọc kết quả của nó. Dùng --keep-going để chạy tiếp.")
                return False
        return True

    ok = run_steps(calibrate)

    if ok and consume:
        if calibrate and not args.no_apply:
            code, seconds = run_command("Ghi ngưỡng đã hiệu chỉnh vào .env",
                                        ["scripts/apply_calibrated_thresholds.py", "--write"])
            results.append(("apply", code, seconds))
            ok = code == 0
        if ok:
            runtime, gate = runtime_thresholds(), gate_thresholds()
            print(f"\nNgưỡng tiến trình mới đọc được: {runtime}")
            print(f"identity_gate trong rules_v1.yaml : {gate}")
            mismatches = threshold_mismatches(runtime, gate)
            if mismatches:
                print("\n❌ Ngưỡng runtime lệch Rule Engine: " + "; ".join(mismatches)
                      + ".\n   Sửa min_identity_confidence_by_match_type trong "
                        "configs/rules_v1.yaml rồi chạy lại pha tiêu thụ:\n"
                        f"   python scripts/run_all_experiments.py --only {','.join(consume)}")
                results.append(("verify", 1, 0.0))
                ok = False
        if ok:
            ok = run_steps(consume)

    print("\n" + "=" * 78)
    print("  TỔNG KẾT")
    print("=" * 78)
    for key, code, seconds in results:
        status = "OK  " if code == 0 else f"LỖI {code}"
        label = INDEX[key][1] if key in INDEX else key
        print(f"  {status}  {key:8} {seconds / 60:6.1f} phút   {label}")
    print(f"\nTổng thời gian: {(time.perf_counter() - total_start) / 60:.1f} phút")

    failed = [k for k, c, _ in results if c != 0]
    if failed or not ok:
        print(f"Thất bại: {', '.join(failed) or 'dừng giữa chừng'}")
        return 1
    print("Kết quả lưu ở experiments/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
