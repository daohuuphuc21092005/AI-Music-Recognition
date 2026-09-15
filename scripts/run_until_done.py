"""
Chạy một thí nghiệm cho tới khi HOÀN TẤT, tự khởi động lại sau mỗi lần bị ngắt.

Vì sao cần: trên máy thiếu RAM, tiến trình nền bị hệ thống giết giữa chừng là
chuyện thường xuyên. Các thí nghiệm dài đều đã có checkpoint nên không mất tiến
độ, nhưng vẫn phải có ai đó bấm chạy lại — script này làm việc đó.

Hai điều kiện an toàn, và cái thứ hai mới là điều dễ sai:

  * Dừng khi file kết quả JSON được ghi mới hơn lúc bắt đầu. Không dựa vào mã
    thoát: tiến trình bị SIGKILL cũng cho mã khác 0 y như lỗi thật.
  * KHÔNG BAO GIỜ chạy hai tiến trình của cùng một thí nghiệm. Chúng sẽ cùng
    append vào một file checkpoint và các dòng có thể xen kẽ nhau — hỏng đúng
    thứ sinh ra để bảo vệ tiến độ. Trước mỗi lần khởi động đều quét tiến trình
    và bỏ qua nếu thí nghiệm đó đang chạy.

Dùng:
    python scripts/run_until_done.py exp08_end_to_end
    python scripts/run_until_done.py exp03_pooling --max-restarts 60
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config

RESULTS_DIR = os.path.join(config.BASE_DIR, "experiments", "results")

# Thư mục thí nghiệm -> tên file kết quả. Không suy từ tên thư mục vì hai bên
# không theo cùng quy ước đặt tên.
RESULT_FILES = {
    "exp01_fingerprint": "exp01_fingerprint_baseline.json",
    "exp02_mert": "exp02_mert_retrieval.json",
    "exp03_pooling": "exp03_pooling.json",
    "exp04_hybrid": "exp04_hybrid_cascade.json",
    "exp05_robustness": "exp05_robustness.json",
    "exp06_unknown": "exp06_unknown_detection.json",
    "exp07_cover": "exp07_cover.json",
    "exp08_end_to_end": "exp08_end_to_end.json",
    "exp09_license_learnability": "exp09_license_learnability.json",
}


def running_pids(folder: str) -> list:
    """PID của các tiến trình đang chạy chính script thí nghiệm này."""
    try:
        import psutil
    except ImportError:
        return []

    marker = os.path.join(folder, "run.py").replace("\\", "/")
    me = os.getpid()
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = process.info["cmdline"] or []
        except Exception:
            continue
        if process.info["pid"] == me:
            continue
        # So theo ĐƯỜNG DẪN script, không so chuỗi tự do: mọi lệnh kiểm tra có
        # chứa tên thí nghiệm trong dòng lệnh đều sẽ khớp nhầm nếu so lỏng.
        if any(arg.replace("\\", "/").endswith(marker) for arg in cmdline):
            found.append(process.info["pid"])
    return found


def result_mtime(folder: str) -> float:
    path = os.path.join(RESULTS_DIR, RESULT_FILES[folder])
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


def checkpoint_count(folder: str) -> int:
    name = os.path.splitext(RESULT_FILES[folder])[0] + ".jsonl"
    path = os.path.join(RESULTS_DIR, ".checkpoints", name)
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        return max(sum(1 for _ in f) - 1, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=sorted(RESULT_FILES))
    parser.add_argument("--max-restarts", type=int, default=80)
    parser.add_argument("--wait", type=float, default=15.0,
                        help="Giây chờ giữa hai lần khởi động lại")
    args = parser.parse_args()

    folder = args.experiment
    script = os.path.join(config.BASE_DIR, "experiments", folder, "run.py")
    if not os.path.exists(script):
        print(f"❌ Không thấy {script}")
        return 1

    baseline_mtime = result_mtime(folder)
    started = time.perf_counter()

    for attempt in range(1, args.max_restarts + 1):
        existing = running_pids(folder)
        if existing:
            print(f"[{attempt}] {folder} đang chạy sẵn (pid {existing}) — chờ",
                  flush=True)
            while running_pids(folder):
                time.sleep(args.wait)
        else:
            done = checkpoint_count(folder)
            print(f"[{attempt}] khởi động {folder}"
                  f"{f' (checkpoint: {done})' if done else ''}", flush=True)
            subprocess.run([sys.executable, "-u", script], cwd=config.BASE_DIR)

        if result_mtime(folder) > baseline_mtime:
            minutes = (time.perf_counter() - started) / 60
            print(f"\n✅ {folder} HOÀN TẤT sau {attempt} lượt, {minutes:.1f} phút")
            print(f"   {os.path.join(RESULTS_DIR, RESULT_FILES[folder])}")
            return 0

        time.sleep(args.wait)

    print(f"\n⚠️  Đã thử {args.max_restarts} lượt mà {folder} chưa xong. "
          f"Checkpoint hiện tại: {checkpoint_count(folder)} — chạy lại để đi tiếp.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
