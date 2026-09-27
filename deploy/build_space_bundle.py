"""
Dựng thư mục sẽ đẩy lên Hugging Face Space: chỉ những gì server cần lúc chạy.

    python deploy/build_space_bundle.py --out space_bundle

Space là một repo riêng mà Hugging Face build thành image, nên nó chỉ nhận đúng phần
cần chạy: không có data/processed (~170 MB CSV chỉ dùng để nạp CSDL), experiments/,
tests/, .claude/. Dockerfile và README (thẻ Space, có khối YAML cấu hình) lấy từ
deploy/space/ và đặt ở gốc gói, đúng chỗ Hugging Face tìm.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Đường dẫn tương đối so với gốc repo, chép nguyên cấu trúc
INCLUDE = (
    "backend",
    "frontend",
    "configs",
    "models",
    "init_db.py",
    "requirements.txt",
    "deploy/__init__.py",
    "deploy/fetch_runtime_data.py",
    "deploy/space/start.sh",
)
# Đặt ở gốc gói: (nguồn trong repo, tên trong gói)
AT_ROOT = (
    ("deploy/space/Dockerfile", "Dockerfile"),
    ("deploy/space/README.md", "README.md"),
)
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache")


def build(out_dir: Path, root: Path = ROOT) -> list:
    """Dựng gói vào `out_dir` (xoá sạch trước). Trả danh sách file, tương đối so với gói."""
    out_dir = Path(out_dir)
    # Thư mục đích bị xoá sạch: không được là repo, thư mục cha của repo hay thư mục nguồn
    resolved, repo = out_dir.resolve(), root.resolve()
    protected = {repo, repo / "deploy"} | {repo / rel.split("/")[0] for rel in INCLUDE}
    if resolved in protected or resolved in repo.parents:
        raise ValueError(f"Thư mục đích không hợp lệ: {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for rel in INCLUDE:
        source = root / rel
        if not source.exists():
            raise FileNotFoundError(f"Thiếu {rel} trong repo")
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=IGNORE)
        else:
            shutil.copy2(source, target)
    for source_rel, name in AT_ROOT:
        shutil.copy2(root / source_rel, out_dir / name)

    # start.sh phải giữ LF: CRLF làm `sh` báo lỗi "set: Illegal option -" trong container
    start = out_dir / "deploy/space/start.sh"
    start.write_bytes(start.read_bytes().replace(b"\r\n", b"\n"))

    return sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out", default="space_bundle", help="thư mục đích (bị xoá sạch trước)")
    args = parser.parse_args()
    files = build(Path(args.out))
    size = sum((Path(args.out) / f).stat().st_size for f in files)
    print(f"Đã dựng {len(files)} file ({size / 1e6:.1f} MB) vào {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
