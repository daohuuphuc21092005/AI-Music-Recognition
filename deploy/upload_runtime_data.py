"""
Đẩy chỉ mục FAISS (và chỉ mục Cover nếu có) lên một dataset repo trên Hugging Face,
để Space tải về lúc khởi động (`deploy/fetch_runtime_data.py`). Chạy MỘT lần, và
chạy lại mỗi khi dựng lại chỉ mục:

    hf auth login                      # hoặc đặt biến HF_TOKEN (token có quyền GHI)
    python deploy/upload_runtime_data.py --repo ten-ban/amr-runtime-data

Repo được tạo ở chế độ PRIVATE nếu chưa có. Khi đó Space cần một token ĐỌC (secret
`HF_TOKEN` của Space). Thêm `--public` nếu muốn ai cũng tải được: file chỉ chứa vector
MERT của các bài FMA (giấy phép Creative Commons) và mã recording_id, không chứa audio.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.fetch_runtime_data import RUNTIME_FILES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--repo", required=True, help="dataset repo, vd. ten-ban/amr-runtime-data")
    parser.add_argument("--public", action="store_true", help="tạo repo công khai (mặc định private)")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    present = [(name, target) for name, target, _ in RUNTIME_FILES if os.path.exists(target)]
    required_missing = [name for name, target, required in RUNTIME_FILES
                        if required and not os.path.exists(target)]
    if required_missing:
        print(f"Thiếu {', '.join(required_missing)}. Dựng trước: "
              "python scripts/rebuild_faiss_index.py --from-db")
        return 1

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    for name, target in present:
        print(f"-> {name} ({os.path.getsize(target) / 1e6:.1f} MB)")
        api.upload_file(path_or_fileobj=target, path_in_repo=name, repo_id=args.repo,
                        repo_type="dataset", commit_message=f"Cập nhật {name}")
    skipped = [name for name, target, _ in RUNTIME_FILES if not os.path.exists(target)]
    if skipped:
        print(f"Không có trên máy này, bỏ qua: {', '.join(skipped)}")
    print(f"Xong: https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
