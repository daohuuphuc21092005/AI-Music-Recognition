"""
Tải các file chỉ mục mà server cần lúc chạy nhưng không nằm được trong git.

    python deploy/fetch_runtime_data.py

Dùng trên Hugging Face Spaces (deploy/space/start.sh gọi trước uvicorn). Nguồn là một
dataset repo trên Hugging Face, do `deploy/upload_runtime_data.py` đẩy lên một lần từ
máy có sẵn các file này:

  RUNTIME_DATA_REPO   vd. "ten-ban/amr-runtime-data" (bắt buộc; trống thì bỏ qua)
  HF_TOKEN            token ĐỌC, cần khi dataset để private

File đích lấy đúng đường dẫn trong backend/config.py, nên server không cần biết file
được tải về từ đâu. Chỉ mục Cover là tuỳ chọn: thiếu thì tầng Cover đứng ngoài, như
khi chạy trên máy dev không có file đó.

Mã thoát: 0 = đủ file bắt buộc (hoặc không cấu hình nguồn), 1 = thiếu file bắt buộc.
"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import config  # noqa: E402

# (tên file trong dataset repo, đường dẫn đích, bắt buộc?)
RUNTIME_FILES = (
    ("mert_faiss.index", config.FAISS_INDEX_PATH, True),
    ("faiss_id_map.json", config.FAISS_ID_MAP_PATH, True),
    ("cover_descriptors.npy", config.COVER_INDEX_PATH, False),
    ("cover_id_map.json", config.COVER_ID_MAP_PATH, False),
)


def fetch(repo_id: str, token: str = None, download=None) -> list:
    """
    Tải từng file về đúng chỗ. Trả danh sách file BẮT BUỘC còn thiếu.
    `download(repo_id, filename, token)` -> đường dẫn file đã tải (tham số để test).
    """
    if download is None:
        download = _hf_download
    listed = _list_files(repo_id, token) if download is _hf_download else None
    missing = []
    for name, target, required in RUNTIME_FILES:
        if listed is not None and name not in listed:
            if required:
                missing.append(name)
            print(f"-> {name}: không có trong {repo_id}{' (BẮT BUỘC)' if required else ', bỏ qua'}")
            continue
        try:
            source = download(repo_id, name, token)
        except Exception as exc:   # mạng, quyền truy cập, file không tồn tại
            print(f"-> {name}: tải thất bại ({type(exc).__name__})")
            if required:
                missing.append(name)
            continue
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        if os.path.abspath(source) != os.path.abspath(target):
            shutil.copyfile(source, target)
        print(f"-> {name}: {os.path.getsize(target) / 1e6:.1f} MB -> {target}")
    return missing


def _list_files(repo_id: str, token: str = None):
    from huggingface_hub import HfApi
    try:
        return set(HfApi(token=token).list_repo_files(repo_id, repo_type="dataset"))
    except Exception as exc:
        print(f"-> không liệt kê được {repo_id} ({type(exc).__name__}); thử tải thẳng")
        return None


def _hf_download(repo_id: str, filename: str, token: str = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=token)


def main() -> int:
    repo_id = os.environ.get("RUNTIME_DATA_REPO", "").strip()
    if not repo_id:
        print("RUNTIME_DATA_REPO trống: không tải chỉ mục (dùng file sẵn có trên đĩa, nếu có)")
        return 0
    missing = fetch(repo_id, os.environ.get("HF_TOKEN") or None)
    if missing:
        print(f"THIẾU file bắt buộc: {', '.join(missing)}. /health sẽ báo faiss INDEX_LOAD_FAILED.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
