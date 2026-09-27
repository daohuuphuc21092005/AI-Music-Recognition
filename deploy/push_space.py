"""
Đẩy gói do build_space_bundle.py dựng lên Hugging Face Space và đặt cấu hình của Space.

    python deploy/push_space.py space_bundle

Chạy trong GitHub Actions (.github/workflows/deploy-space.yml). Biến môi trường:

  HF_TOKEN              token GHI của tài khoản Hugging Face (bắt buộc)
  HF_SPACE_ID           vd. "ten-ban/music-rights-ai" (bắt buộc)
  SPACE_DATABASE_URL    chuỗi kết nối Neon của role music_ai_app -> secret DATABASE_URL
  SPACE_HF_READ_TOKEN   token ĐỌC cho dataset private -> secret HF_TOKEN của Space
  RUNTIME_DATA_REPO     dataset chứa chỉ mục FAISS -> variable của Space

Biến nào trống thì giữ nguyên giá trị Space đang có, để lỡ thiếu một secret trên
GitHub không xoá mất cấu hình đang chạy. Space được tạo CÔNG KHAI nếu chưa có.
Secret không bao giờ được in ra log.
"""
import os
import sys

SECRETS = (("SPACE_DATABASE_URL", "DATABASE_URL"), ("SPACE_HF_READ_TOKEN", "HF_TOKEN"))
VARIABLES = (("RUNTIME_DATA_REPO", "RUNTIME_DATA_REPO"),)


def main() -> int:
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        print("Cách dùng: python deploy/push_space.py <thư mục gói>")
        return 2
    token = os.environ.get("HF_TOKEN", "").strip()
    space_id = os.environ.get("HF_SPACE_ID", "").strip()
    if not token or not space_id:
        print("Thiếu HF_TOKEN hoặc HF_SPACE_ID")
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)

    for env_name, space_name in SECRETS:
        value = os.environ.get(env_name, "").strip()
        if value:
            api.add_space_secret(space_id, space_name, value)
            print(f"-> secret {space_name}: đã đặt")
        else:
            print(f"-> secret {space_name}: {env_name} trống, giữ nguyên trên Space")
    for env_name, space_name in VARIABLES:
        value = os.environ.get(env_name, "").strip()
        if value:
            api.add_space_variable(space_id, space_name, value)
            print(f"-> variable {space_name} = {value}")

    sha = os.environ.get("GITHUB_SHA", "")[:7]
    # delete_patterns="*": file đã bỏ khỏi gói cũng bị xoá trên Space (trừ .gitattributes)
    api.upload_folder(folder_path=sys.argv[1], repo_id=space_id, repo_type="space",
                      delete_patterns="*",
                      commit_message=f"Triển khai từ GitHub {sha}".strip())
    print(f"Xong: https://huggingface.co/spaces/{space_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
