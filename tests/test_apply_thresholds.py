"""
`write_env` phải để lại ĐÚNG MỘT dòng gán cho mỗi khoá ngưỡng.

Vì sao đáng một bộ test riêng: bản cũ chỉ sửa dòng xuất hiện ĐẦU TIÊN rồi bỏ khoá
khỏi danh sách, nên bản sao phía dưới còn nguyên. Trình đọc .env lấy lần gán CUỐI,
nên .env chứa cả `FP_THRESHOLD=0.3` (vừa hiệu chỉnh) lẫn `FP_THRESHOLD=0.15` (cũ)
thì runtime chạy 0.15 — trong khi script vẫn in "đã ghi 3 ngưỡng". Lỗi im lặng
kiểu này sống sót cả ngày trong phiên 2026-09-16 và suýt làm mọi kết luận về
ngưỡng trở thành vô nghĩa.
"""
import importlib.util
import os

import pytest

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "apply_calibrated_thresholds.py")


def load_module():
    spec = importlib.util.spec_from_file_location("apply_thresholds", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assignments(path: str, key: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip().startswith(f"{key}=")]


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "DATABASE_URL=postgresql://user:pass@localhost:5432/db\n"
        "FP_THRESHOLD=0.3\n"
        "AUDIO_ROOT=D:/music-rights-data/audio\n"
        "FP_THRESHOLD=0.15\n"
        "MERT_THRESHOLD=0.97\n",
        encoding="utf-8",
    )
    module = load_module()
    monkeypatch.setattr(module, "ENV_PATH", str(path))
    return module, str(path)


def test_khoa_trung_bi_go_chi_con_mot_dong(env_file):
    module, path = env_file
    module.write_env({"FP_THRESHOLD": 0.30})

    lines = assignments(path, "FP_THRESHOLD")
    assert lines == ["FP_THRESHOLD=0.3"], f"còn dòng trùng: {lines}"


def test_khoa_chua_co_thi_them_moi(env_file):
    module, path = env_file
    module.write_env({"COVER_THRESHOLD": 0.90})

    assert assignments(path, "COVER_THRESHOLD") == ["COVER_THRESHOLD=0.9"]


def test_khong_dung_toi_cac_dong_khac(env_file):
    """File .env chứa cả mật khẩu CSDL — script chỉ được chạm đúng khoá ngưỡng."""
    module, path = env_file
    module.write_env({"FP_THRESHOLD": 0.30, "MERT_THRESHOLD": 0.98})

    content = open(path, encoding="utf-8").read()
    assert "DATABASE_URL=postgresql://user:pass@localhost:5432/db" in content
    assert "AUDIO_ROOT=D:/music-rights-data/audio" in content
    assert assignments(path, "MERT_THRESHOLD") == ["MERT_THRESHOLD=0.98"]
