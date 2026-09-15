"""
Fixture dùng chung cho bộ test.

Nguyên tắc: thiếu phụ thuộc ngoài (PostgreSQL, fpcalc, file audio) thì SKIP kèm
lý do rõ ràng, không để test đỏ vì lý do không liên quan tới code.
"""
import os
import shutil

import pytest

from backend import config
from backend.database.session import SessionLocal, check_connection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_AUDIO = os.path.join(ROOT, "test.mp3")
TEST_AUDIO_ALT = os.path.join(ROOT, "test_audio", "bai_hat_test.mp3")


def _db_available() -> bool:
    ok, _ = check_connection()
    return ok


requires_db = pytest.mark.skipif(
    not _db_available(),
    reason=f"Không kết nối được PostgreSQL ({config.DATABASE_URL.split('@')[-1]}). "
           f"Tạo file .env từ .env.example rồi điền mật khẩu.",
)

requires_fpcalc = pytest.mark.skipif(
    shutil.which("fpcalc") is None,
    reason="Chưa có fpcalc trong PATH "
           "(https://github.com/acoustid/chromaprint/releases)",
)

requires_index = pytest.mark.skipif(
    not (os.path.exists(config.FAISS_INDEX_PATH)
         and os.path.exists(config.FAISS_ID_MAP_PATH)),
    reason="Chưa có FAISS index. Chạy: python scripts/rebuild_faiss_index.py",
)


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session")
def test_audio() -> str:
    if not os.path.exists(TEST_AUDIO):
        pytest.skip(f"Không có file audio mẫu: {TEST_AUDIO}")
    return TEST_AUDIO


@pytest.fixture(scope="session")
def vector_index():
    from backend.services.retrieval_service import load_index
    if not os.path.exists(config.FAISS_INDEX_PATH):
        pytest.skip("Chưa có FAISS index. Chạy: python scripts/rebuild_faiss_index.py")
    return load_index(config.FAISS_INDEX_PATH, config.FAISS_ID_MAP_PATH)


@pytest.fixture(scope="session")
def unmatched_audio(tmp_path_factory) -> str:
    """
    File audio chắc chắn KHÔNG khớp tầng 1 (nhiễu trắng 8 giây).

    Cần fixture riêng vì cả test.mp3 lẫn bai_hat_test.mp3 đều nằm sẵn trong
    reference database nên luôn dừng ở Chromaprint, không bao giờ chạm tầng 2.
    """
    import numpy as np
    import soundfile as sf

    sr = 22050
    noise = np.random.RandomState(42).normal(0, 0.2, sr * 8).astype("float32")
    path = tmp_path_factory.mktemp("audio") / "white_noise.wav"
    sf.write(str(path), noise, sr)
    return str(path)
