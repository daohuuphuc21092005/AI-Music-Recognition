"""Kiểm thử trích xuất embedding MERT."""
import numpy as np
import pytest

from backend import config
from backend.services.embedding_service import extract_mert_embedding


def test_extract_mert_embedding(test_audio):
    vector = extract_mert_embedding(test_audio)

    assert vector is not None
    assert isinstance(vector, np.ndarray)
    assert vector.shape[0] == config.EMBEDDING_DIM

    # Vector phải chuẩn hoá L2 thì inner product mới đúng là cosine similarity
    assert pytest.approx(float(np.linalg.norm(vector)), 0.01) == 1.0


def test_file_hong_tra_none(tmp_path):
    broken = tmp_path / "khong-phai-audio.mp3"
    broken.write_bytes(b"day khong phai du lieu audio")
    assert extract_mert_embedding(str(broken)) is None
