"""Kiểm thử trích xuất embedding MERT."""
import numpy as np
import pytest

from backend import config
from backend.services import embedding_service
from backend.services.embedding_service import (
    EmbeddingAudioError,
    EmbeddingModelError,
    extract_mert_embedding,
)


def test_extract_mert_embedding(test_audio):
    vector = extract_mert_embedding(test_audio)

    assert vector is not None
    assert isinstance(vector, np.ndarray)
    assert vector.shape[0] == config.EMBEDDING_DIM

    # Vector phải chuẩn hoá L2 thì inner product mới đúng là cosine similarity
    assert pytest.approx(float(np.linalg.norm(vector)), 0.01) == 1.0


def test_file_hong_bao_loi_audio(tmp_path):
    """File không giải mã được là lỗi của FILE -> cascade map sang NO_AUDIO."""
    broken = tmp_path / "khong-phai-audio.mp3"
    broken.write_bytes(b"day khong phai du lieu audio")
    with pytest.raises(EmbeddingAudioError):
        extract_mert_embedding(str(broken))


def test_model_khong_nap_duoc_bao_loi_model_khong_do_cho_file(monkeypatch, tmp_path):
    """
    Model hỏng là lỗi phía MÁY CHỦ. Bản cũ nuốt lỗi thành None nên người dùng
    nhận "file không có audio" cho một sự cố không liên quan gì tới file của họ.
    """
    def model_hong():
        raise OSError("khong tai duoc trong so")

    monkeypatch.setattr(embedding_service, "load_model", model_hong)
    with pytest.raises(EmbeddingModelError):
        extract_mert_embedding(str(tmp_path / "bat-ky.mp3"))
