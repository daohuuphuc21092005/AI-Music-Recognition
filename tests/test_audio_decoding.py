"""
Giải mã file đầu vào: định dạng nào phải qua FFmpeg, và FFmpeg lấy từ đâu.

Lỗi gặp thật (2026-09-27): libsndfile không giải mã được AAC, nên `.m4a` qua được
Chromaprint (fpcalc nhúng FFmpeg) nhưng tới MERT thì báo NO_AUDIO — người dùng chỉ
thấy "không đọc được file". Nay m4a/aac đi đường FFmpeg như video.
"""
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from backend.services import audio_service
from backend.services.analysis_pipeline import analyze_audio


@pytest.fixture(autouse=True)
def _clear_ffmpeg_cache():
    audio_service.ffmpeg_exe.cache_clear()
    yield
    audio_service.ffmpeg_exe.cache_clear()


@pytest.mark.parametrize("name, expected", [
    ("bai.m4a", True), ("BAI.AAC", True), ("clip.mp4", True), ("clip.mov", True),
    ("bai.mp3", False), ("bai.wav", False), ("bai.flac", False), ("bai.ogg", False),
])
def test_dinh_dang_can_ffmpeg(name, expected):
    assert audio_service.needs_ffmpeg(name) is expected


def test_m4a_duoc_giai_ma_qua_ffmpeg(monkeypatch, tmp_path):
    decoded = str(tmp_path / "bai_normalized.wav")
    calls = []
    monkeypatch.setattr(audio_service, "extract_and_normalize_audio",
                        lambda path: calls.append(path) or decoded)

    assert audio_service.prepare_for_embedding("x/bai.m4a", "bai.m4a") == (decoded, decoded)
    assert calls == ["x/bai.m4a"]
    # mp3: librosa đọc thẳng, không có file tạm
    assert audio_service.prepare_for_embedding("x/bai.mp3", "bai.mp3") == ("x/bai.mp3", None)


def test_thieu_ffmpeg_thi_m4a_bao_ro_ly_do(monkeypatch):
    monkeypatch.setattr(audio_service, "ffmpeg_available", lambda: False)
    with pytest.raises(audio_service.AudioProcessingError) as exc_info:
        audio_service.prepare_for_embedding("x/bai.m4a", "bai.m4a")
    assert exc_info.value.code == "MODEL_FAILURE"
    assert "m4a" in exc_info.value.message


def test_ffmpeg_lay_tu_imageio_khi_khong_co_tren_path(monkeypatch):
    fake = types.ModuleType("imageio_ffmpeg")
    fake.get_ffmpeg_exe = lambda: "C:/goi-pip/ffmpeg.exe"
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake)
    monkeypatch.setattr(audio_service.shutil, "which", lambda _name: None)
    monkeypatch.setattr(audio_service, "_runs", lambda exe: exe == "C:/goi-pip/ffmpeg.exe")

    assert audio_service.ffmpeg_exe() == "C:/goi-pip/ffmpeg.exe"
    assert audio_service.ffmpeg_available() is True


def test_ffmpeg_tren_path_hong_thi_khong_tinh(monkeypatch):
    """ffmpeg.exe có trên PATH nhưng không chạy (lỗi DLL của conda từng gặp)."""
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)   # import sẽ ném ImportError
    monkeypatch.setattr(audio_service.shutil, "which", lambda _name: "C:/conda/ffmpeg.exe")
    monkeypatch.setattr(audio_service, "_runs", lambda _exe: False)

    assert audio_service.ffmpeg_exe() is None
    assert audio_service.ffmpeg_available() is False


@patch("backend.services.analysis_pipeline.get_audio_duration", return_value=5.0)
@patch("backend.services.analysis_pipeline.process_music_query")
def test_dac_trung_san_xuat_do_tren_ban_giai_ma_truoc_khi_xoa(mock_cascade, _mock_dur,
                                                              monkeypatch, tmp_path):
    """
    Với m4a/video, bản giải mã là file tạm bị xoá cuối bước nhận diện. Trước đây đặc trưng
    sản xuất đo SAU khi xoá, rơi về file gốc librosa không đọc được -> evidence luôn None.
    """
    original = tmp_path / "bai.m4a"
    original.write_bytes(b"khong phai audio that")   # chỉ để qua kiểm tra đuôi/dung lượng
    decoded = tmp_path / "bai_normalized.wav"
    t = np.arange(5 * 24000) / 24000
    sf.write(decoded, (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), 24000)

    monkeypatch.setattr(audio_service, "prepare_for_embedding",
                        lambda _path, _name: (str(decoded), str(decoded)))
    mock_cascade.return_value = {"match_type": "UNKNOWN", "score": 0.1,
                                 "identity_confidence": 0.1, "recording_id": None}

    result = analyze_audio(str(original), filename="bai.m4a")

    assert mock_cascade.call_args.kwargs["mert_audio_path"] == str(decoded)
    assert result["evidence"]["production_features"] is not None
    assert not decoded.exists()   # file tạm vẫn được dọn
