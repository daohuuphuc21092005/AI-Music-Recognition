import os
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from backend import config
from backend.services.window_scan import plan_windows, slice_audio, get_audio_duration
from backend.services.analysis_pipeline import analyze_audio


def test_plan_windows_short_audio():
    """Audio shorter than or equal to window size should return [0.0]."""
    assert plan_windows(20.0, window_s=30, max_windows=8) == [0.0]
    assert plan_windows(30.0, window_s=30, max_windows=8) == [0.0]


def test_plan_windows_long_audio():
    """Audio of 249s with max 8 windows should produce 8 points from 0 to 219."""
    pts = plan_windows(249.0, window_s=30, max_windows=8)
    assert len(pts) == 8
    assert pts[0] == 0.0
    assert abs(pts[-1] - (249.0 - 30.0)) < 1e-4  # 219.0
    for i in range(len(pts) - 1):
        assert pts[i] < pts[i + 1]  # strictly increasing
    for p in pts:
        assert p + 30.0 <= 249.0 + 1e-4


def test_plan_windows_intermediate():
    """Audio of 45s needs ceil(45/30)=2 windows."""
    pts = plan_windows(45.0, window_s=30, max_windows=8)
    assert len(pts) == 2
    assert pts[0] == 0.0
    assert pts[1] == 15.0


def test_slice_audio_creates_valid_wav():
    """Test slice_audio produces a readable wav file."""
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        sr = 22050
        data = np.zeros((sr * 5, 2), dtype=np.float32)
        sf.write(path, data, sr)

        dur = get_audio_duration(path)
        assert abs(dur - 5.0) < 0.1

        slice_path = slice_audio(path, start_s=1.0, duration_s=2.0)
        assert os.path.exists(slice_path)
        slice_dur = get_audio_duration(slice_path)
        assert abs(slice_dur - 2.0) < 0.1
        if os.path.exists(slice_path):
            os.remove(slice_path)
    finally:
        if os.path.exists(path):
            os.remove(path)


@patch("backend.services.analysis_pipeline.get_audio_duration", return_value=240.0)
@patch("backend.services.analysis_pipeline.slice_audio")
@patch("backend.services.analysis_pipeline.process_music_query")
@patch("backend.services.analysis_pipeline.get_full_music_rights")
def test_multiwindow_match_at_window_5(mock_rights, mock_cascade, mock_slice, mock_dur, tmp_path):
    dummy_wav = str(tmp_path / "test.wav")
    open(dummy_wav, "w").close()
    mock_slice.return_value = dummy_wav

    # 4 windows return UNKNOWN, 5th returns NEAR_MATCH
    responses = [
        {"match_type": "UNKNOWN", "score": 0.2, "recording_id": None},
        {"match_type": "UNKNOWN", "score": 0.3, "recording_id": None},
        {"match_type": "UNKNOWN", "score": 0.1, "recording_id": None},
        {"match_type": "UNKNOWN", "score": 0.25, "recording_id": None},
        {"match_type": "NEAR_MATCH", "score": 0.95, "recording_id": "rec-w5", "identity_confidence": 0.95},
        {"match_type": "UNKNOWN", "score": 0.1, "recording_id": None},
        {"match_type": "UNKNOWN", "score": 0.1, "recording_id": None},
        {"match_type": "UNKNOWN", "score": 0.1, "recording_id": None},
    ]
    mock_cascade.side_effect = responses
    mock_rights.return_value = {
        "recording_id": "rec-w5",
        "rights_and_licensing": {
            "license_type": "CC_BY",
            "copyright_status": "COPYRIGHTED",
            "attribution_required": True,
            "commercial_use_allowed": True,
            "monetization_allowed": True,
            "source": "jamendo_api",
        },
        "track_metadata": {"recording_id": "rec-w5", "recording_title": "Track 5", "artist": "Artist 5"},
    }

    res = analyze_audio(dummy_wav, filename="test.wav", usage_context={"platform": "youtube", "commercial_use": False, "monetization": False})
    assert res["match"]["type"] == "NEAR_MATCH"
    assert res["identity"]["recording_id"] == "rec-w5"
    assert "windows" in res["evidence"]
    assert res["evidence"]["windows_scanned"] == 8


@patch("backend.services.analysis_pipeline.get_audio_duration", return_value=120.0)
@patch("backend.services.analysis_pipeline.slice_audio")
@patch("backend.services.analysis_pipeline.process_music_query")
@patch("backend.services.analysis_pipeline.get_full_music_rights")
def test_multiwindow_exact_match_early_stopping(mock_rights, mock_cascade, mock_slice, mock_dur, tmp_path):
    dummy_wav = str(tmp_path / "test.wav")
    open(dummy_wav, "w").close()
    mock_slice.return_value = dummy_wav

    # 1st window returns EXACT_MATCH -> should stop immediately
    mock_cascade.return_value = {
        "match_type": "EXACT_MATCH",
        "score": 0.05,
        "identity_confidence": 0.99,
        "recording_id": "rec-exact",
    }
    mock_rights.return_value = {
        "recording_id": "rec-exact",
        "rights_and_licensing": {
            "license_type": "CC_BY",
            "copyright_status": "COPYRIGHTED",
            "attribution_required": True,
            "commercial_use_allowed": True,
            "monetization_allowed": True,
            "source": "jamendo_api",
        },
        "track_metadata": {"recording_id": "rec-exact", "recording_title": "Exact Track", "artist": "Artist"},
    }

    res = analyze_audio(dummy_wav, filename="test.wav", usage_context={"platform": "youtube", "commercial_use": False, "monetization": False})
    assert res["match"]["type"] == "EXACT_MATCH"
    assert res["identity"]["recording_id"] == "rec-exact"
    assert mock_cascade.call_count == 1
    assert res["evidence"]["stopped_early"] is True


@patch("backend.services.analysis_pipeline.get_audio_duration", return_value=25.0)
@patch("backend.services.analysis_pipeline.process_music_query")
@patch("backend.services.analysis_pipeline.get_full_music_rights")
def test_multiwindow_short_audio_single_call(mock_rights, mock_cascade, mock_dur, tmp_path):
    dummy_wav = str(tmp_path / "test.wav")
    open(dummy_wav, "w").close()

    mock_cascade.return_value = {
        "match_type": "EXACT_MATCH",
        "score": 0.05,
        "identity_confidence": 0.99,
        "recording_id": "rec-short",
    }
    mock_rights.return_value = {
        "recording_id": "rec-short",
        "rights_and_licensing": {
            "license_type": "CC0",
            "copyright_status": "PUBLIC_DOMAIN",
            "attribution_required": False,
            "commercial_use_allowed": True,
            "monetization_allowed": True,
            "source": "fma_metadata",
        },
        "track_metadata": {"recording_id": "rec-short", "recording_title": "Short Track", "artist": "Artist"},
    }

    res = analyze_audio(dummy_wav, filename="test.wav", usage_context={"platform": "youtube", "commercial_use": False, "monetization": False})
    assert mock_cascade.call_count == 1
    assert mock_cascade.call_args[1]["audio_path"] == dummy_wav
    assert res["evidence"]["windows_scanned"] == 1


@patch("backend.services.analysis_pipeline.get_audio_duration", return_value=120.0)
@patch("backend.services.analysis_pipeline.slice_audio")
@patch("backend.services.analysis_pipeline.process_music_query")
def test_multiwindow_all_unknown_picks_best_score(mock_cascade, mock_slice, mock_dur, tmp_path):
    dummy_wav = str(tmp_path / "test.wav")
    open(dummy_wav, "w").close()
    mock_slice.return_value = dummy_wav

    mock_cascade.side_effect = [
        {"match_type": "UNKNOWN", "score": 0.12, "recording_id": None},
        {"match_type": "UNKNOWN", "score": 0.35, "recording_id": None},
        {"match_type": "UNKNOWN", "score": 0.08, "recording_id": None},
        {"match_type": "UNKNOWN", "score": 0.22, "recording_id": None},
        {"match_type": "UNKNOWN", "score": 0.15, "recording_id": None},
    ]

    res = analyze_audio(dummy_wav, filename="test.wav", usage_context={"platform": "youtube", "commercial_use": False, "monetization": False})
    assert res["match"]["type"] == "UNKNOWN"
    assert res["match"]["confidence"] == 0.35
