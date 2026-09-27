import pytest
from unittest.mock import patch, MagicMock

from backend.services.analysis_pipeline import analyze_audio


@patch("backend.services.analysis_pipeline.get_audio_duration", return_value=15.0)
@patch("backend.services.analysis_pipeline.process_music_query")
@patch("backend.services.analysis_pipeline.get_full_music_rights")
def test_worst_case_cc_by_nc_non_commercial(mock_rights, mock_cascade, mock_dur, tmp_path):
    """
    CC_BY_NC declared non-commercial -> CONDITIONAL (ATTRIBUTION_REQUIRED)
    worst_case should evaluate under commercial_use=True, monetization=True -> HIGH (NON_COMMERCIAL_VIOLATION)
    """
    dummy_wav = str(tmp_path / "test.wav")
    open(dummy_wav, "w").close()

    mock_cascade.return_value = {
        "match_type": "EXACT_MATCH",
        "score": 0.05,
        "identity_confidence": 0.99,
        "recording_id": "rec-nc",
    }
    mock_rights.return_value = {
        "recording_id": "rec-nc",
        "rights_and_licensing": {
            "license_type": "CC_BY_NC",
            "copyright_status": "COPYRIGHTED",
            "attribution_required": True,
            "commercial_use_allowed": False,
            "monetization_allowed": False,
            "source": "jamendo_api",
        },
        "track_metadata": {"recording_id": "rec-nc", "recording_title": "NC Track", "artist": "NC Artist"},
    }

    res = analyze_audio(
        dummy_wav,
        filename="test.wav",
        usage_context={"platform": "YOUTUBE", "commercial_use": False, "monetization": False},
    )

    assessment = res["assessment"]
    assert assessment["risk"] == "CONDITIONAL"
    assert assessment["context_declared_by_user"] is True
    assert assessment["worst_case"] is not None
    assert assessment["worst_case"]["risk"] == "HIGH"
    assert assessment["worst_case"]["condition"] == "NON_COMMERCIAL_VIOLATION"
    assert assessment["rights_source_note"] == "Dữ liệu giấy phép từ Jamendo Licensing API"


@patch("backend.services.analysis_pipeline.get_audio_duration", return_value=15.0)
@patch("backend.services.analysis_pipeline.process_music_query")
def test_worst_case_unknown_match(mock_cascade, mock_dur, tmp_path):
    """
    When match is UNKNOWN, worst_case should be None (or match initial UNKNOWN) without leaking HIGH risk.
    """
    dummy_wav = str(tmp_path / "test.wav")
    open(dummy_wav, "w").close()

    mock_cascade.return_value = {
        "match_type": "UNKNOWN",
        "score": 0.2,
        "identity_confidence": 0.2,
        "recording_id": None,
    }

    res = analyze_audio(
        dummy_wav,
        filename="test.wav",
        usage_context={"platform": "YOUTUBE", "commercial_use": False, "monetization": False},
    )

    assessment = res["assessment"]
    assert assessment["risk"] == "UNKNOWN"
    # Since worst-case evaluation for UNKNOWN also yields UNKNOWN, worst_case is None (or same UNKNOWN risk)
    assert assessment["worst_case"] is None or assessment["worst_case"]["risk"] == "UNKNOWN"


@patch("backend.services.analysis_pipeline.get_audio_duration", return_value=15.0)
@patch("backend.services.analysis_pipeline.process_music_query")
@patch("backend.services.analysis_pipeline.get_full_music_rights")
def test_worst_case_already_worst_case_declared(mock_rights, mock_cascade, mock_dur, tmp_path):
    """
    When user already declared commercial_use=True and monetization=True, worst_case must be None.
    """
    dummy_wav = str(tmp_path / "test.wav")
    open(dummy_wav, "w").close()

    mock_cascade.return_value = {
        "match_type": "EXACT_MATCH",
        "score": 0.05,
        "identity_confidence": 0.99,
        "recording_id": "rec-nc2",
    }
    mock_rights.return_value = {
        "recording_id": "rec-nc2",
        "rights_and_licensing": {
            "license_type": "CC_BY_NC",
            "copyright_status": "COPYRIGHTED",
            "attribution_required": True,
            "commercial_use_allowed": False,
            "monetization_allowed": False,
            "source": "fma_metadata",
        },
        "track_metadata": {"recording_id": "rec-nc2", "recording_title": "NC Track 2", "artist": "Artist"},
    }

    res = analyze_audio(
        dummy_wav,
        filename="test.wav",
        usage_context={"platform": "YOUTUBE", "commercial_use": True, "monetization": True},
    )

    assessment = res["assessment"]
    assert assessment["risk"] == "HIGH"
    assert assessment["worst_case"] is None
    assert assessment["rights_source_note"] == "Dữ liệu khai báo bởi nghệ sĩ trên Free Music Archive (không được xác minh độc lập)"


@patch("backend.services.analysis_pipeline.get_audio_duration", return_value=15.0)
@patch("backend.services.analysis_pipeline.process_music_query")
@patch("backend.services.analysis_pipeline.get_full_music_rights")
def test_rights_source_notes_mapping(mock_rights, mock_cascade, mock_dur, tmp_path):
    """
    Verify all 4 source mappings: fma_metadata, jamendo_api, simulated, other.
    """
    dummy_wav = str(tmp_path / "test.wav")
    open(dummy_wav, "w").close()

    mock_cascade.return_value = {
        "match_type": "EXACT_MATCH",
        "score": 0.05,
        "identity_confidence": 0.99,
        "recording_id": "rec-test",
    }

    sources_expected = [
        ("fma_metadata", "Dữ liệu khai báo bởi nghệ sĩ trên Free Music Archive (không được xác minh độc lập)"),
        ("jamendo_api", "Dữ liệu giấy phép từ Jamendo Licensing API"),
        ("simulated", "Dữ liệu thử nghiệm nội bộ (giả lập)"),
        ("custom_db", "Dữ liệu từ nguồn: custom_db"),
    ]

    for src, expected_note in sources_expected:
        mock_rights.return_value = {
            "recording_id": "rec-test",
            "rights_and_licensing": {
                "license_type": "CC0",
                "copyright_status": "PUBLIC_DOMAIN",
                "attribution_required": False,
                "commercial_use_allowed": True,
                "monetization_allowed": True,
                "source": src,
            },
            "track_metadata": {"recording_id": "rec-test", "recording_title": "Test", "artist": "Test"},
        }
        res = analyze_audio(
            dummy_wav,
            filename="test.wav",
            usage_context={"platform": "YOUTUBE", "commercial_use": False, "monetization": False},
        )
        assert res["assessment"]["rights_source_note"] == expected_note
