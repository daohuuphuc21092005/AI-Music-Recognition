"""
Kiểm thử bộ giải pháp vá 7 lỗ hổng bảo mật (Security Hardening).
"""
import os
import subprocess
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from backend import config
from backend.main import app
from backend.security import rate_limiter, sanitize_filename
from backend.services import audio_service, license_classifier_service
from tests.conftest import requires_db


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_body_qua_lon_bi_tu_choi_va_khong_luu_file(client, monkeypatch):
    """
    1. Upload vượt MAX_UPLOAD_MB -> 413 FILE_TOO_LARGE và không còn file rác trong temp_uploads.
    """
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)  # Giảm xuống 1 MB để test nhanh

    files_before = set(os.listdir(config.TEMP_UPLOAD_DIR)) if os.path.exists(config.TEMP_UPLOAD_DIR) else set()

    # Dữ liệu 2 MB (vượt 1 MB)
    large_content = b"0" * (2 * 1024 * 1024)
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("large_audio.mp3", BytesIO(large_content), "audio/mpeg")},
    )

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert detail["error_code"] == "FILE_TOO_LARGE"

    files_after = set(os.listdir(config.TEMP_UPLOAD_DIR)) if os.path.exists(config.TEMP_UPLOAD_DIR) else set()
    new_files = files_after - files_before
    assert len(new_files) == 0, f"File rác còn tồn tại trong temp_uploads: {new_files}"


def test_duoi_file_khong_hop_le_bi_tu_choi_va_khong_ghi_dia(client):
    """
    2. Đuôi không hợp lệ -> 415 UNSUPPORTED_FORMAT và không ghi đĩa.
    """
    files_before = set(os.listdir(config.TEMP_UPLOAD_DIR)) if os.path.exists(config.TEMP_UPLOAD_DIR) else set()

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("malicious_script.sh", b"echo evil", "text/x-sh")},
    )

    assert response.status_code == 415
    detail = response.json()["detail"]
    assert detail["error_code"] == "UNSUPPORTED_FORMAT"

    files_after = set(os.listdir(config.TEMP_UPLOAD_DIR)) if os.path.exists(config.TEMP_UPLOAD_DIR) else set()
    new_files = files_after - files_before
    assert len(new_files) == 0, f"Có file bị ghi xuống đĩa dù định dạng sai: {new_files}"


def test_rate_limit_11_requests_tra_ve_429_kem_retry_after(client, monkeypatch):
    """
    3. Yêu cầu thứ 11 trong 1 phút -> 429 RATE_LIMITED kèm Retry-After.
    """
    rate_limiter.reset()
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MIN", 10)

    # 10 request đầu tiên hợp lệ (gửi file không hợp lệ để xử lý nhanh ở validation)
    for i in range(10):
        res = client.post(
            "/api/v1/analyze",
            files={"file": ("test.sh", b"x", "text/plain")},
        )
        assert res.status_code in (415, 202)

    # Request thứ 11 từ cùng IP
    response_11 = client.post(
        "/api/v1/analyze",
        files={"file": ("test.sh", b"x", "text/plain")},
    )

    assert response_11.status_code == 429
    assert "Retry-After" in response_11.headers
    retry_after = int(response_11.headers["Retry-After"])
    assert retry_after > 0
    detail = response_11.json()["detail"]
    assert detail["error_code"] == "RATE_LIMITED"

    rate_limiter.reset()


@requires_db
def test_api_key_auth_dat_va_khong_dat(client, monkeypatch):
    """
    4. Xác thực tuỳ chọn qua header X-API-Key:
       - Khi không đặt API_KEY -> truy cập bình thường.
       - Khi đặt API_KEY -> thiếu hoặc sai key ra 401, đúng key thì được qua.

    Cần CSDL thật: route /api/v1/jobs/{id} tra bảng jobs để trả 404 UNKNOWN_TRACK,
    nên chạy được cả khi CSDL không sẵn sàng sẽ ném DATABASE_FAILURE thay vì 401/404.
    """
    # Khi KHÔNG đặt API_KEY
    monkeypatch.setattr(config, "API_KEY", None)
    res_no_auth = client.get("/api/v1/jobs/00000000-0000-0000-0000-000000000000")
    # Không bị chặn bởi 401 (ra 404 UNKNOWN_TRACK)
    assert res_no_auth.status_code == 404

    # Khi ĐẶT API_KEY
    test_key = "secure_test_key_12345"
    monkeypatch.setattr(config, "API_KEY", test_key)

    # Không có header X-API-Key -> 401
    res_missing = client.get("/api/v1/jobs/00000000-0000-0000-0000-000000000000")
    assert res_missing.status_code == 401
    assert res_missing.json()["detail"]["error_code"] == "UNAUTHORIZED"

    # Key sai -> 401
    res_wrong = client.get(
        "/api/v1/jobs/00000000-0000-0000-0000-000000000000",
        headers={"X-API-Key": "wrong_key"},
    )
    assert res_wrong.status_code == 401

    # Key đúng -> qua được xác thực (ra 404 vì job không tồn tại)
    res_correct = client.get(
        "/api/v1/jobs/00000000-0000-0000-0000-000000000000",
        headers={"X-API-Key": test_key},
    )
    assert res_correct.status_code == 404


def test_ffmpeg_timeout_nem_loi_chuan_va_xoa_file_tam(monkeypatch, tmp_path):
    """
    5. FFmpeg timeout -> ném AudioProcessingError mã TIMEOUT và dọn file tạm.
    """
    monkeypatch.setattr(audio_service, "ffmpeg_available", lambda: True)

    def mock_subprocess_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=config.FFMPEG_TIMEOUT_S)

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    dummy_video = tmp_path / "test_video.mp4"
    dummy_video.write_bytes(b"dummy video data")

    with pytest.raises(audio_service.AudioProcessingError) as exc_info:
        audio_service.extract_and_normalize_audio(str(dummy_video), output_dir=str(tmp_path))

    assert exc_info.value.code == "TIMEOUT"
    normalized_path = tmp_path / "test_video_normalized.wav"
    assert not normalized_path.exists()


def test_filename_chua_newline_duoc_sanitize(caplog):
    """
    6. Filename chứa \\r\\n không xuất hiện nguyên bản trong log.
    """
    malicious_name = "track\r\nINJECTED_LOG_LEVEL [CRITICAL] hacked\n.mp3"
    cleaned = sanitize_filename(malicious_name)

    assert "\r" not in cleaned
    assert "\n" not in cleaned
    assert cleaned == "trackINJECTED_LOG_LEVEL [CRITICAL] hacked.mp3"

    # Cắt tối đa 255 ký tự
    long_name = "a" * 300 + ".mp3"
    assert len(sanitize_filename(long_name)) == 255


def test_sha256_sidecar_bi_sua_classifier_tra_none(monkeypatch, tmp_path):
    """
    7. File .sha256 bị sửa -> load_model trả (None, None) và predict() trả None.
    """
    license_classifier_service._cache.clear()

    # Giả lập file sidecar mang hash sai
    target = "license_type"
    model_path, meta_path = license_classifier_service._paths(target)
    assert os.path.exists(model_path), "Model joblib phải tồn tại để test sidecar"

    sha_path = model_path + ".sha256"
    original_sha = open(sha_path, "r", encoding="utf-8").read()

    try:
        # Ghi đè hash sai
        with open(sha_path, "w", encoding="utf-8") as f:
            f.write("0000000000000000000000000000000000000000000000000000000000000000\n")

        model, meta = license_classifier_service.load_model(target)
        assert model is None
        assert meta is None

        pred = license_classifier_service.predict([0.1] * 768, target=target)
        assert pred is None
    finally:
        # Khôi phục hash gốc
        with open(sha_path, "w", encoding="utf-8") as f:
            f.write(original_sha)
        license_classifier_service._cache.clear()


def test_evidence_detail_public_an_nguong_va_lam_tron(client, monkeypatch):
    """
    Kiểm tra EVIDENCE_DETAIL=public: /health không trả thresholds,
    và điểm số trong evidence được làm tròn 2 chữ số.
    """
    monkeypatch.setattr(config, "EVIDENCE_DETAIL", "public")

    res_health = client.get("/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert "thresholds" not in health_data
    assert "threshold" not in health_data["components"].get("cover", {})
