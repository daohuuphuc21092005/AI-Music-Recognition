"""Kiểm thử API ở mức HTTP (đặc tả §11)."""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from tests.conftest import requires_db


@pytest.fixture(scope="module")
def client():
    # Dùng context manager để lifespan (nạp FAISS + rules) thực sự chạy
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] in ("ONLINE", "DEGRADED")
    components = body["components"]
    assert components["faiss"]["status"] in ("READY", "UNAVAILABLE")
    assert components["chromaprint"]["status"] in ("AVAILABLE", "MISSING")
    assert components["rule_engine"]["status"] in ("READY", "UNAVAILABLE")
    assert "thresholds" in body


def test_dinh_dang_khong_ho_tro_tra_ma_loi_chuan(client):
    """Lỗi phải là mã chuẩn hoá, tuyệt đối không lộ traceback (§12)."""
    response = client.post(
        "/api/v1/search",
        files={"file": ("ghi_chu.txt", b"day khong phai audio", "text/plain")},
    )

    assert response.status_code == 415
    detail = response.json()["detail"]
    assert detail["error_code"] == "UNSUPPORTED_FORMAT"
    assert "Traceback" not in str(detail)


@requires_db
def test_search_dong_bo_tra_du_4_phan(client, test_audio):
    """Response phải có đủ Identification / Rights / Assessment / Evidence."""
    with open(test_audio, "rb") as f:
        response = client.post(
            "/api/v1/search",
            files={"file": ("test.mp3", f, "audio/mpeg")},
            data={"platform": "YOUTUBE", "commercial_use": "false",
                  "monetization": "false"},
        )

    assert response.status_code == 200, response.json()
    body = response.json()

    for key in ("identity", "match", "rights", "assessment", "recommendation",
                "evidence"):
        assert key in body, f"thiếu khối {key}"

    assert body["match"]["type"] in ("EXACT_MATCH", "NEAR_MATCH", "COVER_MATCH", "UNKNOWN")
    assert body["assessment"]["risk"] in ("LOW", "CONDITIONAL", "HIGH", "UNKNOWN")
    # Ba loại độ tin cậy tách biệt (§2)
    for key in ("identity_confidence", "rights_confidence", "decision_confidence"):
        assert key in body["assessment"]
    # Không hộp đen: phải có lý do + rule đã kích hoạt
    assert body["assessment"]["reason"]
    assert body["evidence"]["rule_engine"]["rule_id"]


@requires_db
def test_luong_job_bat_dong_bo(client, test_audio):
    """POST /analyze -> GET /jobs -> GET /results."""
    with open(test_audio, "rb") as f:
        accepted = client.post(
            "/api/v1/analyze",
            files={"file": ("test.mp3", f, "audio/mpeg")},
            data={"platform": "YOUTUBE", "commercial_use": "true",
                  "monetization": "true"},
        )

    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    assert accepted.json()["status"] == "QUEUED"

    status = client.get(f"/api/v1/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] in ("QUEUED", "PROCESSING", "DONE", "FAILED")

    result = client.get(f"/api/v1/results/{job_id}")
    assert result.status_code == 200
    body = result.json()
    if body["status"] == "DONE":
        assert body["assessment"]["risk"] in ("LOW", "CONDITIONAL", "HIGH", "UNKNOWN")
        assert body["evidence"]["rule_engine"]["rules_version"] == "rules_v1"


@requires_db
def test_job_khong_ton_tai(client):
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/api/v1/jobs/{missing}").status_code == 404
    assert client.get(f"/api/v1/results/{missing}").status_code == 404


@requires_db
def test_tracks_endpoint_id_khong_hop_le(client):
    response = client.get("/api/v1/tracks/khong-phai-uuid")
    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "UNKNOWN_TRACK"


@requires_db
def test_feedback_endpoint(client):
    response = client.post("/api/v1/feedback",
                           json={"verdict": "Unsure", "note": "khong chac"})
    assert response.status_code == 200
    assert response.json()["feedback_id"]


def test_feedback_tu_choi_verdict_la(client):
    response = client.post("/api/v1/feedback", json={"verdict": "Maybe"})
    assert response.status_code == 422  # pydantic chặn từ đầu
    detail = response.json()["detail"]
    assert detail["error_code"] == "INVALID_REQUEST"
    assert detail["details"][0]["field"] == "verdict"


def test_tham_so_sai_tra_khuon_ma_loi_chuan_khong_lap_lai_input(client):
    """
    Trước đây 422 trả danh sách mặc định của FastAPI: frontend không đọc được mã lỗi
    và trường `input` lặp lại nguyên giá trị người dùng gửi.
    """
    response = client.post(
        "/api/v1/search",
        files={"file": ("q.wav", b"RIFF0000WAVE", "audio/wav")},
        data={"platform": "gia-tri-la-XYZ", "commercial_use": "false", "monetization": "false"},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error_code"] == "INVALID_REQUEST"
    assert "platform" in detail["message"] and "YOUTUBE" in detail["message"]
    assert "gia-tri-la-XYZ" not in response.text
    assert "Traceback" not in response.text


def test_health_khong_lo_duong_dan_may_chu(client):
    """/health là endpoint công khai: không được chứa đường dẫn tuyệt đối hay nội dung ngoại lệ."""
    body = client.get("/health").json()
    text = str(body)
    assert "fpcalc_path" not in text
    assert ":\\\\" not in text and ":/" not in text.replace("://", "")
    for component in ("faiss", "rule_engine"):
        assert body["components"][component].get("error") in (
            None, "INDEX_LOAD_FAILED", "RULES_LOAD_FAILED")


@requires_db
@pytest.mark.parametrize("path", ["/api/v1/jobs/abc", "/api/v1/results/abc"])
def test_job_id_sai_dang_tra_ma_loi_chuan(client, path):
    """Trước đây chuỗi không phải UUID chạm thẳng PostgreSQL và ra trang 500 trơn."""
    response = client.get(path)
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["error_code"] == "UNKNOWN_TRACK"
    assert "Traceback" not in str(detail)


def test_loi_khong_luong_truoc_van_dung_khuon_ma_loi(monkeypatch):
    """Lưới an toàn toàn cục: ngoại lệ bất kỳ -> 500 kèm {error_code, message}, không traceback."""
    from backend.api import routes

    def no_tung(*args, **kwargs):
        raise RuntimeError(r"bi mat noi bo C:\Users\ai-do\secret")

    monkeypatch.setattr(routes, "get_full_music_rights", no_tung)
    with TestClient(app, raise_server_exceptions=False) as local_client:
        response = local_client.get("/api/v1/tracks/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["error_code"] == "INTERNAL_ERROR"
    assert "bi mat" not in str(detail)
