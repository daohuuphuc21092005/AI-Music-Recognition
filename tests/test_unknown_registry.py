"""
Sổ bài chưa nhận diện (tab "Thống kê & Tra cứu").

Ba điều phải luôn đúng, và bộ test này khoá chúng lại:
  1. Sổ KHÔNG chạm vào quyết định: ghi sổ thành công hay thất bại thì `assessment`
     vẫn y nguyên (§2.5, §2.6 — ghi chú không phải dữ liệu quyền).
  2. Sổ lỗi không làm hỏng job: job vẫn DONE, evidence chỉ mang mã REGISTRY_FAILED.
  3. Gộp lượt gặp dùng đúng ngưỡng hiệu dụng của tầng 1 (`min_score_for_duration`),
     không một ngưỡng tự đặt.
"""
import copy
import os
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from backend import config
from backend.api import routes
from backend.main import app
from backend.schemas.analysis import PipelineStage
from backend.services import unknown_registry_service as registry
from tests.conftest import requires_db, requires_fpcalc


# --------------------------------------------------------------------------
# Dữ liệu mẫu
# --------------------------------------------------------------------------
def make_result(match_type="UNKNOWN", risk="UNKNOWN", recording_id=None, candidates=None):
    return {
        "identity": {"track": None, "artist": None, "recording_id": recording_id,
                     "composition_id": None},
        "match": {"type": match_type, "confidence": 0.61, "pipeline_stage": "STAGE_3_COVER"},
        "rights": {"license": None, "rights_found": False, "predicted": False},
        "assessment": {
            "risk": risk,
            "category": "UNCATEGORIZED" if risk == "UNKNOWN" else "USER_GENERATED_CONTENT",
            "condition": "HUMAN_REVIEW_REQUIRED",
            "reason": "Không đủ bằng chứng để định danh.",
            "identity_confidence": 0.61,
            "rights_confidence": 0.4,
            "decision_confidence": 0.4,
            "worst_case": None,
        },
        "recommendation": "⚪ CHƯA XÁC ĐỊNH. Cần người kiểm tra thủ công trước khi phát hành.",
        "evidence": {"candidates": candidates or []},
        "latency_ms": 1234.5,
        "model_version": "MERT-v1-95M",
    }


def random_fp(n, seed):
    return np.random.RandomState(seed).randint(0, 2 ** 32, size=n, dtype=np.uint64).astype(np.uint32)


# --------------------------------------------------------------------------
# classify_unknown
# --------------------------------------------------------------------------
@pytest.mark.parametrize("match_type, risk, recording_id, expected", [
    ("UNKNOWN", "UNKNOWN", None, registry.NOT_IDENTIFIED),        # quyền suy đoán bị rights_gate chặn
    ("UNKNOWN", "LOW", None, registry.NOT_IDENTIFIED),            # USER_GENERATED_CONTENT
    ("EXACT_MATCH", "UNKNOWN", str(uuid.uuid4()), registry.RIGHTS_UNKNOWN),
    ("NEAR_MATCH", "UNKNOWN", str(uuid.uuid4()), registry.RIGHTS_UNKNOWN),
    ("EXACT_MATCH", "LOW", str(uuid.uuid4()), None),              # đã xác định
    ("COVER_MATCH", "HIGH", str(uuid.uuid4()), None),
    ("UNAVAILABLE", "UNKNOWN", None, None),                       # lỗi hạ tầng, không phải nhạc lạ
])
def test_classify_unknown(match_type, risk, recording_id, expected):
    result = make_result(match_type, risk, recording_id)
    assert registry.classify_unknown(result) == expected


# --------------------------------------------------------------------------
# find_best_entry — gộp lượt gặp bằng fingerprint
# --------------------------------------------------------------------------
def test_fingerprint_giong_het_thi_gop_vao_muc_cu():
    fp = random_fp(900, seed=1)
    found = registry.find_best_entry(fp, 112.0, [("a", fp, 112.0)])
    assert found == ("a", pytest.approx(1.0))


def test_doan_cat_tu_giua_bai_van_gop_duoc():
    """So khớp dò toàn bộ độ lệch nên đoạn cắt giữa bài vẫn khớp mục đã có."""
    full = random_fp(900, seed=2)
    found = registry.find_best_entry(full[200:500], 37.0, [("a", full, 112.0)])
    assert found is not None and found[0] == "a"


def test_bai_khac_thi_tao_muc_moi():
    found = registry.find_best_entry(random_fp(600, seed=3), 75.0,
                                     [("a", random_fp(600, seed=4), 75.0)])
    assert found is None


def test_chon_muc_khop_nhat_va_bo_qua_muc_rong():
    target = random_fp(600, seed=5)
    partial = target.copy()
    partial[300:] = random_fp(300, seed=6)       # khớp một nửa
    entries = [("empty", np.empty(0, dtype=np.uint32), 30.0),
               ("half", partial, 75.0),
               ("full", target, 75.0)]
    assert registry.find_best_entry(target, 75.0, entries)[0] == "full"


def test_nguong_nang_theo_doan_ngan_hon_trong_cap():
    """
    Cùng một điểm khớp 0.35: đủ với đoạn 30 s (sàn τFP 0.30) nhưng KHÔNG đủ khi một
    trong hai đoạn chỉ 5 s (điểm nền nhiễu 5 s tới 0.37 — xem NOISE_FLOOR_BY_DURATION).
    """
    entry = random_fp(300, seed=7)
    query = random_fp(300, seed=8)
    query[:105] = entry[:105]                     # 105 / 300 = 0.35
    assert registry.find_best_entry(query, 30.0, [("a", entry, 30.0)]) is not None
    assert registry.find_best_entry(query, 30.0, [("a", entry, 5.0)]) is None
    assert registry.find_best_entry(query, 5.0, [("a", entry, 30.0)]) is None


# --------------------------------------------------------------------------
# Tiện ích
# --------------------------------------------------------------------------
def test_escape_like_khong_de_ky_tu_dai_dien_lot_qua():
    assert registry.escape_like("100%_a\\b") == "100\\%\\_a\\\\b"


def test_clean_text_bo_ky_tu_dieu_khien_va_chuoi_rong_thanh_none():
    assert registry.clean_text("Bài\r\nhát\x00") == "Bàihát"
    assert registry.clean_text("dòng 1\ndòng 2\x07", keep_newlines=True) == "dòng 1\ndòng 2"
    assert registry.clean_text("   ") is None
    assert registry.clean_text(None) is None


def _row(**overrides):
    base = dict(
        unknown_id=uuid.uuid4(), kind="NOT_IDENTIFIED", recording_id=None,
        first_filename="ban_thu_noi_bo.mp3", first_seen_at=None, last_seen_at=None,
        sighting_count=3, last_match_type="UNKNOWN", last_risk_level="UNKNOWN",
        last_category="UNCATEGORIZED", review_status="PENDING", reviewer_title=None,
        reviewer_artist=None, catalogue_title=None, catalogue_artist=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_nhan_hien_thi_uu_tien_ghi_chu_roi_catalogue_roi_ten_file():
    assert registry._label(_row()) == "ban_thu_noi_bo.mp3"
    assert registry._label(_row(catalogue_title="Bài A", catalogue_artist="Ca sĩ A")) == "Bài A — Ca sĩ A"
    # Tên do người thẩm định đặt không ghép với nghệ sĩ của catalogue
    assert registry._label(_row(reviewer_title="Bài B", catalogue_artist="Ca sĩ A")) == "Bài B"
    assert registry._label(_row(reviewer_artist="Ca sĩ C")) == "Chưa rõ tên bài — Ca sĩ C"


def test_che_do_public_an_ten_file(monkeypatch):
    monkeypatch.setattr(config, "EVIDENCE_DETAIL", "public")
    item = registry._summary(_row())
    assert "first_filename" not in item
    assert item["label"] == "Chưa đặt tên"


def test_buoc_ghi_so_khai_bao_trong_enum_pipeline_stage():
    """GET /jobs đọc stage qua PipelineStage — thiếu giá trị này thì trả 500."""
    assert "REGISTRY_LOOKUP" in {stage.value for stage in PipelineStage}


# --------------------------------------------------------------------------
# Gắn vào run_job — sổ không chạm vào quyết định, sổ lỗi không làm hỏng job
# --------------------------------------------------------------------------
class FakeDb:
    def __init__(self):
        self.rollbacks = 0
        self.closed = False

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


@pytest.fixture
def stages(monkeypatch):
    seen = []
    monkeypatch.setattr(routes.job_service, "mark_stage",
                        lambda db, job_id, stage: seen.append(stage))
    monkeypatch.setattr(config, "UNKNOWN_REGISTRY_ENABLED", True)
    return seen


def test_so_loi_thi_danh_gia_giu_nguyen_va_ghi_ma_loi(monkeypatch, stages):
    def boom(*args, **kwargs):
        raise RuntimeError(r"bi mat noi bo C:\Users\ai-do\secret")
    monkeypatch.setattr(routes.unknown_registry_service, "record_unknown", boom)

    result = make_result()
    before = copy.deepcopy(result["assessment"])
    db = FakeDb()
    routes.attach_unknown_registry(db, "job-1", "a.mp3", "a.mp3", {"platform": "YOUTUBE"}, result)

    assert result["assessment"] == before
    assert result["evidence"]["unknown_registry"] == {
        "error_code": "REGISTRY_FAILED", "kind": registry.NOT_IDENTIFIED}
    assert "secret" not in str(result)
    assert db.rollbacks == 1
    assert stages == ["REGISTRY_LOOKUP"]


def test_ghi_so_thanh_cong_cung_khong_doi_danh_gia(monkeypatch, stages):
    calls = []

    def fake_record(db, **kwargs):
        calls.append(kwargs)
        # Kể cả khi mục đã được thẩm định đặt tên, quyết định vẫn không đổi
        return {"unknown_id": "u-1", "kind": kwargs["kind"], "is_new": False,
                "sighting_count": 7, "reviewer_title": "Bài đã biết"}

    monkeypatch.setattr(routes.unknown_registry_service, "record_unknown", fake_record)
    result = make_result(match_type="EXACT_MATCH", risk="UNKNOWN", recording_id=str(uuid.uuid4()))
    before_assessment = copy.deepcopy(result["assessment"])
    before_identity = copy.deepcopy(result["identity"])
    before_recommendation = result["recommendation"]

    routes.attach_unknown_registry(FakeDb(), "job-2", "a.mp3", "a.mp3", {}, result)

    assert result["assessment"] == before_assessment
    assert result["identity"] == before_identity
    assert result["recommendation"] == before_recommendation
    assert result["evidence"]["unknown_registry"]["sighting_count"] == 7
    assert calls[0]["kind"] == registry.RIGHTS_UNKNOWN


def test_ket_qua_da_xac_dinh_thi_khong_ghi_so(monkeypatch, stages):
    monkeypatch.setattr(routes.unknown_registry_service, "record_unknown",
                        lambda *a, **k: pytest.fail("không được ghi sổ"))
    result = make_result(match_type="EXACT_MATCH", risk="LOW", recording_id=str(uuid.uuid4()))
    routes.attach_unknown_registry(FakeDb(), "job-3", "a.mp3", "a.mp3", {}, result)
    assert "unknown_registry" not in result["evidence"]
    assert stages == []


def test_tat_so_thi_khong_ghi(monkeypatch, stages):
    monkeypatch.setattr(config, "UNKNOWN_REGISTRY_ENABLED", False)
    monkeypatch.setattr(routes.unknown_registry_service, "record_unknown",
                        lambda *a, **k: pytest.fail("không được ghi sổ"))
    result = make_result()
    routes.attach_unknown_registry(FakeDb(), "job-4", "a.mp3", "a.mp3", {}, result)
    assert "unknown_registry" not in result["evidence"]


def test_run_job_van_done_khi_so_loi(monkeypatch):
    """Luồng thật của /analyze: sổ ném lỗi -> job vẫn DONE với kết quả nguyên vẹn."""
    db = FakeDb()
    done = {}
    monkeypatch.setattr(config, "UNKNOWN_REGISTRY_ENABLED", True)
    monkeypatch.setattr(routes, "SessionLocal", lambda: db)
    monkeypatch.setattr(routes, "analyze_audio", lambda *a, **k: make_result())
    monkeypatch.setattr(routes.job_service, "mark_processing", lambda *a: None)
    monkeypatch.setattr(routes.job_service, "mark_stage", lambda *a: None)
    monkeypatch.setattr(routes.job_service, "mark_done",
                        lambda _db, job_id, result: done.update(job_id=job_id, result=result))
    monkeypatch.setattr(routes.job_service, "log_analysis_result", lambda *a, **k: None)
    monkeypatch.setattr(routes.job_service, "mark_failed",
                        lambda *a: pytest.fail("job không được FAILED vì sổ lỗi"))
    monkeypatch.setattr(routes.unknown_registry_service, "record_unknown",
                        lambda *a, **k: (_ for _ in ()).throw(OperationalError("x", {}, Exception())))
    monkeypatch.setattr(routes.job_concurrency_limiter, "release", lambda: None)

    routes.run_job("job-5", "khong_ton_tai.mp3", "a.mp3", {"platform": "YOUTUBE"}, None)

    assert done["job_id"] == "job-5"
    assert done["result"]["assessment"] == make_result()["assessment"]
    assert done["result"]["evidence"]["unknown_registry"]["error_code"] == "REGISTRY_FAILED"
    assert db.closed


# --------------------------------------------------------------------------
# API — không cần CSDL: thay hàm service ngay trong routes
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_stats_khong_bi_route_id_bat_nham(client, monkeypatch):
    seen = {}

    def fake_stats(db, utc_offset_min=0):
        seen["offset"] = utc_offset_min
        return {"totals": {"entries": 2}, "registry_enabled": True}

    monkeypatch.setattr(routes.unknown_registry_service, "get_stats", fake_stats)
    response = client.get("/api/v1/unknown-tracks/stats", params={"utc_offset_min": 420})
    assert response.status_code == 200
    assert response.json()["totals"]["entries"] == 2
    assert seen["offset"] == 420
    assert client.get("/api/v1/unknown-tracks/stats",
                      params={"utc_offset_min": 5000}).status_code == 422


def test_tra_cuu_chuyen_dung_tham_so(client, monkeypatch):
    seen = {}

    def fake_list(db, **kwargs):
        seen.update(kwargs)
        return {"total": 0, "limit": kwargs["limit"], "offset": kwargs["offset"], "items": []}

    monkeypatch.setattr(routes.unknown_registry_service, "list_entries", fake_list)
    response = client.get("/api/v1/unknown-tracks", params={
        "q": "lạc trôi", "kind": "NOT_IDENTIFIED", "status": "PENDING",
        "sort": "sightings", "limit": 5, "offset": 10})
    assert response.status_code == 200
    assert seen == {"q": "lạc trôi", "kind": "NOT_IDENTIFIED", "status": "PENDING",
                    "sort": "sightings", "limit": 5, "offset": 10}


@pytest.mark.parametrize("params", [
    {"limit": 0}, {"limit": 101}, {"offset": -1}, {"status": "abc"},
    {"kind": "not_identified"}, {"sort": "random"}, {"q": "x" * 101},
])
def test_tham_so_sai_tra_invalid_request(client, params):
    response = client.get("/api/v1/unknown-tracks", params=params)
    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "INVALID_REQUEST"


def test_id_sai_dang_va_khong_ton_tai_tra_404(client, monkeypatch):
    response = client.get("/api/v1/unknown-tracks/khong-phai-uuid")
    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "UNKNOWN_TRACK"

    monkeypatch.setattr(routes.unknown_registry_service, "get_entry", lambda db, uid: None)
    response = client.get(f"/api/v1/unknown-tracks/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "UNKNOWN_TRACK"


def test_patch_chi_gui_truong_co_mat(client, monkeypatch):
    seen = {}

    def fake_update(db, uid, fields):
        seen.update(uid=uid, fields=fields)
        return {"unknown_id": uid, "reviewer_title": fields.get("title")}

    monkeypatch.setattr(routes.unknown_registry_service, "update_review", fake_update)
    uid = str(uuid.uuid4())
    response = client.patch(f"/api/v1/unknown-tracks/{uid}",
                            json={"title": "Bài X", "review_status": "REVIEWED"})
    assert response.status_code == 200
    assert seen["uid"] == uid
    assert set(seen["fields"]) == {"title", "review_status"}
    assert getattr(seen["fields"]["review_status"], "value", None) == "REVIEWED"


@pytest.mark.parametrize("body", [
    {"title": "x" * 256}, {"note": "x" * 2001}, {"review_status": "DONE"},
])
def test_patch_du_lieu_sai_tra_invalid_request(client, body):
    response = client.patch(f"/api/v1/unknown-tracks/{uuid.uuid4()}", json=body)
    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "INVALID_REQUEST"


def test_loi_csdl_tra_database_failure_khong_lo_chi_tiet(client, monkeypatch):
    def boom(db, **kwargs):
        raise OperationalError("SELECT bi mat", {}, Exception("password=hunter2"))

    monkeypatch.setattr(routes.unknown_registry_service, "list_entries", boom)
    response = client.get("/api/v1/unknown-tracks")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error_code"] == "DATABASE_FAILURE"
    assert "hunter2" not in str(detail) and "SELECT" not in str(detail)


def test_api_key_bao_ve_so(client, monkeypatch):
    monkeypatch.setattr(routes.unknown_registry_service, "get_stats", lambda db, **k: {"totals": {}})
    monkeypatch.setattr(config, "API_KEY", "khoa_bi_mat_123")
    assert client.get("/api/v1/unknown-tracks/stats").status_code == 401
    assert client.patch(f"/api/v1/unknown-tracks/{uuid.uuid4()}", json={}).status_code == 401
    ok = client.get("/api/v1/unknown-tracks/stats", headers={"X-API-Key": "khoa_bi_mat_123"})
    assert ok.status_code == 200


# --------------------------------------------------------------------------
# Tích hợp với PostgreSQL + fpcalc thật
# --------------------------------------------------------------------------
@pytest.fixture
def registry_db(db_session):
    exists = db_session.execute(text("SELECT to_regclass('unknown_tracks')")).scalar()
    if not exists:
        pytest.skip("Chưa có bảng unknown_tracks. Chạy: python init_db.py --schema-only")
    created = []
    yield db_session, created
    db_session.rollback()
    for unknown_id in created:
        db_session.execute(text("DELETE FROM unknown_tracks WHERE unknown_id = :u"), {"u": unknown_id})
    db_session.commit()


@pytest.fixture
def fresh_noise(tmp_path):
    """Nhiễu với seed ngẫu nhiên: không thể trùng một mục người dùng đã gửi thật."""
    import soundfile as sf
    sr = 22050
    signal = np.random.default_rng().normal(0, 0.2, sr * 20).astype("float32")
    path = tmp_path / "noise.wav"
    sf.write(str(path), signal, sr)
    return str(path)


@requires_db
@requires_fpcalc
def test_gui_lai_cung_bai_thi_gop_vao_mot_muc(registry_db, fresh_noise):
    db, created = registry_db
    fingerprint = registry.fingerprint_for_registry(fresh_noise)
    assert fingerprint is not None

    first = registry.register_sighting(
        db, job_id=None, result=make_result(), usage_context={"platform": "YOUTUBE"},
        filename="noise.wav", fingerprint=fingerprint)
    created.append(first["unknown_id"])
    second = registry.register_sighting(
        db, job_id=None, result=make_result(risk="LOW"), usage_context={"platform": "TIKTOK"},
        filename="noise_lan2.wav", fingerprint=fingerprint)

    assert first["is_new"] is True and first["sighting_count"] == 1
    assert second["unknown_id"] == first["unknown_id"]
    assert second["is_new"] is False and second["sighting_count"] == 2
    assert second["match_score"] == pytest.approx(1.0)

    tag = f"Bài thử nghiệm {uuid.uuid4().hex[:8]}"
    entry = registry.update_review(db, first["unknown_id"],
                                   {"title": tag, "note": "dòng 1\ndòng 2", "review_status": "REVIEWED"})
    assert entry["reviewer_title"] == tag
    assert entry["review_status"] == "REVIEWED" and entry["reviewed_at"]
    assert entry["first_filename"] == "noise.wav"          # tên file của lượt ĐẦU
    assert [s["platform"] for s in entry["sightings"]] == ["TIKTOK", "YOUTUBE"]
    assert "job_id" not in str(entry)

    found = registry.list_entries(db, q=tag)
    assert [item["unknown_id"] for item in found["items"]] == [first["unknown_id"]]
    assert registry.list_entries(db, q="%")["total"] >= 0     # ký tự đại diện không lỗi SQL

    stats = registry.get_stats(db, utc_offset_min=420)
    assert stats["totals"]["entries"] >= 1
    assert stats["totals"]["resighted_entries"] >= 1
    days = stats["sightings_by_day"]
    assert len(days) == registry.DAYS_IN_STATS
    assert days == sorted(days, key=lambda d: d["date"])
    assert days[-1]["count"] >= 2                      # hai lượt gặp vừa ghi rơi vào hôm nay


@requires_db
def test_rights_unknown_gop_theo_recording_id(registry_db):
    db, created = registry_db
    recording_id = str(uuid.uuid4())
    result = make_result(match_type="EXACT_MATCH", risk="UNKNOWN", recording_id=recording_id)
    first = registry.register_sighting(db, job_id=None, result=result, usage_context={},
                                       filename="a.mp3")
    created.append(first["unknown_id"])
    second = registry.register_sighting(db, job_id=None, result=result, usage_context={},
                                        filename="b.mp3")
    assert first["kind"] == registry.RIGHTS_UNKNOWN
    assert second["unknown_id"] == first["unknown_id"] and second["sighting_count"] == 2
