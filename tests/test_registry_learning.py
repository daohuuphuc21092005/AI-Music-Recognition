"""
Sổ bài chưa nhận diện "học" bài UNKNOWN (2026-09-27): lưu vector MERT từng đoạn +
descriptor Cover, để lần sau nhận ra cả bản đã nén, cắt, đổi tông của bài đó.

Điều phải luôn đúng:
  1. Ngưỡng nhận lại là τMERT / τCover đã hiệu chỉnh — không một ngưỡng tự đặt cho sổ.
  2. Đặc trưng tính đúng cách chỉ mục chính tính (đoạn 15 s bước 10 s; Cover 30 s đầu).
  3. Học hỏng thì lượt gặp vẫn được ghi (bằng fingerprint), không làm hỏng job.
  4. Không đụng tới `assessment` — khoá ở tests/test_unknown_registry.py.
"""
import uuid

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import text

from backend import config
from backend.services import audio_service, cover_service, embedding_service, registry_features
from backend.services import unknown_registry_service as registry
from tests.conftest import requires_db
from tests.test_unknown_registry import make_result


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def random_unit(seed, dim=768):
    return unit(np.random.default_rng(seed).normal(size=dim))


def near(vector, cosine, seed=0):
    """Vector đơn vị có cosine đúng bằng `cosine` với `vector`."""
    noise = random_unit(seed, vector.size)
    noise = unit(noise - (noise @ vector) * vector)
    return unit(cosine * vector + np.sqrt(1 - cosine ** 2) * noise)


def descriptor(seed):
    """Descriptor Cover tổng hợp: cùng khuôn (12 x 64, trừ trung bình từng khung, L2)."""
    frames = np.random.default_rng(seed).random((12, cover_service.DESCRIPTOR_FRAMES))
    return cover_service.normalize_frames(frames).astype(np.float32)


def shift_pitch(desc, semitones):
    return np.roll(desc.reshape(12, -1), semitones, axis=0).reshape(-1)


# --------------------------------------------------------------------------
# Cắt đoạn giống chỉ mục chính
# --------------------------------------------------------------------------
def test_cat_doan_bai_ngan_theo_luoi_cua_chi_muc_chinh():
    assert registry_features.plan_segments(0.5) == []
    assert registry_features.plan_segments(8) == [(0.0, 8.0)]
    # scripts/build_embeddings.py: cửa sổ 15 s, bước 10 s
    assert registry_features.plan_segments(40) == [(0.0, 15.0), (10.0, 25.0), (20.0, 35.0)]


def test_bai_dai_rai_deu_toi_da_so_doan(monkeypatch):
    monkeypatch.setattr(config, "UNKNOWN_REGISTRY_MAX_SEGMENTS", 5)
    segments = registry_features.plan_segments(240)
    assert len(segments) == 5
    assert segments[0] == (0.0, 15.0)
    assert segments[-1] == (225.0, 240.0)              # phủ tới đuôi bài
    assert all(end - start == pytest.approx(15.0) for start, end in segments)


# --------------------------------------------------------------------------
# So khớp: đúng ngưỡng đã hiệu chỉnh
# --------------------------------------------------------------------------
def test_mert_nhan_ban_bien_doi_tren_nguong_tau_mert():
    stored = random_unit(1)
    references = [("bai-a", stored), ("bai-b", random_unit(2))]
    query = np.vstack([random_unit(3), near(stored, 0.995)])   # một đoạn khớp là đủ

    owner, score = registry_features.best_mert_match(query, references)
    assert owner == "bai-a" and score == pytest.approx(0.995, abs=1e-4)


def test_mert_duoi_nguong_thi_khong_nhan():
    stored = random_unit(1)
    below = near(stored, config.MERT_THRESHOLD - 0.01)
    assert registry_features.best_mert_match(np.vstack([below]), [("bai-a", stored)]) is None
    assert registry_features.best_mert_match(None, [("bai-a", stored)]) is None
    assert registry_features.best_mert_match(np.vstack([stored]), []) is None


def test_mert_lay_max_qua_moi_doan_cua_mot_muc():
    a1, a2 = random_unit(10), random_unit(11)
    query = np.vstack([near(a2, 0.99)])
    owner, score = registry_features.best_mert_match(
        query, [("bai-a", a1), ("bai-a", a2), ("bai-b", random_unit(12))])
    assert owner == "bai-a" and score == pytest.approx(0.99, abs=1e-4)


def test_cover_nhan_ban_doi_tong_kem_oti():
    stored = descriptor(1)
    query = np.vstack([shift_pitch(stored, 3)])            # truy vấn cao hơn 3 bán cung
    owner, score, oti, row = registry_features.best_cover_match(
        query, [("bai-a", stored), ("bai-b", descriptor(2))])
    assert owner == "bai-a" and score == pytest.approx(1.0, abs=1e-5)
    assert "cao hơn bản gốc 3 bán cung" in cover_service.describe_oti(oti)
    assert row == 0


def test_cover_bai_khac_duoi_nguong_tau_cover():
    assert registry_features.best_cover_match(
        np.vstack([descriptor(5)]), [("bai-a", descriptor(6))]) is None


def test_chi_hoc_doan_moi_theo_dung_nguong():
    stored = [random_unit(1), random_unit(2)]
    features = {"mert": [(0.0, 15.0, near(stored[0], 0.99)),          # đã nhận ra được -> bỏ
                         (10.0, 25.0, random_unit(3)),                 # mới
                         (20.0, 35.0, near(random_unit(3), 0.995)),    # trùng đoạn vừa nhận -> bỏ
                         (30.0, 45.0, random_unit(4))],                # mới nhưng hết chỗ
                "cover_ref": shift_pitch(descriptor(1), 4)}
    mert, cover = registry_features.novel_features(features, stored, [descriptor(1)],
                                                   mert_room=1, cover_room=5)
    assert [(s, e) for s, e, _v in mert] == [(10.0, 25.0)]
    assert cover is None                                   # bản đổi tông của descriptor đã có

    only_cover = {"mert": [], "cover_ref": descriptor(2)}
    assert registry_features.novel_features(only_cover, [], [descriptor(1)], 5, 5)[1] is not None
    assert registry_features.novel_features(only_cover, [], [], 5, 0)[1] is None   # hết chỗ


def test_vector_qua_bytea_giu_nguyen_va_bao_sai_so_chieu():
    vector = random_unit(7)
    raw = registry_features.to_bytes(vector)
    assert len(raw) == 768 * 4
    assert np.array_equal(registry_features.from_bytes(raw, 768), vector)
    with pytest.raises(ValueError):
        registry_features.from_bytes(raw, 512)


def test_phien_ban_dac_trung_ghi_du_model_revision_cach_cat():
    version = registry_features.mert_version()
    assert config.MERT_MODEL_VERSION in version and "15s" in version
    assert config.MERT_MODEL_REVISION[:12] in version


# --------------------------------------------------------------------------
# Trích đặc trưng từ file (MERT thay bằng hàm giả: không nạp model trong unit test)
# --------------------------------------------------------------------------
def test_trich_dac_trung_dung_so_doan_va_co_cover(monkeypatch, tmp_path):
    calls = []

    def fake_embed(signal, sr):
        calls.append(len(signal) / sr)
        return random_unit(len(calls))

    monkeypatch.setattr(embedding_service, "embed_signal", fake_embed)
    monkeypatch.setattr(registry_features, "noise_prototypes", lambda: np.vstack([random_unit(500)]))
    sr = 24000
    t = np.arange(40 * sr) / sr
    path = tmp_path / "bai.wav"
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)

    features = registry_features.extract_features(str(path))

    assert [(s, e) for s, e, _v in features["mert"]] == [(0.0, 15.0), (10.0, 25.0), (20.0, 35.0)]
    assert calls == pytest.approx([15.0, 15.0, 15.0])
    assert features["cover_ref"].shape == (12 * cover_service.DESCRIPTOR_FRAMES,)
    matrix, factors = features["cover_queries"]
    assert matrix.shape[1] == features["cover_ref"].size and len(factors) == len(matrix)


def test_doan_giong_tieng_on_khong_duoc_hoc(monkeypatch, tmp_path):
    """Hai đoạn ồn trắng khác nhau có cosine MERT 0,998 (đo thật) — không được làm bằng chứng."""
    noise_vector = random_unit(42)
    vectors = iter([near(noise_vector, 0.99, seed=1), random_unit(43), near(noise_vector, 0.999, seed=2)])
    monkeypatch.setattr(embedding_service, "embed_signal", lambda _sig, _sr: next(vectors))
    monkeypatch.setattr(registry_features, "noise_prototypes", lambda: np.vstack([noise_vector]))
    sf.write(tmp_path / "a.wav", np.zeros(40 * 24000, dtype="float32") + 0.01, 24000)

    features = registry_features.extract_features(str(tmp_path / "a.wav"))

    assert [(s, e) for s, e, _v in features["mert"]] == [(10.0, 25.0)]
    assert features["noise_like_segments"] == 2


def test_mau_tieng_on_dung_ba_mau_va_tinh_mot_lan(monkeypatch):
    calls = []
    monkeypatch.setattr(registry_features, "_noise_matrix", None)
    monkeypatch.setattr(embedding_service, "embed_signal",
                        lambda sig, sr: calls.append(len(sig) / sr) or random_unit(len(calls)))
    first = registry_features.noise_prototypes()
    again = registry_features.noise_prototypes()
    assert first.shape == (3, 768) and again is first
    assert calls == pytest.approx([15.0, 15.0, 15.0])


def test_hoc_hong_thi_van_ghi_luot_gap_bang_fingerprint(monkeypatch, tmp_path):
    temp = tmp_path / "bai_normalized.wav"
    temp.write_bytes(b"x")
    monkeypatch.setattr(audio_service, "prepare_for_embedding", lambda _p, _n: (str(temp), str(temp)))

    def boom(_path):
        raise RuntimeError("MERT hỏng")

    monkeypatch.setattr(registry_features, "extract_features", boom)
    assert registry.features_for_registry("x/bai.m4a", "bai.m4a") == (None, "FEATURES_FAILED")
    assert not temp.exists()                               # file tạm vẫn được dọn

    captured = {}
    monkeypatch.setattr(registry, "fingerprint_for_registry", lambda _p: (20.0, b"AQAA"))
    monkeypatch.setattr(registry, "register_sighting",
                        lambda _db, **kw: captured.update(kw) or {"unknown_id": "u"})
    info = registry.record_unknown(None, job_id=None, audio_path="x/bai.m4a", result=make_result(),
                                   usage_context={}, filename="bai.m4a", kind=registry.NOT_IDENTIFIED)
    assert captured["fingerprint"] == (20.0, b"AQAA") and captured["features"] is None
    assert info["features_error"] == "FEATURES_FAILED"


def test_tat_hoc_thi_khong_trich_dac_trung(monkeypatch):
    monkeypatch.setattr(config, "UNKNOWN_REGISTRY_LEARN", False)
    monkeypatch.setattr(registry, "fingerprint_for_registry", lambda _p: None)
    monkeypatch.setattr(registry, "features_for_registry",
                        lambda *_a: pytest.fail("không được trích đặc trưng khi đã tắt học"))
    monkeypatch.setattr(registry, "register_sighting", lambda _db, **kw: {"unknown_id": "u"})
    registry.record_unknown(None, job_id=None, audio_path="a.mp3", result=make_result(),
                            usage_context={}, filename="a.mp3", kind=registry.NOT_IDENTIFIED)


def test_rights_unknown_khong_hoc(monkeypatch):
    """Bài đã có trong kho (RIGHTS_UNKNOWN) có sẵn trong chỉ mục chính — không học lại."""
    monkeypatch.setattr(registry, "features_for_registry",
                        lambda *_a: pytest.fail("RIGHTS_UNKNOWN không cần đặc trưng"))
    monkeypatch.setattr(registry, "register_sighting", lambda _db, **kw: {"unknown_id": "u"})
    registry.record_unknown(None, job_id=None, audio_path="a.mp3",
                            result=make_result("EXACT_MATCH", "UNKNOWN", str(uuid.uuid4())),
                            usage_context={}, filename="a.mp3", kind=registry.RIGHTS_UNKNOWN)


# --------------------------------------------------------------------------
# Tích hợp PostgreSQL: gộp qua MERT / Cover và lưu thêm đặc trưng
# --------------------------------------------------------------------------
def fake_features(mert_vectors, cover_desc, cover_query=None):
    query = cover_desc if cover_query is None else cover_query
    return {"duration": 30.0,
            "mert": [(i * 10.0, i * 10.0 + 15.0, v) for i, v in enumerate(mert_vectors)],
            "cover_ref": cover_desc, "cover_queries": (np.vstack([query]), [1.0])}


@pytest.fixture
def learning_db(db_session):
    if not db_session.execute(text("SELECT to_regclass('unknown_track_features')")).scalar():
        pytest.skip("Chưa có bảng unknown_track_features. Chạy: python init_db.py --schema-only")
    created = []
    yield db_session, created
    db_session.rollback()
    for unknown_id in created:
        db_session.execute(text("DELETE FROM unknown_tracks WHERE unknown_id = :u"), {"u": unknown_id})
    db_session.commit()


@requires_db
def test_so_hoc_roi_nhan_ra_ban_nen_va_ban_doi_tong(learning_db):
    db, created = learning_db
    song = [random_unit(100 + i) for i in range(3)]
    cover = descriptor(100)

    def sight(features, name):
        info = registry.register_sighting(db, job_id=None, result=make_result(),
                                          usage_context={}, filename=name, features=features)
        if info["is_new"]:
            created.append(info["unknown_id"])
        return info

    first = sight(fake_features(song, cover), "bai.mp3")
    assert first["is_new"] and first["match_method"] is None
    assert first["features_learned"] == 4 and first["features_total"] == 4   # 3 MERT + 1 Cover

    # Bản nén/cắt: đoạn gần nhưng không trùng -> nhận qua MERT, học thêm đoạn mới
    compressed = sight(fake_features([near(song[1], 0.99, seed=5), random_unit(900)], descriptor(901)),
                       "bai_nen.m4a")
    assert compressed["unknown_id"] == first["unknown_id"] and compressed["match_method"] == "MERT"
    assert compressed["match_score"] == pytest.approx(0.99, abs=1e-3)
    assert compressed["threshold"] == pytest.approx(config.MERT_THRESHOLD)
    # đoạn đã khớp không lưu lại; chỉ đoạn mới (900) + Cover mới (901)
    assert compressed["features_learned"] == 2 and compressed["sighting_count"] == 2

    # Bản đổi tông: MERT không nhận (MERT mã hoá cao độ tuyệt đối) nhưng Cover nhận
    pitched = sight(fake_features([random_unit(950)], descriptor(951),
                                  cover_query=shift_pitch(cover, 2)), "bai_tong_cao.wav")
    assert pitched["unknown_id"] == first["unknown_id"] and pitched["match_method"] == "COVER"
    assert "cao hơn bản gốc 2 bán cung" in pitched["oti_text"]

    other = sight(fake_features([random_unit(990)], descriptor(991)), "bai_khac.mp3")
    assert other["is_new"] and other["unknown_id"] != first["unknown_id"]

    again = sight(fake_features(song, cover), "bai_gui_lai.mp3")   # gửi lại đúng bài đầu
    assert again["unknown_id"] == first["unknown_id"] and again["features_learned"] == 0

    entry = registry.get_entry(db, first["unknown_id"])
    assert entry["learned"] == {"mert": 3 + 1 + 1, "cover": 3}
    assert [s["match_method"] for s in entry["sightings"]] == ["MERT", "COVER", "MERT", None]
