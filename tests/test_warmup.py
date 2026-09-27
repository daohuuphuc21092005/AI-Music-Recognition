"""
Làm nóng lúc khởi động + khoá của ba bộ đệm nạp lười.

Vì sao cần khoá: luồng làm nóng chạy song song với request đến sớm. Không khoá thì
cả hai cùng thấy bộ đệm rỗng và cùng nạp — với MERT là hai bản model trên GPU 4 GB,
với fingerprint là giải mã ~24.000 dòng (~20 s) hai lần.
"""
import threading
import time

import numpy as np
from fastapi.testclient import TestClient

from backend import main
from backend.services import cover_service, embedding_service, fingerprint_service

N_THREADS = 4


def run_together(fn, n: int = N_THREADS) -> list:
    """Gọi `fn` từ n luồng xuất phát cùng lúc; trả kết quả của từng luồng."""
    barrier = threading.Barrier(n)
    results = [None] * n

    def worker(i):
        barrier.wait()
        results[i] = fn()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return results


def test_mert_chi_nap_mot_lan_khi_nhieu_luong_cung_goi(monkeypatch):
    calls = []

    class FakeModel:
        device = None

        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append("model")
            time.sleep(0.2)
            return FakeModel()

    class FakeExtractor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return "processor"

    monkeypatch.setattr(embedding_service, "_model", None)
    monkeypatch.setattr(embedding_service, "_processor", None)
    monkeypatch.setattr(embedding_service, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(embedding_service, "Wav2Vec2FeatureExtractor", FakeExtractor)

    results = run_together(embedding_service.load_model)

    assert calls == ["model"]
    assert len({id(model) for _, model in results}) == 1
    # Không luồng nào nhận model chưa được đưa lên thiết bị
    assert all(model.device is not None for _, model in results)


def test_chi_muc_cover_chi_nap_mot_lan(monkeypatch):
    calls = []

    def slow_load():
        calls.append(1)
        time.sleep(0.2)
        return cover_service.CoverIndex(matrix=np.zeros((3, 4), dtype=np.float32),
                                        id_map=["a", "b", "c"])

    monkeypatch.setattr(cover_service, "_default_cover_index", None)
    monkeypatch.setattr(cover_service, "_index_loaded", False)
    monkeypatch.setattr(cover_service, "load_cover_index", slow_load)

    results = run_together(cover_service.get_default_cover_index)

    assert len(calls) == 1
    assert len({id(index) for index in results}) == 1


class _FakeResult:
    def __init__(self, one=None, rows=None):
        self._one, self._rows = one, rows

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._rows


class _FakeDb:
    """Đủ cho `_reference_fingerprints`: một câu lấy chữ ký bảng, một câu lấy dữ liệu."""

    def __init__(self, rows):
        self.rows = rows

    def execute(self, statement):
        if "count(*)" in str(statement):
            return _FakeResult(one=(len(self.rows), "id-min", "id-max"))
        return _FakeResult(rows=self.rows)


def test_fingerprint_tham_chieu_chi_giai_ma_mot_lan(monkeypatch):
    rows = [(f"rec-{i}", f"fp-{i}", 30.0) for i in range(20)]
    decoded = []

    def slow_decode(raw):
        decoded.append(raw)
        time.sleep(0.01)
        return [1, 2, 3], 1

    monkeypatch.setattr(fingerprint_service, "_reference_cache",
                        {"signature": None, "data": ([], 0)})
    monkeypatch.setattr(fingerprint_service, "decode_fingerprint", slow_decode)

    db = _FakeDb(rows)
    results = run_together(lambda: fingerprint_service._reference_fingerprints(db))

    assert len(decoded) == len(rows)            # không phải 4 × 20
    assert len({id(references) for references, _ in results}) == 1
    assert all(len(references) == len(rows) for references, _ in results)


def test_lam_nong_ghi_trang_thai_tung_buoc_va_khong_lo_noi_dung_loi(monkeypatch):
    def broken():
        raise RuntimeError(r"C:\Users\ai-do\model\khong-mo-duoc.bin")

    monkeypatch.setattr(main, "_warm_fingerprints", lambda: {"references": 3})
    monkeypatch.setattr(main.embedding_service, "warm_up", broken)

    report = {"status": "RUNNING", "steps": {}}
    main.warm_up(report, {"fingerprint": True, "mert": True, "cover": False})

    assert report["status"] == "DONE"
    assert report["steps"]["fingerprint"]["status"] == "READY"
    # Một bước hỏng không chặn các bước sau, và nội dung lỗi chỉ nằm trong log
    assert report["steps"]["mert"]["status"] == "FAILED"
    assert report["steps"]["cover"] == {"status": "SKIPPED"}
    assert "khong-mo-duoc" not in str(report)


def test_lifespan_chay_lam_nong_nen_va_health_bao_trang_thai(monkeypatch):
    started = threading.Event()
    captured = {}

    def fake_warm_up(report, plan):
        captured["plan"] = plan
        report["steps"]["fingerprint"] = {"status": "READY", "seconds": 0.0}
        report["status"] = "DONE"
        started.set()

    monkeypatch.setattr(main.config, "WARMUP_ON_STARTUP", True)
    monkeypatch.setattr(main, "warm_up", fake_warm_up)

    with TestClient(main.app) as client:
        assert started.wait(10), "lifespan không khởi chạy luồng làm nóng"
        body = client.get("/health").json()

    assert set(captured["plan"]) == {"fingerprint", "mert", "cover"}
    assert body["warmup"]["status"] == "DONE"
    assert body["warmup"]["steps"]["fingerprint"]["status"] == "READY"
