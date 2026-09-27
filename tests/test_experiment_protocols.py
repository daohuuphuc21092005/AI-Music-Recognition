"""
Hạ tầng thí nghiệm: giao thức held-out, thứ tự cascade của EXP-04, chốt chặn của
EXP-05 và bước kiểm ngưỡng của run_all_experiments.

Các chỗ này sai thì số liệu vẫn in ra bình thường — không có lỗi nào nổi lên —
nên chỉ test mới bắt được.
"""
import pytest

from experiments import common
from experiments.common import HeldOutProtocol, overlay_partners


def manifest_rows(sources, with_column=False, partners=None):
    rows = []
    for source in sources:
        for transformation in ("crop_15s", "audio_overlay"):
            row = {"source_recording_id": source, "transformation": transformation}
            if with_column:
                row["overlay_source_recording_id"] = (
                    partners[source] if transformation == "audio_overlay" else "")
            rows.append(row)
    return rows


def test_suy_ra_bai_tron_chong_theo_quy_tac_cua_augment_audio():
    # augment_audio.py trộn nguồn thứ i với nguồn thứ i+1, vòng tròn
    assert overlay_partners(manifest_rows(["a", "b", "c"])) == {"a": "b", "b": "c", "c": "a"}


def test_cot_tuong_minh_duoc_uu_tien_hon_suy_luan():
    partners = {"a": "c", "b": "a", "c": "b"}
    rows = manifest_rows(["a", "b", "c"], with_column=True, partners=partners)
    assert overlay_partners(rows) == partners


def test_manifest_ghi_noi_van_suy_dung_phan_cu():
    """--append: phần cũ không có cột, phần mới có — vòng tròn của phần cũ không được lệch."""
    old = manifest_rows(["a", "b", "c"])
    new = manifest_rows(["x", "y"], with_column=True, partners={"x": "y", "y": "x"})
    single = [{"source_recording_id": "z", "transformation": "crop_15s",
               "overlay_source_recording_id": ""}]
    assert overlay_partners(old + new + single) == {
        "a": "b", "b": "c", "c": "a", "x": "y", "y": "x"}


def make_protocol(monkeypatch, rows, fingerprints, neighbours, min_score=0.10):
    monkeypatch.setattr(common, "fingerprint_neighbours",
                        lambda sources, rows=None: {s: neighbours.get(s, []) for s in sources})
    return HeldOutProtocol(rows, min_score=min_score, rows=fingerprints)


def test_held_out_loai_ban_trung_gan_trung_va_bai_tron_chong(monkeypatch):
    fingerprints = [("a", "FP_A"), ("a_dup", "FP_A"), ("b", "FP_B"), ("c", "FP_C"),
                    ("a_edit", "FP_A_EDIT"), ("b_dup", "FP_B"), ("khong_lien_quan", "FP_X")]
    neighbours = {"a": [["a_edit", 0.158], ["khong_lien_quan", 0.077]]}
    protocol = make_protocol(monkeypatch, manifest_rows(["a", "b", "c"]), fingerprints, neighbours)

    crop = {"source_recording_id": "a", "transformation": "crop_15s"}
    overlay = {"source_recording_id": "a", "transformation": "audio_overlay"}

    # Nhãn đúng vẫn chỉ là lớp trùng hệt
    assert protocol.exact_class("a") == {"a", "a_dup"}
    # Held-out: + bản gần trùng trên ngưỡng, KHÔNG kéo theo láng giềng dưới ngưỡng
    assert protocol.exclusions(crop) == {"a", "a_dup", "a_edit"}
    # audio_overlay: + bài bị trộn chồng (b) và bản trùng của nó
    assert protocol.exclusions(overlay) == {"a", "a_dup", "a_edit", "b", "b_dup"}

    described = protocol.describe()
    assert described["sources_with_exact_duplicates"] == 2  # a và b
    assert described["sources_with_near_duplicates"] == 1
    assert described["near_duplicates"] == {"a": [["a_edit", 0.158]]}


def test_cascade_cua_exp04_dung_thu_tu_production():
    from experiments.exp04_hybrid.run import cascade_decision

    def stage(accepted, rec, latency):
        return {"accepted": accepted, "recording_id": rec if accepted else None,
                "score": 0.5, "latency_ms": latency}

    fp, mert, cover = stage(False, "x", 10), stage(False, "y", 20), stage(True, "z", 30)
    full = cascade_decision(fp, mert, cover, with_cover=True)
    assert full["stage"] == "STAGE_3_COVER" and full["recording_id"] == "z"
    assert full["latency_ms"] == 60

    no_cover = cascade_decision(fp, mert, cover, with_cover=False)
    assert no_cover["stage"] == "STAGE_2_MERT_RETRIEVAL" and no_cover["recording_id"] is None

    # Tầng trước khớp thì dừng, tầng sau không được ghi đè
    assert cascade_decision(stage(True, "x", 10), mert, cover, True)["recording_id"] == "x"
    assert cascade_decision(fp, stage(True, "y", 20), cover, True)["stage"] == \
        "STAGE_2_MERT_RETRIEVAL"


def test_exp05_tu_choi_ghep_hai_bo_truy_van_khac_nhau():
    from experiments.exp05_robustness.run import consistency_problems

    def result(sources):
        return {"raw_results": [{"source_recording_id": s, "transformation": "crop_15s"}
                                for s in sources],
                "parameters": {"current_tau_fp": 0.3, "tau_fp": 0.3}}

    assert consistency_problems(result(["a", "b"]), result(["a", "b"])) == []
    problems = consistency_problems(result(["a", "b"]), result(["c", "d"]))
    assert problems and "chung 0" in problems[0]


@pytest.mark.parametrize("gate, expected", [
    ({"EXACT_MATCH": 0.30, "NEAR_MATCH": 0.98, "COVER_MATCH": 0.90}, 0),
    ({"EXACT_MATCH": 0.15, "NEAR_MATCH": 0.98, "COVER_MATCH": 0.90}, 1),
    ({"EXACT_MATCH": 0.30, "NEAR_MATCH": 0.98}, 1),
])
def test_run_all_dung_lai_khi_nguong_lech_rule_engine(gate, expected):
    from scripts.run_all_experiments import threshold_mismatches

    runtime = {"EXACT_MATCH": 0.30, "NEAR_MATCH": 0.98, "COVER_MATCH": 0.90}
    assert len(threshold_mismatches(runtime, gate)) == expected


def test_nguong_runtime_hien_tai_khop_rule_engine():
    """Chính cấu hình đang commit phải qua được bước kiểm của run_all_experiments."""
    from scripts.run_all_experiments import gate_thresholds, threshold_mismatches
    from backend import config

    runtime = {"EXACT_MATCH": config.FP_THRESHOLD, "NEAR_MATCH": config.MERT_THRESHOLD,
               "COVER_MATCH": config.COVER_THRESHOLD}
    assert threshold_mismatches(runtime, gate_thresholds()) == []


def test_exp08_chon_phan_tang_theo_giay_phep():
    from experiments.exp08_end_to_end.run import select_queries

    licenses = {"a": "CC_BY_NC_ND", "b": "CC_BY_NC_ND", "c": "CC_BY_NC_ND", "d": "CC0",
                "e": "CC_BY", "f": "CC0"}
    queries = [{"source_recording_id": s, "path": f"{s}_{t}.wav"}
               for s in "abcdef" for t in ("crop", "pitch")]

    chosen = select_queries(queries, per_license=2, license_of=licenses.get)
    assert {q["source_recording_id"] for q in chosen} == {"a", "b", "d", "e", "f"}
    assert len(chosen) == 10  # đủ mọi biến đổi của bài được chọn

    assert {q["source_recording_id"] for q in select_queries(queries, sources=2)} == {"a", "b"}
    assert select_queries(queries) == queries


@pytest.mark.parametrize("flags, expected_written", [
    (["--write"], {"MERT_THRESHOLD": 0.99}),
    (["--write", "--allow-loosen"], {"FP_THRESHOLD": 0.10, "MERT_THRESHOLD": 0.99}),
])
def test_ap_nguong_tu_dong_siet_khong_tu_dong_noi(monkeypatch, flags, expected_written):
    """
    Đề xuất HẠ ngưỡng (nới) không được tự ghi vào .env — cần --allow-loosen; đề xuất
    NÂNG (siết) thì ghi luôn. EXP-01 từng đề xuất τFP 0.30 -> 0.10 và chủ dự án giữ 0.30.
    """
    import sys

    from scripts import apply_calibrated_thresholds as apply

    monkeypatch.setattr(apply.config, "FP_THRESHOLD", 0.30)
    monkeypatch.setattr(apply.config, "MERT_THRESHOLD", 0.98)
    monkeypatch.setattr(apply.config, "COVER_THRESHOLD", 0.90)
    monkeypatch.setattr(apply, "tau_fp", lambda: (0.10, "nới"))
    monkeypatch.setattr(apply, "tau_mert", lambda: (0.99, "siết"))
    monkeypatch.setattr(apply, "tau_cover", lambda: (0.90, "giữ"))
    written = {}
    monkeypatch.setattr(apply, "write_env", lambda updates: written.update(updates))
    monkeypatch.setattr(sys, "argv", ["apply_calibrated_thresholds.py", *flags])

    assert apply.main() == 0
    assert written == expected_written
