"""
Chứng minh bản so khớp vector hoá bằng numpy cho kết quả GIỐNG HỆT bản tham
chiếu thuần Python của pyacoustid (`acoustid._match_fingerprints`).

Bản vector hoá nhanh hơn ~65 lần, nhưng chỉ được phép dùng nếu nó không làm
thay đổi điểm số — nếu không, mọi ngưỡng và mọi kết quả thí nghiệm sau này sẽ
không còn so sánh được với chuẩn Chromaprint.
"""
import csv
import os

import acoustid
import numpy as np
import pytest

from backend import config
from backend.services.chromaprint_codec import decode_fingerprint
from backend.services.fingerprint_service import MAX_ALIGN_OFFSET, match_decoded

csv.field_size_limit(10 ** 9)


def load_reference_fingerprints(limit: int = 6):
    path = os.path.join(config.DATA_DIR, "fingerprints_master.csv")
    if not os.path.exists(path):
        pytest.skip("Chưa có fingerprints_master.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return [row for _, row in zip(range(limit), csv.DictReader(f))]


def as_array(fp) -> np.ndarray:
    return np.asarray(decode_fingerprint(fp)[0], dtype=np.uint32)


def test_vectorized_matches_reference_implementation():
    """
    Khi giới hạn cùng dải offset ±120 item, bản của ta phải cho điểm GIỐNG HỆT
    `acoustid._match_fingerprints`. (Mặc định hệ thống dò toàn bộ offset — xem
    EXP-01 — nên phải truyền max_align_offset để so sánh đúng điều kiện.)
    """
    rows = load_reference_fingerprints(5)
    query = as_array(rows[0]["fingerprint"])

    for row in rows:
        candidate = as_array(row["fingerprint"])
        fast = match_decoded(query, candidate, max_align_offset=MAX_ALIGN_OFFSET)
        reference = acoustid._match_fingerprints(
            [int(x) for x in query], [int(x) for x in candidate]
        )
        assert fast == pytest.approx(reference, abs=1e-12)


def test_do_toan_bo_offset_bat_duoc_doan_cat_giua_bai():
    """Đoạn cắt từ giữa bài chỉ khớp khi dò toàn bộ offset."""
    rows = load_reference_fingerprints(1)
    full = as_array(rows[0]["fingerprint"])
    if full.size < 400:
        pytest.skip("Fingerprint quá ngắn để cắt phần giữa")

    middle = full[250:450]  # ~31s -> ~56s, lệch xa hơn cửa sổ ±120 item

    narrow = match_decoded(middle, full, max_align_offset=MAX_ALIGN_OFFSET)
    wide = match_decoded(middle, full, max_align_offset=0)

    assert wide == pytest.approx(1.0), "dò toàn bộ offset phải khớp tuyệt đối"
    assert narrow < 0.5, "cửa sổ hẹp của pyacoustid không bắt được đoạn giữa bài"


def test_self_match_is_one():
    rows = load_reference_fingerprints(1)
    vector = as_array(rows[0]["fingerprint"])
    assert match_decoded(vector, vector) == pytest.approx(1.0)


def test_empty_fingerprint_scores_zero():
    empty = np.array([], dtype=np.uint32)
    other = np.array([1, 2, 3], dtype=np.uint32)
    assert match_decoded(empty, other) == 0.0
    assert match_decoded(other, empty) == 0.0


def test_unrelated_tracks_score_low():
    """Hai bản ghi khác nhau phải cho điểm thấp hơn nhiều so với ngưỡng."""
    rows = load_reference_fingerprints(6)
    a = as_array(rows[0]["fingerprint"])
    scores = [match_decoded(a, as_array(r["fingerprint"])) for r in rows[1:]]
    assert max(scores) < config.FP_THRESHOLD
