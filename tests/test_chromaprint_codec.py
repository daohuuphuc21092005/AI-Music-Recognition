"""
Chứng minh bộ giải nén Chromaprint thuần Python là ĐÚNG.

Phép kiểm chứng: với cùng một file audio, chuỗi base64 do `fpcalc` sinh ra,
sau khi giải nén bằng module của chúng ta, phải trùng KHÍT từng phần tử với dãy
số nguyên do `fpcalc -raw` sinh ra. Đây là lý do duy nhất khiến việc bỏ thư viện
native libchromaprint là chấp nhận được.
"""
import subprocess

import pytest

from backend.services.chromaprint_codec import (
    InvalidFingerprintError,
    decode_fingerprint,
)
from tests.conftest import TEST_AUDIO, TEST_AUDIO_ALT, requires_fpcalc


def run_fpcalc(path: str, raw: bool = False) -> dict:
    cmd = ["fpcalc"] + (["-raw"] if raw else []) + [path]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True).stdout
    parsed = {}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        parsed[key] = value
    return parsed


@requires_fpcalc
@pytest.mark.parametrize("audio_path", [TEST_AUDIO, TEST_AUDIO_ALT])
def test_decode_matches_fpcalc_raw(audio_path):
    """Giải nén base64 phải ra đúng dãy số của fpcalc -raw."""
    import os
    if not os.path.exists(audio_path):
        pytest.skip(f"Không có file audio: {audio_path}")

    compressed = run_fpcalc(audio_path)["FINGERPRINT"]
    expected = [int(x) for x in run_fpcalc(audio_path, raw=True)["FINGERPRINT"].split(",")]

    decoded, algorithm = decode_fingerprint(compressed)

    assert algorithm == 1
    assert len(decoded) == len(expected)
    assert decoded == expected


def test_decode_rejects_garbage():
    with pytest.raises(InvalidFingerprintError):
        decode_fingerprint("AQ")  # ngắn hơn 4 byte header


def test_decode_accepts_bytes_and_str():
    """API phải nhận cả str lẫn bytes như thư viện native."""
    import csv
    import os

    from backend import config

    csv.field_size_limit(10 ** 9)
    path = os.path.join(config.DATA_DIR, "fingerprints_master.csv")
    if not os.path.exists(path):
        pytest.skip("Chưa có fingerprints_master.csv")

    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    as_str, _ = decode_fingerprint(row["fingerprint"])
    as_bytes, _ = decode_fingerprint(row["fingerprint"].encode("ascii"))

    assert as_str == as_bytes
    assert len(as_str) > 0
