"""
Nhận diện một file audio bằng SQL thuần (không qua API) — tiện để soi nhanh DB.

Trước đây file này nằm trong tests/ với tên test_recognition.py, nhưng hàm
`test_audio_recognition(test_audio_path)` nhận tham số nên pytest coi
`test_audio_path` là một fixture không tồn tại và báo lỗi.

Dùng:  python scripts/recognize_track.py [đường_dẫn_audio]
"""
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from backend.database.session import engine
from backend.services.chromaprint_codec import decode_fingerprint
from backend.services.fingerprint_service import match_decoded

import numpy as np


def get_chromaprint(audio_path):
    """Fingerprint dạng base64 nén — đúng định dạng đang lưu trong bảng fingerprints."""
    try:
        result = subprocess.run(["fpcalc", audio_path], stdout=subprocess.PIPE,
                                text=True, check=True)
        duration, fingerprint = 0.0, ""
        for line in result.stdout.splitlines():
            if line.startswith("DURATION="):
                duration = float(line.split("=")[1])
            elif line.startswith("FINGERPRINT="):
                fingerprint = line.split("=")[1]
        return fingerprint, duration
    except FileNotFoundError:
        print("❌ Chưa có fpcalc trong PATH. "
              "Tải tại https://github.com/acoustid/chromaprint/releases")
        return None, 0.0
    except Exception as e:
        print(f"Lỗi fpcalc: {e}")
        return None, 0.0


def recognize(audio_path: str) -> int:
    print(f"🔍 Đang trích xuất dữ liệu từ file: {audio_path}")
    fp, duration = get_chromaprint(audio_path)
    if not fp:
        return 1

    query = np.asarray(decode_fingerprint(fp)[0], dtype=np.uint32)

    sql = text("""
        SELECT r.recording_id, r.title AS song_title, r.artist,
               c.title AS composition_title, rg.license_type, rg.copyright_status,
               f.fingerprint
        FROM fingerprints f
        JOIN recordings r ON f.recording_id = r.recording_id
        LEFT JOIN compositions c ON r.composition_id = c.composition_id
        LEFT JOIN rights rg ON r.recording_id = rg.recording_id
    """)

    # So khớp bằng chính thuật toán Chromaprint thay vì LEVENSHTEIN trên chuỗi
    # base64: khoảng cách chỉnh sửa trên chuỗi đã nén KHÔNG phản ánh độ giống
    # nhau của âm thanh, đó là lý do bản cũ cho kết quả vô nghĩa.
    best, best_score = None, 0.0
    with engine.connect() as conn:
        for row in conn.execute(sql):
            try:
                candidate = np.asarray(decode_fingerprint(row.fingerprint)[0],
                                       dtype=np.uint32)
            except Exception:
                continue
            score = match_decoded(query, candidate)
            if score > best_score:
                best, best_score = row, score

    if not best:
        print("\n❌ Không tìm thấy bài hát tương thích trong CSDL.")
        return 0

    print("\n🎉 === BẢN GHI GIỐNG NHẤT TRONG CSDL ===")
    print(f"🎵 Tên bài hát       : {best.song_title}")
    print(f"🎤 Ca sĩ/Artist      : {best.artist}")
    print(f"🎼 Tác phẩm gốc      : {best.composition_title}")
    print(f"📜 Loại bản quyền    : {best.license_type}")
    print(f"🛡️ Trạng thái bản quyền: {best.copyright_status}")
    print(f"🆔 Recording ID      : {best.recording_id}")
    print(f"📊 Điểm Chromaprint  : {best_score:.4f}")
    return 0


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test_audio/bai_hat_test.mp3"
    raise SystemExit(recognize(path))
