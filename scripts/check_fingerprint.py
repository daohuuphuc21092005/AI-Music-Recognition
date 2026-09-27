"""
Chạy thử tầng 1 (Chromaprint) trên một file audio bất kỳ.

Trước đây file này nằm ở thư mục gốc với tên test_fingerprint.py nên pytest tự
động thu thập nó và thực thi code ở tầng module (kết nối DB, chạy nhận diện)
ngay lúc collect — cả bộ test đỏ vì lý do không liên quan.

Dùng:  python scripts/check_fingerprint.py [đường_dẫn_audio]
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.session import SessionLocal
from backend.services.fingerprint_service import (
    FingerprintBackendUnavailable,
    backend_status,
    search_fingerprint,
)


def main(audio_path: str = "test.mp3") -> int:
    if not os.path.exists(audio_path):
        print(f"❌ Không tìm thấy file: {audio_path}")
        return 1

    print(f"Trạng thái Chromaprint: {backend_status()}")
    print(f"Đang phân tích: {audio_path}")

    db = SessionLocal()
    try:
        result = search_fingerprint(db, audio_path)
    except FingerprintBackendUnavailable as e:
        print(f"❌ {e}")
        return 1
    finally:
        db.close()

    print("\n--- KẾT QUẢ TẦNG 1 (CHROMAPRINT) ---")
    for key in ("match_type", "recording_id", "fingerprint_score", "threshold",
                "candidates_compared", "query_duration",
                "best_candidate_below_threshold"):
        if key in result:
            print(f"{key:32}: {result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "test.mp3"))
