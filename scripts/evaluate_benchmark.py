"""
Đo độ trễ pipeline trên reference database THẬT.

Bản cũ dựng 10 vector ngẫu nhiên rồi đo thời gian chạy trên đó, nên con số thu
được không phản ánh hệ thống thật (không chạm tới FAISS index, không chạm tới
114 recording có embedding).

Đây chỉ là phép đo latency phục vụ phát triển. Các thí nghiệm chính thức
EXP-01..EXP-08 sẽ nằm trong thư mục experiments/ ở giai đoạn sau.
"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from backend.database.session import SessionLocal
from backend.services.cascade_service import process_music_query
from backend.services.retrieval_service import load_index


def run_benchmark(test_audio: str = "test.mp3"):
    if not os.path.exists(test_audio):
        print(f"❌ Không tìm thấy file audio: {test_audio}")
        return 1

    db = SessionLocal()
    try:
        vector_index = load_index(config.FAISS_INDEX_PATH, config.FAISS_ID_MAP_PATH)

        start_time = time.time()
        result = process_music_query(test_audio, db, vector_index, top_k=config.TOP_K)
        latency_ms = (time.time() - start_time) * 1000

        evidence = result.get("evidence", {})
        print("\n========= KẾT QUẢ BENCHMARK KĨ THUẬT =========")
        print(f"1. Stage kích hoạt : {result.get('pipeline_stage')}")
        print(f"2. Match Type      : {result.get('match_type')}")
        print(f"3. Latency tổng    : {latency_ms:.2f} ms")
        print(f"4. Độ trễ từng bước: {evidence.get('timings_ms')}")
        print(f"5. Số Candidates   : {len(result.get('candidates', []))}")
        print(f"6. Reference DB    : {vector_index.ntotal} vector / "
              f"{vector_index.n_recordings} recording")
        print(f"7. Ngưỡng áp dụng  : {evidence.get('thresholds')}")
        print(f"8. Lý do quyết định: {evidence.get('decision_reason')}")
        print("==============================================\n")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(run_benchmark(sys.argv[1] if len(sys.argv) > 1 else "test.mp3"))
