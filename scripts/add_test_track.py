import os
import sys
import uuid
import subprocess
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from backend.database.session import engine


def get_chromaprint(audio_path):
    """
    Lấy fingerprint dạng base64 nén — ĐÚNG định dạng đang lưu trong
    fingerprints_master.csv. Bản cũ dùng cờ `-raw` (mảng số nguyên) nên bài hát
    tự nạp vào không bao giờ khớp được khi so sánh.
    """
    try:
        result = subprocess.run(["fpcalc", audio_path], stdout=subprocess.PIPE, text=True, check=True)
        duration, fingerprint = 0.0, ""
        for line in result.stdout.splitlines():
            if line.startswith("DURATION="): duration = float(line.split("=")[1])
            elif line.startswith("FINGERPRINT="): fingerprint = line.split("=")[1]
        return fingerprint, duration
    except Exception as e:
        print(f"Lỗi fpcalc: {e}")
        return None, 0.0

def inject_test_track(audio_path):
    if not os.path.exists(audio_path):
        print(f"❌ Không tìm thấy file nhạc tại: {audio_path}")
        return

    print(f"🎵 Đang xử lý file: {audio_path}")
    fingerprint, duration = get_chromaprint(audio_path)
    
    if not fingerprint:
        print("❌ Trích xuất vân âm thanh thất bại!")
        return

    # Khởi tạo các ID và thời gian dùng chung
    rec_id = str(uuid.uuid4())
    comp_id = str(uuid.uuid4())
    rights_id = str(uuid.uuid4())
    current_time = datetime.now()
    
    with engine.connect() as conn:
        # 1. Thêm Tác phẩm (Composition) - Khớp 7 cột
        conn.execute(text("""
            INSERT INTO compositions (
                composition_id, title, composer, year, 
                public_domain_status, source, verified_at
            )
            VALUES (
                :comp_id, 'Test Song', 'Test Composer', 2024, 
                'no', 'manual_test_injection', :time
            )
        """), {"comp_id": comp_id, "time": current_time})

        # 2. Thêm Bản ghi (Recording) - Khớp 11 cột
        conn.execute(text("""
            INSERT INTO recordings (
                recording_id, composition_id, title, artist, album, 
                release_year, duration, source_dataset, source_track_id, 
                audio_path, metadata_verified
            )
            VALUES (
                :rec_id, :comp_id, 'Test Song (Original)', 'Test Artist', 'Test Album', 
                2024, :duration, 'local_test', 'track_001', 
                :path, true
            )
        """), {"rec_id": rec_id, "comp_id": comp_id, "duration": duration, "path": audio_path})

        # 3. Thêm Vân âm thanh (Fingerprint) - Khớp các cột (không cần id vì tự sinh mặc định)
        conn.execute(text("""
            INSERT INTO fingerprints (
                recording_id, algorithm, fingerprint, duration, created_at
            )
            VALUES (
                :rec_id, 'chromaprint', :fp, :duration, :time
            )
        """), {"rec_id": rec_id, "fp": fingerprint, "duration": duration, "time": current_time})

        # 4. Thêm Bản quyền (Rights) - Khớp 15 cột
        conn.execute(text("""
            INSERT INTO rights (
                rights_id, recording_id, composition_id, license_type, copyright_status, 
                attribution_required, commercial_use_allowed, modification_allowed, 
                monetization_allowed, revenue_share_required, territory, platform, 
                source, source_url, verified_at
            )
            VALUES (
                :rid, :rec_id, :comp_id, 'CC_BY', 'PROTECTED', 
                true, false, false, 
                false, true, 'WORLDWIDE', 'ALL', 
                'manual_test_injection', 'http://example.com/license', :time
            )
        """), {"rid": rights_id, "rec_id": rec_id, "comp_id": comp_id, "time": current_time})
        
        conn.commit()
    
    print(f"✅ Đã thêm thành công bài hát vào CSDL!")
    print(f"🆔 Recording ID của bài hát này là: {rec_id}")

if __name__ == "__main__":
    # Thay 'bai_hat_test.mp3' bằng tên file bạn vừa upload vào thư mục test_audio
    inject_test_track("test_audio/bai_hat_test.mp3")