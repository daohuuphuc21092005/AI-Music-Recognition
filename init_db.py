import os
import re

import pandas as pd
from sqlalchemy import inspect, text

from backend import config
from backend.database.session import engine

def clean_and_filter_df(df: pd.DataFrame, table_name: str, engine) -> pd.DataFrame:
    """
    1. Loại bỏ các đuôi biến dạng như _m999, _m3274 ở cuối tên cột (ví dụ: duration_m999 -> duration)
    2. Chỉ lọc lấy đúng các cột có khai báo trong Bảng CSDL PostgreSQL
    """
    new_columns = {}
    for col in df.columns:
        # Regex xóa _m theo sau bởi các chữ số ở cuối chuỗi
        clean_col = re.sub(r'_m\d+$', '', str(col)).strip()
        new_columns[col] = clean_col
    df = df.rename(columns=new_columns)

    # Lấy danh sách cột thực tế từ Schema DB
    inspector = inspect(engine)
    db_columns = [c['name'] for c in inspector.get_columns(table_name)]

    # Giữ lại cột hợp lệ
    valid_cols = [c for c in df.columns if c in db_columns]
    return df[valid_cols]

def init_database():
    print("=== BẮT ĐẦU TẠO BẢNG & NẠP DỮ LIỆU VÀO POSTGRESQL ===")
    
    create_tables_sql = """
    CREATE TABLE IF NOT EXISTS compositions (
        composition_id UUID PRIMARY KEY,
        title TEXT,
        composer TEXT,
        year INTEGER,
        public_domain_status VARCHAR(50),
        source TEXT,
        verified_at TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS recordings (
        recording_id UUID PRIMARY KEY,
        composition_id UUID REFERENCES compositions(composition_id),
        title TEXT,
        artist TEXT,
        album TEXT,
        release_year INTEGER,
        duration FLOAT,
        source_dataset TEXT,
        source_track_id TEXT,
        audio_path TEXT,
        metadata_verified BOOLEAN
    );

    CREATE TABLE IF NOT EXISTS rights (
        rights_id UUID PRIMARY KEY,
        recording_id UUID REFERENCES recordings(recording_id),
        composition_id UUID REFERENCES compositions(composition_id),
        license_type VARCHAR(100),
        copyright_status VARCHAR(50),
        attribution_required BOOLEAN,
        commercial_use_allowed BOOLEAN,
        modification_allowed BOOLEAN,
        monetization_allowed BOOLEAN,
        revenue_share_required BOOLEAN,
        policy_action VARCHAR(30),
        license_purchased BOOLEAN,
        revenue_share_agreed BOOLEAN,
        recording_public_domain BOOLEAN,
        territory VARCHAR(50),
        platform VARCHAR(50),
        valid_from DATE,
        valid_until DATE,
        source TEXT,
        source_url TEXT,
        verified_at TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS fingerprints (
        fingerprint_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        recording_id UUID REFERENCES recordings(recording_id),
        algorithm VARCHAR(50) DEFAULT 'chromaprint',
        fingerprint TEXT,
        duration FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS embeddings (
        embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        recording_id UUID REFERENCES recordings(recording_id),
        segment_start FLOAT,
        segment_end FLOAT,
        model VARCHAR(50) DEFAULT 'MERT',
        model_version VARCHAR(50) DEFAULT 'MERT-v1-95M',
        dimension INTEGER DEFAULT 768,
        vector TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS test_queries (
        query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        recording_id UUID REFERENCES recordings(recording_id),
        composition_id UUID REFERENCES compositions(composition_id),
        transformation VARCHAR(100),
        snr FLOAT,
        pitch_shift FLOAT,
        tempo_factor FLOAT,
        codec VARCHAR(20),
        bitrate INTEGER,
        segment_start FLOAT,
        duration FLOAT,
        expected_match_type VARCHAR(50)
    );

    -- Bảng jobs & feedback phục vụ API bất đồng bộ ở §11 (analyze/jobs/results/feedback).
    CREATE TABLE IF NOT EXISTS jobs (
        job_id UUID PRIMARY KEY,
        status VARCHAR(20) NOT NULL,
        filename TEXT,
        platform VARCHAR(30),
        commercial_use BOOLEAN,
        monetization BOOLEAN,
        error_code VARCHAR(40),
        error_message TEXT,
        result JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS feedback (
        feedback_id UUID PRIMARY KEY,
        job_id UUID,
        recording_id UUID,
        verdict VARCHAR(20),
        note TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS analysis_results (
        job_id UUID PRIMARY KEY,
        query_id UUID,
        recording_candidate UUID,
        composition_candidate UUID,
        fingerprint_score FLOAT,
        embedding_score FLOAT,
        cover_score FLOAT,
        match_type VARCHAR(50),
        risk_level VARCHAR(50),
        confidence FLOAT,
        decision_reason TEXT,
        latency_ms FLOAT,
        model_version VARCHAR(50),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    # CREATE TABLE IF NOT EXISTS không thêm cột cho bảng đã tồn tại, nên các cột
    # mới của Rule Engine phải thêm bằng ALTER TABLE (dự án không dùng Alembic).
    migrations_sql = """
    ALTER TABLE rights ADD COLUMN IF NOT EXISTS policy_action VARCHAR(30);
    ALTER TABLE rights ADD COLUMN IF NOT EXISTS license_purchased BOOLEAN;
    ALTER TABLE rights ADD COLUMN IF NOT EXISTS revenue_share_agreed BOOLEAN;
    ALTER TABLE rights ADD COLUMN IF NOT EXISTS recording_public_domain BOOLEAN;
    -- Bước đang chạy của job, để màn hình Processing hiển thị tiến trình THẬT
    -- thay vì một thanh loading giả (§13).
    ALTER TABLE jobs ADD COLUMN IF NOT EXISTS stage VARCHAR(40);
    """

    with engine.connect() as conn:
        conn.execute(text(create_tables_sql))
        conn.execute(text(migrations_sql))
        conn.commit()
    print("-> Đã khởi tạo xong 7 bảng chuẩn Đề cương + jobs/feedback cho API §11!")

    data_dir = config.DATA_DIR
    
    files_to_import = [
        ("compositions_master.csv", "compositions"),
        ("metadata_master.csv", "recordings"),
        ("rights_master.csv", "rights"),
        ("fingerprints_master.csv", "fingerprints"),
        ("embeddings_master.csv", "embeddings"),
        ("test_queries_master.csv", "test_queries")
    ]

    # Ghi chú: bản cũ gọi SET session_replication_role = 'replica' để tắt kiểm tra
    # khoá ngoại, nhưng df.to_sql() mở một connection KHÁC nên lệnh đó không có
    # tác dụng gì. Nay giữ nguyên kiểm tra khoá ngoại: dữ liệu sai phải báo lỗi,
    # không được nạp vào im lặng. Thứ tự nạp bên dưới đã đúng chiều phụ thuộc.
    with engine.connect() as conn:
        
        for csv_file, table_name in files_to_import:
            file_path = os.path.join(data_dir, csv_file)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Dọn dẹp tên cột và lọc cột chuẩn
                    df = clean_and_filter_df(df, table_name, engine)
                    
                    # Xóa dữ liệu cũ
                    conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE;"))
                    conn.commit()
                    
                    # Nạp dữ liệu mới
                    df.to_sql(table_name, engine, if_exists='append', index=False, method='multi', chunksize=1000)
                    print(f"-> Nạp thành công {len(df)} bản ghi vào bảng '{table_name}' từ {csv_file}")
                except Exception as e:
                    print(f"❌ Lỗi khi nạp file {csv_file} vào bảng '{table_name}': {e}")
            else:
                print(f"-> Bỏ qua nạp '{table_name}': Chưa tìm thấy file {file_path}")
                
        conn.commit()

    print("\n=== HOÀN THÀNH QUÁ TRÌNH TẠO VÀ NẠP CSDL ===")

if __name__ == "__main__":
    init_database()