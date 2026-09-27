"""
Sinh CSV mẫu cho các bảng còn thiếu.

CẢNH BÁO: bản cũ GHI ĐÈ vô điều kiện và đã xoá mất test_queries_master.csv thật
(từ hàng trăm dòng còn đúng 1 dòng mẫu). Nay script từ chối ghi đè file đã tồn
tại, trừ khi chạy với cờ --force.
"""
import argparse
import os
import uuid
from datetime import datetime

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true",
                    help="Cho phép ghi đè file đã tồn tại (mất dữ liệu hiện có)")
args = parser.parse_args()

os.makedirs("data/processed", exist_ok=True)


def write_csv(df: pd.DataFrame, path: str) -> None:
    if os.path.exists(path) and not args.force:
        print(f"⏭️  Bỏ qua {path}: đã tồn tại (dùng --force nếu thực sự muốn ghi đè)")
        return
    df.to_csv(path, index=False)
    print(f"✅ Đã ghi {path}")

# 1. Sinh file embeddings_master.csv chuẩn 100% Schema Đề cương
emb_data = [{
    "embedding_id": str(uuid.uuid4()),
    "recording_id": str(uuid.uuid4()),
    "segment_start": 0.0,
    "segment_end": 15.0,
    "model": "MERT",
    "model_version": "MERT-v1-95M",
    "dimension": 768,
    "vector": ",".join(["0.0"] * 768),
    "created_at": datetime.now().isoformat()
}]
write_csv(pd.DataFrame(emb_data), "data/processed/embeddings_master.csv")

# 2. Sinh file test_queries_master.csv chuẩn Schema Đề cương
test_data = [{
    "query_id": str(uuid.uuid4()),
    "recording_id": str(uuid.uuid4()),
    "composition_id": None,
    "transformation": "crop_15s",
    "snr": 15.0,
    "pitch_shift": 0.0,
    "tempo_factor": 1.0,
    "codec": "mp3",
    "bitrate": 320,
    "segment_start": 0.0,
    "duration": 15.0,
    "expected_match_type": "EXACT_MATCH"
}]
write_csv(pd.DataFrame(test_data), "data/processed/test_queries_master.csv")

print("Hoàn tất.")