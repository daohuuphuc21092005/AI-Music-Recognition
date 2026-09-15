"""
Sinh chỉ mục Cover (CQT chroma 768 chiều) cho các bản ghi tham chiếu.

Dùng:
    python scripts/build_cover_index.py
"""
import json
import os
import sys
import time
from glob import glob

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from backend.services import cover_service


def main() -> int:
    t0 = time.time()
    print("=== DỰNG CHỈ MỤC COVER REFERENCE (CQT CHROMA + OTI) ===")

    metadata_path = config.METADATA_CSV
    if not os.path.exists(metadata_path):
        print(f"❌ Không tìm thấy metadata: {metadata_path}")
        return 1

    df_meta = pd.read_csv(metadata_path)
    # Bản đồ fma/xx_yyyy.mp3 -> recording_id
    fma_to_id = {}
    for _, row in df_meta.iterrows():
        ap = str(row.get("audio_path", ""))
        rec_id = str(row.get("recording_id", ""))
        if ap and ap.startswith("fma/"):
            # Chuẩn hoá key: "00_0006" hoặc "fma/00_0006.mp3"
            base_key = os.path.splitext(os.path.basename(ap))[0]
            fma_to_id[base_key] = rec_id
            fma_to_id[ap] = rec_id

    # 1. Tìm các file audio tham chiếu có sẵn
    # Ưu tiên các file gốc 30s trong data/test_queries/
    test_queries_dir = os.path.join(config.BASE_DIR, "data", "test_queries")
    orig_files = glob(os.path.join(test_queries_dir, "*__original_crop30s.wav"))

    found_targets = []  # list of (audio_path, recording_id)
    seen_ids = set()

    for path in sorted(orig_files):
        fname = os.path.basename(path)
        track_key = fname.split("__")[0]  # e.g. "00_0006"
        rec_id = fma_to_id.get(track_key)
        if rec_id and rec_id not in seen_ids:
            found_targets.append((path, rec_id))
            seen_ids.add(rec_id)

    # Thử thêm AUDIO_ROOT nếu có
    if os.path.exists(config.AUDIO_ROOT):
        for root, _, files in os.walk(config.AUDIO_ROOT):
            for f in files:
                if f.endswith((".mp3", ".wav", ".flac")):
                    base_key = os.path.splitext(f)[0]
                    rec_id = fma_to_id.get(base_key)
                    if rec_id and rec_id not in seen_ids:
                        found_targets.append((os.path.join(root, f), rec_id))
                        seen_ids.add(rec_id)

    print(f"Tìm thấy {len(found_targets)} bản ghi có audio tham chiếu.")
    if not found_targets:
        print("⚠️ Không tìm thấy file audio nào để tạo chỉ mục.")
        return 1

    descriptors = []
    id_map = []

    print(f"Đang trích xuất CQT chroma descriptors ({cover_service.DESCRIPTOR_FRAMES} frames x 12 chroma = 768 chiều)...")
    for i, (audio_path, rec_id) in enumerate(found_targets, 1):
        try:
            desc = cover_service.descriptor_from_file(audio_path, offset=0.0, duration=30.0)
            descriptors.append(desc)
            id_map.append(rec_id)
            if i % 25 == 0 or i == len(found_targets):
                print(f"  [{i:3d}/{len(found_targets):3d}] {os.path.basename(audio_path)} -> {rec_id[:8]}...")
        except Exception as e:
            print(f"  Lỗi trích xuất {audio_path}: {e}")

    if not descriptors:
        print("❌ Không trích xuất được descriptor nào.")
        return 1

    matrix = np.vstack(descriptors).astype("float32")
    os.makedirs(os.path.dirname(config.COVER_INDEX_PATH), exist_ok=True)

    np.save(config.COVER_INDEX_PATH, matrix)
    with open(config.COVER_ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2)

    elapsed = round(time.time() - t0, 2)
    print(f"\n✅ ĐÃ TẠO THÀNH CÔNG CHỈ MỤC COVER:")
    print(f"  Ma trận descriptors: {config.COVER_INDEX_PATH} (shape {matrix.shape})")
    print(f"  Bản đồ ID          : {config.COVER_ID_MAP_PATH} ({len(id_map)} bản ghi)")
    print(f"  Thời gian thực hiện: {elapsed}s")

    # Kiểm tra nạp lại
    loaded = cover_service.load_cover_index(config.COVER_INDEX_PATH, config.COVER_ID_MAP_PATH)
    if loaded and loaded.n_items == len(id_map):
        print("✅ Kiểm tra nạp lại chỉ mục: HỢP LỆ")
        return 0
    else:
        print("❌ Lỗi kiểm tra nạp lại chỉ mục.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
