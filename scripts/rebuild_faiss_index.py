"""
Dựng lại FAISS index + bản đồ ID từ embeddings_master.csv.

Bối cảnh (xem plan Giai đoạn 0):
  - data/processed/mert_faiss.index cũ bị CẮT CỤT (header khai 80.509 vector
    nhưng file chỉ chứa ~3.200) -> faiss.read_index() ném lỗi -> API crash.
  - embeddings_master.csv có recording_id là ID nguồn của dataset (vd. Jamendo
    "1391859") chứ không phải UUID, nên không join được với bảng recordings.
    Script gốc build_full_mert_index.py tách ID từ tên file WAV nên mới như vậy.

Script này:
  1. Đọc CSV, loại các dòng vector sai số chiều.
  2. Ánh xạ recording_id: source_track_id -> UUID thật (metadata_master.csv).
  3. Chuẩn hoá L2 -> IndexFlatIP (cosine) -> ghi index + faiss_id_map.json.
  4. Ghi lại CSV với khoá UUID để init_db.py nạp được vào bảng embeddings.

Khôi phục CSV gốc nếu cần: git checkout HEAD -- data/processed/embeddings_master.csv
"""
import csv
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np

from backend import config

csv.field_size_limit(10 ** 9)


def load_source_id_map(metadata_csv: str) -> dict:
    """source_track_id -> recording_id (UUID). Bỏ các ID xuất hiện nhiều lần."""
    pairs = []
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append((row["source_track_id"], row["recording_id"]))

    counts = Counter(src for src, _ in pairs)
    ambiguous = {src for src, n in counts.items() if n > 1}
    if ambiguous:
        print(f"⚠️  {len(ambiguous)} source_track_id trùng lặp trong metadata -> bỏ qua: "
              f"{sorted(ambiguous)[:5]}")
    return {src: rec for src, rec in pairs if src not in ambiguous}


def main():
    emb_csv = config.EMBEDDINGS_CSV
    meta_csv = config.METADATA_CSV
    dim = config.EMBEDDING_DIM

    print("=== DỰNG LẠI FAISS INDEX TỪ EMBEDDINGS_MASTER.CSV ===")
    for path in (emb_csv, meta_csv):
        if not os.path.exists(path):
            print(f"❌ Không tìm thấy: {path}")
            return 1

    src_to_uuid = load_source_id_map(meta_csv)
    print(f"📖 metadata_master.csv: {len(src_to_uuid)} source_track_id duy nhất")

    vectors, rows_out = [], []
    stats = Counter()

    with open(emb_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            stats["total"] += 1
            parts = row["vector"].split(",")
            if len(parts) != dim:
                stats["bad_dim"] += 1
                continue

            raw_id = str(row["recording_id"]).strip()
            if raw_id in src_to_uuid:
                rec_uuid = src_to_uuid[raw_id]
                stats["remapped"] += 1
            elif len(raw_id) == 36 and raw_id.count("-") == 4:
                rec_uuid = raw_id  # đã là UUID (chạy lại lần 2)
                stats["already_uuid"] += 1
            else:
                stats["unmapped"] += 1
                continue

            try:
                vec = np.asarray(parts, dtype="float32")
            except ValueError:
                stats["bad_values"] += 1
                continue

            row["recording_id"] = rec_uuid
            rows_out.append(row)
            vectors.append(vec)

    print(f"📊 Tổng dòng: {stats['total']} | sai số chiều: {stats['bad_dim']} | "
          f"đổi khoá được: {stats['remapped']} | đã là UUID: {stats['already_uuid']} | "
          f"không ánh xạ được: {stats['unmapped']}")

    if not vectors:
        print("❌ Không có vector hợp lệ nào. Dừng lại, KHÔNG ghi đè file cũ.")
        return 1

    matrix = np.vstack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_rows = int((norms == 0).sum())
    if zero_rows:
        print(f"⚠️  {zero_rows} vector có norm = 0 (bị bỏ qua khi chuẩn hoá)")
    matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss.write_index(index, config.FAISS_INDEX_PATH)

    id_map = {
        "model": config.MERT_MODEL,
        "model_version": config.MERT_MODEL_VERSION,
        "dimension": dim,
        "count": len(rows_out),
        "source_csv": os.path.basename(emb_csv),
        "recording_ids": [r["recording_id"] for r in rows_out],
        "segments": [[float(r["segment_start"]), float(r["segment_end"])] for r in rows_out],
    }
    with open(config.FAISS_ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id_map, f)

    with open(emb_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    unique_recs = len(set(id_map["recording_ids"]))
    print(f"\n✅ Index: {config.FAISS_INDEX_PATH} -> ntotal={index.ntotal}, dim={index.d}")
    print(f"✅ ID map: {config.FAISS_ID_MAP_PATH} -> {len(rows_out)} vector / "
          f"{unique_recs} recording duy nhất")
    print(f"✅ Đã ghi lại {emb_csv} với recording_id dạng UUID")
    print(f"\nℹ️  Stage 2 (MERT) chỉ phủ {unique_recs} recording. "
          f"Phần còn lại của reference DB chưa có embedding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
