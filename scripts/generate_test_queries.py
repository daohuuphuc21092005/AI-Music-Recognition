"""
Sinh kế hoạch truy vấn kiểm thử (bảng test_queries) theo bộ biến đổi ở §8.

Đây là BẢN KẾ HOẠCH, không phải file audio: mỗi dòng mô tả một truy vấn cần
dựng (cắt đoạn nào, nén codec gì, nhiễu bao nhiêu dB, dịch cao độ mấy nửa cung).
Khi đã có audio gốc, `scripts/augment_audio.py` đọc bảng này để tạo file thật.

Bản cũ chỉ lấy 20 bản ghi đầu tiên và 6 biến đổi, thiếu hẳn nhóm truy vấn
"ngoài cơ sở dữ liệu" — mà đó chính là thứ EXP-06 (Unknown Track Detection) cần.
"""
import argparse
import csv
import json
import os
import sys
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from backend import config
from backend.database.session import engine

OUTPUT_CSV = os.path.join(config.DATA_DIR, "test_queries_master.csv")

# Bộ biến đổi bám theo CLAUDE.md §8 (Data augmentation cho test set)
TRANSFORMATIONS = [
    # name,                    snr,  pitch, tempo, codec,  bitrate, seg_start, dur,  expected
    ("original",               None,  0.0,  1.00, "wav",   1411,    0.0,      30.0, "EXACT_MATCH"),
    ("crop_15s",               None,  0.0,  1.00, "wav",   1411,   30.0,      15.0, "EXACT_MATCH"),
    ("crop_10s",               None,  0.0,  1.00, "wav",   1411,   60.0,      10.0, "EXACT_MATCH"),
    ("mp3_128k",               None,  0.0,  1.00, "mp3",    128,    0.0,      30.0, "EXACT_MATCH"),
    ("mp3_64k",                None,  0.0,  1.00, "mp3",     64,    0.0,      30.0, "EXACT_MATCH"),
    ("aac_96k",                None,  0.0,  1.00, "aac",     96,    0.0,      30.0, "EXACT_MATCH"),
    ("noise_snr20",            20.0,  0.0,  1.00, "wav",   1411,    0.0,      30.0, "EXACT_MATCH"),
    ("noise_snr10",            10.0,  0.0,  1.00, "wav",   1411,    0.0,      30.0, "NEAR_MATCH"),
    ("noise_snr5",              5.0,  0.0,  1.00, "wav",   1411,    0.0,      30.0, "NEAR_MATCH"),
    ("voice_overlay",          15.0,  0.0,  1.00, "wav",   1411,    0.0,      30.0, "NEAR_MATCH"),
    ("gain_minus12db",         None,  0.0,  1.00, "wav",   1411,    0.0,      30.0, "EXACT_MATCH"),
    ("eq_lowpass_4k",          None,  0.0,  1.00, "wav",   1411,    0.0,      30.0, "NEAR_MATCH"),
    ("pitch_plus_1",           None,  1.0,  1.00, "wav",   1411,    0.0,      30.0, "NEAR_MATCH"),
    ("pitch_minus_1",          None, -1.0,  1.00, "wav",   1411,    0.0,      30.0, "NEAR_MATCH"),
    ("pitch_plus_2",           None,  2.0,  1.00, "wav",   1411,    0.0,      30.0, "NEAR_MATCH"),
    ("pitch_minus_2",          None, -2.0,  1.00, "wav",   1411,    0.0,      30.0, "NEAR_MATCH"),
    ("tempo_0_90",             None,  0.0,  0.90, "wav",   1411,    0.0,      30.0, "NEAR_MATCH"),
    ("tempo_0_95",             None,  0.0,  0.95, "wav",   1411,    0.0,      30.0, "NEAR_MATCH"),
    ("tempo_1_05",             None,  0.0,  1.05, "wav",   1411,    0.0,      30.0, "NEAR_MATCH"),
    ("tempo_1_10",             None,  0.0,  1.10, "wav",   1411,    0.0,      30.0, "NEAR_MATCH"),
    ("reencode_mp3_chain",     None,  0.0,  1.00, "mp3",    192,    0.0,      30.0, "EXACT_MATCH"),
]


def load_embedding_recording_ids() -> set:
    """Các bản ghi có embedding — ưu tiên vì chỉ chúng mới dùng được cho tầng 2."""
    if not os.path.exists(config.FAISS_ID_MAP_PATH):
        return set()
    with open(config.FAISS_ID_MAP_PATH, encoding="utf-8") as f:
        return set(json.load(f).get("recording_ids", []))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50,
                        help="Số bản ghi IN-DATABASE đưa vào bộ test")
    parser.add_argument("--unknown", type=int, default=25,
                        help="Số truy vấn NGOÀI cơ sở dữ liệu (cho EXP-06)")
    args = parser.parse_args()

    with_embeddings = load_embedding_recording_ids()

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT recording_id, composition_id FROM recordings"
        )).fetchall()

    if not rows:
        print("⚠️ Bảng recordings trống. Chạy init_db.py trước!")
        return 1

    # Ưu tiên bản ghi có embedding, phần còn lại lấy bổ sung
    ranked = sorted(rows, key=lambda r: str(r[0]) not in with_embeddings)
    in_db = ranked[:args.limit]
    # Nhóm "unknown": bản ghi KHÔNG có embedding và sẽ bị loại khỏi reference
    # index khi chạy EXP-06, dùng để đo False Match Rate.
    unknown_pool = [r for r in ranked if str(r[0]) not in with_embeddings]
    unknown = unknown_pool[-args.unknown:] if unknown_pool else []

    queries = []
    for row in in_db:
        for (name, snr, pitch, tempo, codec, bitrate, seg_start, dur, expected) in TRANSFORMATIONS:
            queries.append({
                "query_id": str(uuid.uuid4()),
                "recording_id": str(row[0]),
                "composition_id": str(row[1]) if row[1] else None,
                "transformation": name,
                "snr": snr,
                "pitch_shift": pitch,
                "tempo_factor": tempo,
                "codec": codec,
                "bitrate": bitrate,
                "segment_start": seg_start,
                "duration": dur,
                "expected_match_type": expected,
            })

    # Truy vấn ngoài CSDL: hệ thống PHẢI trả UNKNOWN chứ không được đoán bừa (§2)
    for row in unknown:
        queries.append({
            "query_id": str(uuid.uuid4()),
            "recording_id": str(row[0]),
            "composition_id": str(row[1]) if row[1] else None,
            "transformation": "out_of_database",
            "snr": None,
            "pitch_shift": 0.0,
            "tempo_factor": 1.0,
            "codec": "wav",
            "bitrate": 1411,
            "segment_start": 0.0,
            "duration": 30.0,
            "expected_match_type": "UNKNOWN",
        })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = list(queries[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(queries)

    n_with_emb = sum(1 for r in in_db if str(r[0]) in with_embeddings)
    print(f"✅ Đã tạo {len(queries)} kịch bản test tại: {OUTPUT_CSV}")
    print(f"   - {len(in_db)} bản ghi in-database × {len(TRANSFORMATIONS)} biến đổi "
          f"(trong đó {n_with_emb} bản ghi có embedding)")
    print(f"   - {len(unknown)} truy vấn out-of-database (EXP-06)")
    print("ℹ️  Đây là KẾ HOẠCH truy vấn. Cần audio gốc để dựng file thật.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
