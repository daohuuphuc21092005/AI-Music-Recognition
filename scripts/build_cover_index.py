"""
Dựng chỉ mục Cover (CQT chroma, 12 x 64 = 768 chiều) cho MỌI bản ghi có audio thật.

Dùng:
    python scripts/build_cover_index.py

Mỗi bản ghi một descriptor lấy từ 30 giây đầu (offset 0) — đúng đoạn mà
`cover_service.identify_cover` trích từ file truy vấn. Reference và truy vấn phải
cùng quy ước thì điểm mới so được với τCover.

Bản cũ ghép bản ghi theo TÊN FILE (chỉ nhận `fma/...`) và ưu tiên các file
`*__original_crop30s.wav` trong data/test_queries/, nên chỉ mục chỉ có ~100 bài —
đúng các bài nguồn của tập truy vấn. EXP-07 khi đó hiệu chỉnh τCover trên một
"CSDL" mà truy vấn nào cũng có sẵn đáp án. Giờ bản ghi đi theo `audio_path` như
build_embeddings.py, và chạy tiếp được sau khi bị ngắt nhờ checkpoint.
"""
import base64
import csv
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from backend import config
from backend.services import cover_service
from experiments.common import Checkpoint

csv.field_size_limit(10 ** 9)

OFFSET_S = 0.0
DURATION_S = 30.0   # tham số mặc định của cover_service.identify_cover
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac")


def resolve_targets() -> list:
    """(đường dẫn audio tuyệt đối, recording_id) cho mọi bản ghi có file trên đĩa."""
    targets, seen, missing = [], set(), 0
    with open(config.METADATA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rec_id, audio_path = row.get("recording_id", ""), row.get("audio_path", "")
            if not rec_id or not audio_path or rec_id in seen:
                continue
            path = config.resolve_audio_path(audio_path)
            if not path:
                missing += 1
                continue
            if not path.lower().endswith(AUDIO_EXTENSIONS):
                continue
            seen.add(rec_id)
            targets.append((path, rec_id))

    if missing:
        print(f"⏭️  {missing} bản ghi có audio_path nhưng không thấy file "
              f"(AUDIO_ROOT = {config.AUDIO_ROOT})")
    return sorted(targets)


def main() -> int:
    t0 = time.time()
    print("=== DỰNG CHỈ MỤC COVER REFERENCE (CQT CHROMA + OTI) ===")

    if not os.path.exists(config.METADATA_CSV):
        print(f"❌ Không tìm thấy metadata: {config.METADATA_CSV}")
        return 1

    targets = resolve_targets()
    print(f"Tìm thấy {len(targets)} bản ghi có audio.")
    if not targets:
        print("⚠️ Không tìm thấy file audio nào để tạo chỉ mục.")
        return 1

    # Descriptor lưu dạng base64 của float32: đúng từng bit như lúc tính, và chỉ
    # ~4 KB mỗi bản ghi thay vì ~14 KB nếu ghi 768 số thực dạng text.
    checkpoint = Checkpoint("build_cover_index", {
        "targets": len(targets),
        "offset_s": OFFSET_S,
        "duration_s": DURATION_S,
        "chroma_sr": cover_service.CHROMA_SR,
        "descriptor_frames": cover_service.DESCRIPTOR_FRAMES,
        "normalization": "mean_centered_frames_l2",
    }, keys_only=True)

    processed = 0
    for position, (audio_path, rec_id) in enumerate(targets, 1):
        if checkpoint.has(audio_path):
            continue
        try:
            descriptor = cover_service.descriptor_from_file(
                audio_path, offset=OFFSET_S, duration=DURATION_S)
            record = {"recording_id": rec_id,
                      "descriptor": base64.b64encode(
                          descriptor.astype("<f4").tobytes()).decode("ascii")}
        except Exception as e:
            # Ghi cả lỗi vào checkpoint để lần chạy sau không thử mãi một file hỏng
            record = {"recording_id": rec_id, "error": f"{type(e).__name__}: {e}"}
        checkpoint.add(audio_path, record)
        processed += 1
        if processed % 500 == 0:
            print(f"  [{position:>6}/{len(targets)}] {time.time() - t0:.0f}s", flush=True)

    descriptors, id_map, failures = [], [], []
    for record in checkpoint.iter_records():
        if "error" in record:
            failures.append(record)
            continue
        descriptors.append(np.frombuffer(base64.b64decode(record["descriptor"]), dtype="<f4"))
        id_map.append(record["recording_id"])

    if failures:
        print(f"⚠️  {len(failures)} file không trích được descriptor, ví dụ:")
        for record in failures[:5]:
            print(f"    {record['recording_id']}: {record['error']}")
    if not descriptors:
        print("❌ Không trích xuất được descriptor nào.")
        checkpoint.close()
        return 1

    matrix = np.vstack(descriptors).astype("float32")
    os.makedirs(os.path.dirname(config.COVER_INDEX_PATH), exist_ok=True)
    np.save(config.COVER_INDEX_PATH, matrix)
    with open(config.COVER_ID_MAP_PATH, "w", encoding="utf-8") as f:
        import json
        json.dump(id_map, f, indent=2)

    loaded = cover_service.load_cover_index(config.COVER_INDEX_PATH, config.COVER_ID_MAP_PATH)
    if not loaded or loaded.n_items != len(id_map):
        print("❌ Lỗi kiểm tra nạp lại chỉ mục.")
        checkpoint.close()
        return 1
    checkpoint.close(remove=True)

    print("\n✅ ĐÃ TẠO CHỈ MỤC COVER:")
    print(f"  Ma trận descriptors: {config.COVER_INDEX_PATH} (shape {matrix.shape})")
    print(f"  Bản đồ ID          : {config.COVER_ID_MAP_PATH} ({len(id_map)} bản ghi)")
    print(f"  Thời gian thực hiện: {time.time() - t0:.0f}s")
    print("ℹ️  Chạy lại EXP-07 để hiệu chỉnh τCover trên chỉ mục mới.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
