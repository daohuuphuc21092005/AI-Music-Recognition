"""
Sinh embedding MERT cho audio THẬT và nạp vào reference database.

Đây là script thay thế cho `scripts/build_full_mert_index.py` đã bị xoá — bản cũ
lấy `recording_id` từ TÊN FILE, đó chính là nguyên nhân khiến embeddings_master.csv
không join được với bảng recordings.

Cách dùng:

    # Một file, gán vào một recording_id có sẵn
    python scripts/build_embeddings.py --file test.mp3 --recording-id <UUID>

    # Toàn bộ corpus, ánh xạ theo recordings.audio_path trong CSDL
    python scripts/build_embeddings.py --all

    # Giới hạn trong một thư mục
    python scripts/build_embeddings.py --dir D:/music-rights-data/audio/fma

Quy ước cắt đoạn (giữ nguyên như dữ liệu hiện có): cửa sổ 15s, bước nhảy 10s
(chồng lấn 5s). Vector được mean-pooling theo thời gian rồi chuẩn hoá L2.

Sau khi chạy xong, script tự gọi lại việc dựng index để CSV và FAISS luôn khớp.
"""
import argparse
import csv
import os
import subprocess
import sys
import uuid
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import numpy as np
import torch

from backend import config
from backend.services.embedding_service import load_model
from experiments.common import Checkpoint

csv.field_size_limit(10 ** 9)

WINDOW_S = 15.0
HOP_S = 10.0
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac")


def segment_embeddings(audio_path: str):
    """Trả danh sách (segment_start, segment_end, vector 768 chiều đã chuẩn hoá)."""
    processor, model = load_model()
    audio, sr = librosa.load(audio_path, sr=config.MERT_SAMPLE_RATE, mono=True)
    duration = len(audio) / sr

    results = []
    start = 0.0
    while start + WINDOW_S <= duration + 1e-6 or (start == 0.0 and duration > 1.0):
        end = min(start + WINDOW_S, duration)
        chunk = audio[int(start * sr):int(end * sr)]
        if len(chunk) < sr:            # bỏ đoạn ngắn hơn 1 giây
            break

        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            pooled = torch.mean(hidden, dim=1)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vector = normalized.squeeze().numpy()

        results.append((round(start, 2), round(end, 2), vector))
        start += HOP_S
        if end >= duration:
            break
    return results, duration


def load_existing(path: str):
    if not os.path.exists(path):
        return [], ["embedding_id", "recording_id", "segment_start", "segment_end",
                    "model", "model_version", "dimension", "vector", "created_at"]
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def resolve_targets(args) -> list:
    """Trả danh sách (audio_path, recording_id)."""
    if args.file:
        if not args.recording_id:
            print("❌ Cần --recording-id khi dùng --file")
            return []
        return [(args.file, args.recording_id)]

    from sqlalchemy import text

    from backend.database.session import engine

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT recording_id, audio_path FROM recordings WHERE audio_path IS NOT NULL"
        )).fetchall()

    # Đi theo `audio_path` của CSDL chứ không quét thư mục rồi khớp theo TÊN FILE.
    # Corpus thật nằm rải trong nhiều thư mục con của AUDIO_ROOT (fma/, jamendo/),
    # nên quét một thư mục là bỏ sót phần còn lại; và khớp theo tên file thì hai
    # nguồn trùng tên sẽ gán embedding cho nhầm bản ghi — đúng loại lỗi mà script
    # này sinh ra để loại bỏ.
    only_dir = os.path.abspath(args.dir) if args.dir else None

    targets, missing = [], 0
    for rec_id, audio_path in rows:
        path = config.resolve_audio_path(str(audio_path))
        if not path:
            missing += 1
            continue
        if not path.lower().endswith(AUDIO_EXTENSIONS):
            continue
        # --dir trở thành BỘ LỌC: chỉ nhận file nằm trong thư mục đó
        if only_dir and not os.path.abspath(path).startswith(only_dir + os.sep):
            continue
        targets.append((path, str(rec_id)))

    if missing:
        print(f"⏭️  {missing} bản ghi có audio_path nhưng không thấy file "
              f"(AUDIO_ROOT = {config.AUDIO_ROOT})")
    return sorted(targets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Một file audio cụ thể")
    parser.add_argument("--recording-id", help="recording_id (UUID) cho --file")
    parser.add_argument("--dir", help="Chỉ xử lý audio nằm trong thư mục này. "
                                      "Bỏ trống = mọi recording có audio_path.")
    parser.add_argument("--all", action="store_true",
                        help="Mọi recording có audio_path (không giới hạn thư mục)")
    parser.add_argument("--replace", action="store_true",
                        help="Xoá embedding cũ của các recording này trước khi thêm")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="Không tự dựng lại FAISS index sau khi ghi CSV")
    args = parser.parse_args()

    if not args.file and not args.dir and not args.all:
        parser.error("Cần --file, --dir hoặc --all")

    targets = resolve_targets(args)
    if not targets:
        print("Không có file nào để xử lý.")
        return 1

    rows, fieldnames = load_existing(config.EMBEDDINGS_CSV)
    print(f"Embedding hiện có: {len(rows)} dòng / "
          f"{len({r['recording_id'] for r in rows})} bản ghi")

    target_ids = {rec_id for _, rec_id in targets}
    if args.replace:
        before = len(rows)
        rows = [r for r in rows if r["recording_id"] not in target_ids]
        print(f"Đã xoá {before - len(rows)} dòng embedding cũ của "
              f"{len(target_ids)} bản ghi")

    now = datetime.now().isoformat()

    # Checkpoint: suy luận MERT cho vài nghìn bản ghi mất nhiều GIỜ, và CSV chỉ
    # được ghi ở cuối — bị ngắt giữa chừng (máy hết RAM chẳng hạn) là mất sạch,
    # nên corpus lớn sẽ không bao giờ dựng xong. Khoá là đường dẫn audio.
    # keys_only: mỗi bản ghi chứa vector 768 chiều dạng text (~33 KB), nên tới
    # bản ghi thứ vài nghìn thì việc nạp lại cả checkpoint đã ngốn vài trăm MB —
    # đủ để tiến trình bị giết, và mỗi lượt chạy lại chỉ tiến được vài chục file.
    # Ở đây chỉ cần biết file nào đã xong; nội dung đọc sau, theo dòng.
    checkpoint = Checkpoint("build_embeddings", {
        "targets": len(targets),
        "window_s": WINDOW_S,
        "hop_s": HOP_S,
        "mert_model_version": config.MERT_MODEL_VERSION,
        "embedding_dim": config.EMBEDDING_DIM,
        "replace": bool(args.replace),
    }, keys_only=True)

    added = 0
    for audio_path, rec_id in targets:
        if checkpoint.has(audio_path):
            continue
        if not os.path.exists(audio_path):
            print(f"⏭️  Không tìm thấy {audio_path}")
            continue
        segments, duration = segment_embeddings(audio_path)
        checkpoint.add(audio_path, {
            "recording_id": rec_id,
            "duration": round(float(duration), 2),
            "segments": [
                {"start": start, "end": end,
                 "vector": ",".join(map(str, vector.tolist()))}
                for start, end, vector in segments
            ],
        })
        print(f"✅ {os.path.basename(audio_path)}: {len(segments)} segment "
              f"({duration:.1f}s) -> {rec_id}")

    for entry in checkpoint.iter_records():
        for segment in entry["segments"]:
            rows.append({
                "embedding_id": str(uuid.uuid4()),
                "recording_id": entry["recording_id"],
                "segment_start": segment["start"],
                "segment_end": segment["end"],
                "model": "MERT",
                "model_version": config.MERT_MODEL_VERSION,
                "dimension": config.EMBEDDING_DIM,
                "vector": segment["vector"],
                "created_at": now,
            })
            added += 1

    with open(config.EMBEDDINGS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    checkpoint.close(remove=True)
    print(f"\n💾 Đã ghi {config.EMBEDDINGS_CSV}: {len(rows)} dòng (+{added})")

    if not args.no_rebuild:
        print("\n▶️  Dựng lại FAISS index...")
        subprocess.run([sys.executable,
                        os.path.join(config.BASE_DIR, "scripts", "rebuild_faiss_index.py")],
                       check=True)
        print("ℹ️  Nhớ chạy `python init_db.py` để nạp embedding mới vào PostgreSQL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
