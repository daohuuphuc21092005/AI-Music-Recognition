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

    # Corpus lớn: chia thành nhiều lượt ngắn, mỗi lượt thoát hẳn để trả RAM về HĐH
    # (mã thoát 10 = còn việc). Lượt cuối tự ghi CSV rồi dựng lại FAISS.
    do { python scripts/build_embeddings.py --all --limit 200 --no-rebuild }
      while ($LASTEXITCODE -eq 10)
    python scripts/rebuild_faiss_index.py

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


FIELDNAMES = ["embedding_id", "recording_id", "segment_start", "segment_end",
              "model", "model_version", "dimension", "vector", "created_at"]

# Dừng sớm theo --limit mà vẫn còn việc: mã riêng để vòng lặp gọi bên ngoài biết
# là phải chạy tiếp, chứ không nhầm với "đã xong".
EXIT_MORE_WORK = 10


def existing_fieldnames(path: str) -> list:
    if not os.path.exists(path):
        return FIELDNAMES
    with open(path, newline="", encoding="utf-8") as f:
        return csv.DictReader(f).fieldnames or FIELDNAMES


def iter_existing(path: str, skip_ids: set):
    """
    Đọc từng dòng embedding cũ, bỏ các bản ghi sắp được ghi lại.

    Đọc theo dòng chứ không nạp cả file: với 24.375 bài thì CSV là ~48.000 dòng,
    mỗi dòng chứa vector 768 chiều dạng text (~16 KB) — nạp hết vào một list là
    ~800 MB RAM, đúng lúc tiến trình đã phình to nhất.
    """
    if not os.path.exists(path):
        return
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["recording_id"] not in skip_ids:
                yield row


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
    parser.add_argument("--limit", type=int, metavar="N",
                        help="Chỉ xử lý N bản ghi CHƯA có trong checkpoint rồi thoát "
                             f"(mã {EXIT_MORE_WORK} nếu vẫn còn việc). Tiến trình MERT "
                             "phình dần theo số file đã xử lý (đo được ~3 MB/file: 1,2 GB "
                             "lúc đầu -> 2,0 GB sau 475 file), nên chạy một lượt hàng chục "
                             "nghìn file thì bị hệ điều hành giết vì hết RAM. Chia thành "
                             "nhiều lượt ngắn, mỗi lượt thoát hẳn để trả bộ nhớ về HĐH.")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="Không tự dựng lại FAISS index sau khi ghi CSV")
    args = parser.parse_args()

    if not args.file and not args.dir and not args.all:
        parser.error("Cần --file, --dir hoặc --all")

    targets = resolve_targets(args)
    if not targets:
        print("Không có file nào để xử lý.")
        return 1

    fieldnames = existing_fieldnames(config.EMBEDDINGS_CSV)
    skip_ids = {rec_id for _, rec_id in targets} if args.replace else set()
    if skip_ids:
        print(f"--replace: bỏ embedding cũ của {len(skip_ids)} bản ghi khi ghi lại CSV")

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

    processed = 0
    for audio_path, rec_id in targets:
        if checkpoint.has(audio_path):
            continue
        if args.limit and processed >= args.limit:
            break
        if not os.path.exists(audio_path):
            print(f"⏭️  Không tìm thấy {audio_path}")
            continue
        processed += 1
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

    remaining = sum(1 for path, _ in targets if not checkpoint.has(path))
    if remaining:
        checkpoint.close()
        done = len(targets) - remaining
        print(f"\n⏸️  Dừng theo --limit: đã xong {done}/{len(targets)} bản ghi, "
              f"còn {remaining}. Chạy lại đúng lệnh này để tiếp tục.")
        return EXIT_MORE_WORK

    # Ghi thẳng ra file tạm rồi thay thế: vừa không giữ 48.000 dòng trong RAM,
    # vừa không để lại CSV cụt nếu tiến trình bị giết giữa chừng.
    tmp_path = config.EMBEDDINGS_CSV + ".tmp"
    kept = added = 0
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in iter_existing(config.EMBEDDINGS_CSV, skip_ids):
            writer.writerow(row)
            kept += 1
        for entry in checkpoint.iter_records():
            for segment in entry["segments"]:
                writer.writerow({
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
    os.replace(tmp_path, config.EMBEDDINGS_CSV)

    checkpoint.close(remove=True)
    print(f"\n💾 Đã ghi {config.EMBEDDINGS_CSV}: {kept + added} dòng "
          f"({kept} giữ lại + {added} mới)")

    if not args.no_rebuild:
        print("\n▶️  Dựng lại FAISS index...")
        subprocess.run([sys.executable,
                        os.path.join(config.BASE_DIR, "scripts", "rebuild_faiss_index.py")],
                       check=True)
        print("ℹ️  Nhớ chạy `python init_db.py` để nạp embedding mới vào PostgreSQL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
