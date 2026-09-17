"""
EXP-07 bổ sung — vì sao tầng Cover trượt tempo chậm và đoạn cắt ngắn.

Server lấy 30 giây đầu của truy vấn rồi co giãn về `DESCRIPTOR_FRAMES` khung, nên
chỉ bất biến nhịp độ khi truy vấn chứa TRỌN đoạn nội dung của reference (cũng là
30 giây đầu). Bản chậm 0.90 dài 33 giây bị cắt ở giây 30, mất 10% nội dung cuối.

Phép kiểm chỉ đổi đúng một yếu tố: 30 giây đầu (như server) so với trọn truy vấn,
trên toàn chỉ mục cover. Với đoạn cắt 10/15 giây, so thêm với reference cắt đúng
cùng khoảng để tách "lệch trục thời gian" khỏi "khác nội dung".

Chỉ đo trên truy vấn CÓ trong CSDL — không nói gì về FMR. Muốn đổi cách đọc truy
vấn ở server thì phải chạy lại experiments/exp07_cover/run.py.

    python experiments/exp07_cover/query_span_check.py            # 100 bài nguồn đầu
    python experiments/exp07_cover/query_span_check.py --sources 5
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend import config  # noqa: E402
from backend.services import cover_service  # noqa: E402

TRANSFORMATIONS = ["tempo_0_90", "tempo_0_95", "tempo_1_05", "tempo_1_10", "crop_10s", "crop_15s"]
MANIFEST = config.BASE_DIR / "data" / "test_queries" / "manifest.csv"


def rank_and_score(index, descriptor: np.ndarray, column: int) -> tuple:
    scores = (cover_service.transpositions(descriptor) @ index.matrix.T).max(axis=0)
    return int((scores > scores[column]).sum()) + 1, float(scores[column])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=100)
    args = parser.parse_args()

    index = cover_service.get_default_cover_index()
    if index is None or index.matrix is None:
        print("Chưa có chỉ mục cover: chạy scripts/build_cover_index.py trước.")
        return 1
    column_of = {str(rec): i for i, rec in enumerate(index.id_map)}

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    rows = {(r["source_recording_id"], r["transformation"]): r for r in manifest}
    sources = list(dict.fromkeys(r["source_recording_id"] for r in manifest))[:args.sources]

    started = time.perf_counter()
    print(f"{len(sources)} bài nguồn, chỉ mục {len(index.id_map)} bài, τCover {config.COVER_THRESHOLD}")
    for name in TRANSFORMATIONS:
        stats = {"30 s đầu": [], "trọn truy vấn": []}
        same_span = []
        for source in sources:
            row = rows.get((source, name))
            if row is None or source not in column_of:
                continue
            path = config.resolve_audio_path(row["path"]) or row["path"]
            for mode, duration in (("30 s đầu", 30.0), ("trọn truy vấn", None)):
                descriptor = cover_service.descriptor_from_file(path, duration=duration)
                stats[mode].append(rank_and_score(index, descriptor, column_of[source]))
            if name.startswith("crop"):
                source_path = config.resolve_audio_path(row["source_file"]) or row["source_file"]
                query = cover_service.descriptor_from_file(path, duration=None)
                reference = cover_service.descriptor_from_file(
                    source_path, offset=float(row["crop_start_s"]),
                    duration=float(row["duration_s"]))
                same_span.append(cover_service.cover_similarity(query, reference)[0])

        parts = [f"{name:11s} n={len(stats['trọn truy vấn'])}"]
        for mode, values in stats.items():
            ranks = np.array([v[0] for v in values])
            scores = np.array([v[1] for v in values])
            accepted = (ranks == 1) & (scores >= config.COVER_THRESHOLD)
            parts.append(f"{mode}: @1 {np.mean(ranks == 1):.2f} · qua τ {np.mean(accepted):.2f} "
                         f"· điểm TB {scores.mean():.3f}")
        print(" | ".join(parts), flush=True)
        if same_span:
            print(f"{'':11s} reference cắt cùng khoảng: điểm TB {np.mean(same_span):.3f}, "
                  f"thấp nhất {np.min(same_span):.3f}")
    print(f"Xong sau {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
