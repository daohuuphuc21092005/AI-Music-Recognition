"""
EXP-07 bổ sung — tầng Cover cắt truy vấn thế nào thì bắt được đổi tốc độ.

Reference là 30 giây đầu ở nhịp gốc, co giãn về `DESCRIPTOR_FRAMES` khung, nên chỉ
bất biến nhịp độ khi đoạn truy vấn chứa TRỌN đúng phần nội dung đó. Bản chậm 0.90
chứa nó trong 33,3 giây đầu; cắt cố định 30 giây thì mất 10% nội dung cuối.

Ba cách cắt, cùng toàn bộ chỉ mục cover:
  - "30 s đầu"     : cách server dùng TRƯỚC 2026-09-17
  - "trọn truy vấn": chẩn đoán — chỉ đúng khi file truy vấn chính là đoạn 30 s đã
                     đổi nhịp (như tập kiểm thử), không dùng được cho file tải lên dài
  - "production"   : cắt theo từng hệ số `config.COVER_TEMPO_FACTORS`, lấy max —
                     cách server dùng hiện nay (cover_service.query_descriptors)

Với đoạn cắt 10/15 giây, so thêm với reference cắt đúng cùng khoảng để tách "lệch
trục thời gian" khỏi "khác nội dung".

`--off-grid`: các hệ số production (0.90 … 1.10) trùng đúng các mức tempo của tập
kiểm thử, nên số ở trên có thể lạc quan. Phần này tự đổi nhịp bài nguồn sang 0.92 và
1.07 (không có trong bảng hệ số) bằng librosa.effects.time_stretch để đo phần tổng quát.

Chỉ đo trên truy vấn CÓ trong CSDL — không nói gì về FMR; FMR đo ở run.py.

    python experiments/exp07_cover/query_span_check.py            # 100 bài nguồn đầu
    python experiments/exp07_cover/query_span_check.py --sources 5 --off-grid
"""
import argparse
import csv
import os
import sys
import time

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend import config  # noqa: E402
from backend.services import cover_service  # noqa: E402

TRANSFORMATIONS = ["tempo_0_90", "tempo_0_95", "tempo_1_05", "tempo_1_10", "crop_10s", "crop_15s"]
OFF_GRID_TEMPOS = (0.92, 1.07)
MANIFEST = config.BASE_DIR / "data" / "test_queries" / "manifest.csv"


def rank_and_score(index, descriptors: np.ndarray, column: int) -> tuple:
    scores, _, _ = cover_service.best_scores(descriptors, index.matrix)
    return int((scores > scores[column]).sum()) + 1, float(scores[column])


def summarize(label: str, stats: dict) -> str:
    parts = [label]
    for mode, values in stats.items():
        ranks = np.array([v[0] for v in values])
        scores = np.array([v[1] for v in values])
        accepted = (ranks == 1) & (scores >= config.COVER_THRESHOLD)
        parts.append(f"{mode}: @1 {np.mean(ranks == 1):.2f} · qua τ {np.mean(accepted):.2f} "
                     f"· điểm TB {scores.mean():.3f}")
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=100)
    parser.add_argument("--off-grid", action="store_true",
                        help="Đo thêm nhịp độ nằm ngoài bảng hệ số production")
    args = parser.parse_args()

    index = cover_service.get_default_cover_index()
    if index is None or index.matrix is None:
        print("Chưa có chỉ mục cover: chạy scripts/build_cover_index.py trước.")
        return 1
    column_of = {str(rec): i for i, rec in enumerate(index.id_map)}

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    rows = {(r["source_recording_id"], r["transformation"]): r for r in manifest}
    sources = [s for s in dict.fromkeys(r["source_recording_id"] for r in manifest)
               if s in column_of][:args.sources]

    started = time.perf_counter()
    print(f"{len(sources)} bài nguồn, chỉ mục {len(index.id_map)} bài, τCover "
          f"{config.COVER_THRESHOLD}, hệ số production {config.COVER_TEMPO_FACTORS}")
    for name in TRANSFORMATIONS:
        stats = {"30 s đầu": [], "trọn truy vấn": [], "production": []}
        same_span = []
        for source in sources:
            row = rows.get((source, name))
            if row is None:
                continue
            path = config.resolve_audio_path(row["path"]) or row["path"]
            column = column_of[source]
            stats["30 s đầu"].append(rank_and_score(
                index, cover_service.descriptor_from_file(path, duration=30.0), column))
            stats["trọn truy vấn"].append(rank_and_score(
                index, cover_service.descriptor_from_file(path, duration=None), column))
            stats["production"].append(rank_and_score(
                index, cover_service.query_descriptors_from_file(path)[0], column))
            if name.startswith("crop"):
                source_path = config.resolve_audio_path(row["source_file"]) or row["source_file"]
                query = cover_service.descriptor_from_file(path, duration=None)
                reference = cover_service.descriptor_from_file(
                    source_path, offset=float(row["crop_start_s"]),
                    duration=float(row["duration_s"]))
                same_span.append(cover_service.cover_similarity(query, reference)[0])
        print(summarize(f"{name:11s} n={len(stats['production'])}", stats), flush=True)
        if same_span:
            print(f"{'':11s} reference cắt cùng khoảng: điểm TB {np.mean(same_span):.3f}, "
                  f"thấp nhất {np.min(same_span):.3f}")

    if args.off_grid:
        print(f"\nNhịp độ ngoài bảng hệ số {OFF_GRID_TEMPOS} (librosa.effects.time_stretch "
              f"trên 30 s đầu bài nguồn):")
        for tempo in OFF_GRID_TEMPOS:
            stats = {"30 s đầu": [], "production": []}
            for source in sources:
                row = rows.get((source, "original_crop30s"))
                if row is None:
                    continue
                source_path = config.resolve_audio_path(row["source_file"]) or row["source_file"]
                audio, sr = librosa.load(source_path, sr=cover_service.CHROMA_SR, mono=True,
                                         duration=30.0)
                stretched = librosa.effects.time_stretch(audio, rate=tempo)
                column = column_of[source]
                stats["30 s đầu"].append(rank_and_score(
                    index, cover_service.build_descriptor(stretched[:int(30.0 * sr)], sr), column))
                stats["production"].append(rank_and_score(
                    index, cover_service.query_descriptors(stretched, sr)[0], column))
            print(summarize(f"tempo {tempo:<5} n={len(stats['production'])}", stats), flush=True)

    print(f"Xong sau {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
