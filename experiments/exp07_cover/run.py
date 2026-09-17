"""
EXP-07 — Cover/Version Identification: baseline CQT/chroma + OTI (§15).

Vì sao có thí nghiệm này, và vì sao nó có hình dạng như hiện tại — hai chuyện
khác nhau, cần nói rõ cả hai:

**Vì sao có:** EXP-05 đo được dịch cao độ là điểm mù **0%** của cả Chromaprint
lẫn MERT. Ở mức cửa sổ, EXP-03 còn cho thấy MERT chỉ đạt 0–12,5% trên nhóm pitch.
Cả hai đều mã hoá cao độ TUYỆT ĐỐI. Chroma thì gập phổ về 12 bậc, nên dịch cao độ
chỉ là một phép xoay vòng — đó là chỗ `cover_service` được thiết kế để thắng.

**Vì sao KHÔNG phải thí nghiệm cover thật:** §8 yêu cầu tập cover
(SecondHandSongs subset, 100–200 composition × 2–5 version). Repo hiện **không có
bản thu cover nào**. Không thể đo Recall@K trên cover khi không có cover — nên
thí nghiệm này đo đúng thứ đo được: **tính bất biến với dịch cao độ và đổi tốc
độ**, tức là cơ chế cốt lõi mà một hệ cover dựa vào. Đây là điều kiện CẦN, không
phải điều kiện đủ. Chưa có số này thì không đáng bàn tới CoverHunter/CSI.

Giao thức: dùng nguyên bộ cửa sổ của EXP-03 (`experiments/common.py`), nên cột
MERT trong bảng so sánh là số ĐO TRÊN CÙNG cửa sổ, không phải số mượn từ thí
nghiệm khác điều kiện. Script tự đối chiếu tham số giao thức và từ chối chạy nếu
hai bên lệch nhau.

Metrics (§15): Recall@1/@5, MRR, mAP; kèm sweep để hiệu chỉnh τCover (§4).

Chạy trước:
    python scripts/augment_audio.py --from-db
    python scripts/build_cover_index.py      # cho phần chấm theo điều kiện server
    python experiments/exp03_pooling/run.py  # tuỳ chọn: cột so sánh MERT
"""
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import librosa
import numpy as np

from backend import config
from backend.services.cover_service import (
    CHROMA_SR,
    build_descriptor,
    descriptor_from_file,
    load_cover_index,
    search,
    transpositions,
)
from experiments.common import (
    HeldOutProtocol,
    OVERLAP_MIN,
    QUERY_HOP_S,
    REF_HOP_S,
    RESULTS_DIR,
    WINDOW_S,
    load_query_manifest,
    overlap_ratio,
    print_table,
    protocol_params,
    query_crop_start,
    query_pitch_steps,
    query_tempo_factor,
    resolve_source_audio,
    save_result,
    windows,
)

EXPERIMENT_ID = "exp07_cover"
K_VALUES = (1, 5)
# Từ 0.50: descriptor trừ trung bình cho điểm thấp hơn hẳn bản cũ (bài khác nhau
# còn ~0.87 thay vì ~0.97), dải cũ 0.80–1.00 có thể bỏ sót ngưỡng đúng.
THRESHOLD_SWEEP = [round(0.50 + 0.01 * i, 2) for i in range(51)]  # 0.50 .. 1.00
SWEEP_PRINT = (0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0)
MAX_FALSE_MATCH = 0.05       # §16: False Match Rate <= 5% (ràng buộc CỨNG)
PREFERRED_FALSE_MATCH = 0.005  # mục tiêu chặt hơn, xem pick_threshold()

# Nhóm phép biến đổi, để đọc bảng theo bản chất tác động chứ không theo tên file
FAMILIES = {
    "Dịch cao độ": ["pitch_plus_1", "pitch_minus_1", "pitch_plus_2", "pitch_minus_2"],
    "Đổi tốc độ": ["tempo_0_90", "tempo_0_95", "tempo_1_05", "tempo_1_10"],
    "Cắt đoạn": ["original_crop30s", "crop_15s", "crop_10s"],
    "Nén codec": ["mp3_128k", "mp3_64k", "aac_96k"],
    "Nhiễu": ["noise_snr20", "noise_snr10", "noise_snr5"],
    "Biên độ / EQ": ["gain_minus12db", "eq_lowpass_4k"],
}

# Mẫu âm không có hoà âm: tầng Cover không được nhận chúng thành bài nào.
NOISE_COLORS = {"trang": 0.0, "hong": 0.5, "nau": 1.0}  # biên độ phổ ∝ 1 / f^alpha
NOISE_SEEDS = range(10)

# Server trích 30 giây đầu của file truy vấn (tham số mặc định của identify_cover)
RUNTIME_DURATION_S = 30.0


def noise_probe_descriptors(seconds: float = WINDOW_S) -> list:
    """Descriptor của nhiễu trắng/hồng/nâu dài `seconds` giây — sinh tổng hợp, lặp lại được."""
    length = int(seconds * CHROMA_SR)
    freqs = np.fft.rfftfreq(length, 1.0 / CHROMA_SR)
    freqs[0] = freqs[1]
    probes = []
    for color, alpha in NOISE_COLORS.items():
        for seed in NOISE_SEEDS:
            spectrum = np.fft.rfft(np.random.RandomState(seed).normal(0, 1, length))
            signal = np.fft.irfft(spectrum / freqs ** alpha, n=length)
            signal = (0.5 * signal / np.max(np.abs(signal))).astype(np.float32)
            probes.append((f"{color}_{seed}", build_descriptor(signal, CHROMA_SR)))
    return probes


def describe_file(path: str, hop: float, timing: list):
    """Trả danh sách (start, end, descriptor) cho một file."""
    audio, sr = librosa.load(path, sr=CHROMA_SR, mono=True)
    duration = len(audio) / sr
    out = []
    for start, end in windows(duration, hop):
        chunk = audio[int(start * sr):int(end * sr)]
        if len(chunk) < sr:
            continue
        t0 = time.perf_counter()
        descriptor = build_descriptor(chunk, sr)
        timing.append((time.perf_counter() - t0) * 1000)
        out.append((start, end, descriptor))
    return out, duration


def is_relevant(ref: dict, query: dict) -> bool:
    """Cửa sổ reference có phải nhãn đúng của truy vấn này không."""
    return (ref["source"] == query["source"]
            and overlap_ratio(ref["start"], ref["end"],
                              query["orig_start"], query["orig_end"]) >= OVERLAP_MIN)


def threshold_sweep(top1_records: list, held_out_scores: list, noise_scores: list) -> list:
    """
    Precision/Recall/F1 khi chấp nhận Top-1 có điểm >= τ, kèm tỉ lệ nhận nhầm.

    `false_match_rate`: tỉ lệ truy vấn mà điểm cao nhất SAU KHI bỏ hẳn bài nguồn
    vẫn >= τ — tức một bài ngoài CSDL sẽ bị gán nhầm (§16 yêu cầu <= 5%).
    """
    n = len(top1_records)
    sweep = []
    for threshold in THRESHOLD_SWEEP:
        accepted = [ok for score, ok in top1_records if score >= threshold]
        correct = sum(accepted)
        precision = correct / len(accepted) if accepted else 0.0
        recall = correct / n if n else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        false_match = sum(1 for score in held_out_scores if score >= threshold)
        noise_match = sum(1 for score in noise_scores if score >= threshold)
        sweep.append({"threshold": threshold, "precision": round(precision, 4),
                      "recall": round(recall, 4), "f1": round(f1, 4),
                      "accepted": len(accepted),
                      "false_match_rate": round(false_match / max(len(held_out_scores), 1), 4),
                      "noise_match_rate": round(noise_match / max(len(noise_scores), 1), 4)})
    return sweep


def pick_threshold(sweep: list) -> tuple:
    """
    (dòng sweep được chọn, có đạt mục tiêu an toàn không).

    Ràng buộc trước, F1 sau — chọn theo F1 rồi mới xét FMR là lặp lại sai lầm của
    τMERT = 0.90. Thứ tự: không nhận mẫu nhiễu nào, FMR ≤ 0.005, rồi mới F1 cao
    nhất; không ngưỡng nào đạt 0.005 thì nới về trần 5% của §16.

    Vì sao siết tới 0.005 chứ không dừng ở §16 (giống hệt lý do của τFP trong
    scripts/apply_calibrated_thresholds.py): trên chỉ mục 24.375 bài, trần 5% chọn
    τ = 0.82 (FMR 4,5%), còn τ = 0.90 cho FMR 0,4% mà recall chỉ giảm 0.76 -> 0.68.
    Cover là tầng CUỐI của cascade: nhận nhầm ở đây đi thẳng ra một kết luận về
    quyền, không còn tầng nào phía dưới đỡ, nên 1/20 bài lạ bị gán bừa là quá đắt.
    """
    for bound in (PREFERRED_FALSE_MATCH, MAX_FALSE_MATCH):
        safe = [r for r in sweep
                if r["false_match_rate"] <= bound and r["noise_match_rate"] == 0.0]
        if safe:
            return max(safe, key=lambda r: (r["f1"], -r["threshold"])), True
    return max(sweep, key=lambda r: (r["f1"], r["threshold"])), False


def describe(values: list, percentile: int = None) -> dict:
    if not values:
        return {"count": 0}
    out = {"count": len(values), "mean": round(float(np.mean(values)), 4),
           "max": round(float(np.max(values)), 4)}
    if percentile is not None:
        out[f"p{percentile:02d}"] = round(float(np.percentile(values, percentile)), 4)
    return out


def runtime_protocol(manifest: list):
    """
    Chấm τCover đúng điều kiện server: 30 giây đầu của file truy vấn, tìm trên
    TOÀN BỘ chỉ mục cover (cover_service.identify_cover).

    Phần cấp cửa sổ chỉ có reference của các bài nguồn. Điểm cao nhất mà một bài
    ngoài CSDL đạt được tăng theo số ứng viên, nên ngưỡng chọn ở đó quá lạc quan
    khi chỉ mục thật có hàng chục nghìn bài.
    """
    index = load_cover_index()
    if index is None or not index.n_items:
        print("\nChưa có chỉ mục cover -> bỏ phần chấm theo điều kiện server. "
              "Chạy: python scripts/build_cover_index.py")
        return None

    id_map = [str(rec) for rec in index.id_map]
    columns_of = {}
    for column, rec in enumerate(id_map):
        columns_of.setdefault(rec, []).append(column)
    # Held-out loại bản trùng, bản gần trùng và bài bị trộn chồng (audio_overlay)
    protocol = HeldOutProtocol(manifest)

    print(f"\nChấm theo điều kiện server: {len(manifest)} truy vấn x chỉ mục "
          f"{len(columns_of)} bài...")
    top1_records, held_out, timing, per_transformation = [], [], [], {}
    skipped = 0
    for row in manifest:
        source = row["source_recording_id"]
        same_audio = protocol.exact_class(source)
        source_columns = [c for rec in same_audio for c in columns_of.get(rec, [])]
        excluded = [c for rec in protocol.exclusions(row) for c in columns_of.get(rec, [])]
        path = row["path"] if os.path.isabs(row["path"]) else os.path.join(config.BASE_DIR, row["path"])
        # Bài nguồn không nằm trong chỉ mục thì không có đáp án đúng để chấm
        if not source_columns or not os.path.exists(path):
            skipped += 1
            continue
        try:
            t0 = time.perf_counter()
            descriptor = descriptor_from_file(path, offset=0.0, duration=RUNTIME_DURATION_S)
            best = (transpositions(descriptor) @ index.matrix.T).max(axis=0)
            timing.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            print(f"  bỏ qua {row['path']}: {type(e).__name__}: {e}")
            skipped += 1
            continue

        top = int(best.argmax())
        score = float(best[top])
        correct = id_map[top] in same_audio
        best[excluded] = -np.inf
        held_out.append(float(best.max()) if len(excluded) < len(best) else 0.0)
        top1_records.append((score, correct))

        bucket = per_transformation.setdefault(row["transformation"],
                                               {"n": 0, "hit@1": 0, "scores": []})
        bucket["n"] += 1
        bucket["hit@1"] += int(correct)
        bucket["scores"].append(score)

    if not top1_records:
        print("  không chấm được truy vấn nào (bài nguồn không có trong chỉ mục?)")
        return None

    noise_scores = [float((transpositions(descriptor) @ index.matrix.T).max())
                    for _, descriptor in noise_probe_descriptors(RUNTIME_DURATION_S)]
    sweep = threshold_sweep(top1_records, held_out, noise_scores)
    best_row, safe = pick_threshold(sweep)
    return {
        "held_out_exclusions": protocol.describe(),
        "gallery_recordings": len(columns_of),
        "queries": len(top1_records),
        "skipped_queries": skipped,
        "accuracy@1": round(sum(ok for _, ok in top1_records) / len(top1_records), 4),
        "threshold_sweep": sweep,
        "recommended_tau_cover": best_row,
        "tau_cover_meets_fmr_target": safe,
        "score_distribution": {
            "top1_correct": describe([s for s, ok in top1_records if ok], 5),
            "top1_wrong": describe([s for s, ok in top1_records if not ok], 95),
            "held_out_best": describe(held_out, 95),
            "noise_best": describe(noise_scores),
        },
        "latency_ms": {"query_mean": round(float(np.mean(timing)), 1),
                       "query_p95": round(float(np.percentile(timing, 95)), 1)},
        "per_transformation": {
            t: {"n": b["n"], "accuracy@1": round(b["hit@1"] / b["n"], 4),
                "mean_top1_score": round(float(np.mean(b["scores"])), 4)}
            for t, b in sorted(per_transformation.items())
        },
    }


def print_runtime(runtime: dict) -> None:
    print_table(
        f"EXP-07 — Sweep τCover theo điều kiện server ({runtime['queries']} truy vấn, "
        f"chỉ mục {runtime['gallery_recordings']} bài)",
        [[r["threshold"], r["precision"], r["recall"], r["f1"], r["accepted"],
          f"{r['false_match_rate'] * 100:.1f}%", f"{r['noise_match_rate'] * 100:.0f}%"]
         for r in runtime["threshold_sweep"] if r["threshold"] in SWEEP_PRINT],
        ["τCover", "Precision", "Recall", "F1", "Chấp nhận", "False Match", "Nhiễu"],
    )
    dist = runtime["score_distribution"]
    print(f"Accuracy@1 {runtime['accuracy@1']} | điểm Top-1 khi đúng TB "
          f"{dist['top1_correct'].get('mean')} | bài ngoài CSDL TB "
          f"{dist['held_out_best'].get('mean')} (P95 {dist['held_out_best'].get('p95')}, "
          f"max {dist['held_out_best'].get('max')}) | nhiễu max {dist['noise_best'].get('max')}")
    best = runtime["recommended_tau_cover"]
    print(f"τCover đề xuất (đưa vào cấu hình): {best['threshold']} — P={best['precision']} "
          f"R={best['recall']} F1={best['f1']} "
          f"False Match Rate={best['false_match_rate'] * 100:.1f}% "
          f"({'ĐẠT' if runtime['tau_cover_meets_fmr_target'] else 'KHÔNG ĐẠT'} "
          f"ngưỡng ≤5% của §16 và không nhận nhiễu)")


def load_exp03_reference(reference_windows: int, query_windows: int):
    """
    Nạp số liệu MERT của EXP-03 để so sánh, SAU KHI kiểm tra hai bên cùng giao thức.

    Không có EXP-03 thì vẫn chạy được, chỉ là bảng so sánh thiếu cột MERT — thà
    thiếu cột còn hơn đặt cạnh nhau hai con số đo ở hai điều kiện khác nhau.
    """
    path = os.path.join(RESULTS_DIR, "exp03_pooling.json")
    if not os.path.exists(path):
        print("Chưa có kết quả EXP-03 -> bỏ cột so sánh MERT.")
        return None

    with open(path, encoding="utf-8") as f:
        exp03 = json.load(f)

    expected = protocol_params(reference_windows, query_windows)
    actual = {k: exp03["parameters"].get(k) for k in expected}
    if actual != expected:
        print("EXP-03 chạy ở giao thức KHÁC -> bỏ cột so sánh MERT.")
        print(f"  EXP-07: {expected}")
        print(f"  EXP-03: {actual}")
        return None
    return exp03["metrics"]["mean"]["per_transformation"]


def main() -> int:
    try:
        manifest = load_query_manifest()
    except FileNotFoundError as e:
        print(e)
        return 1

    sources = resolve_source_audio(manifest)
    if not sources:
        print("Không tìm thấy file audio gốc nào để dựng reference.")
        return 1

    ref_timing, query_timing = [], []

    print(f"Dựng reference chroma từ {len(sources)} file audio gốc "
          f"(cửa sổ {WINDOW_S:g}s / bước {REF_HOP_S:g}s)...")
    ref_meta, ref_vectors = [], []
    for src, path in sources.items():
        rows, duration = describe_file(path, REF_HOP_S, ref_timing)
        for start, end, descriptor in rows:
            ref_meta.append({"source": src, "start": start, "end": end})
            ref_vectors.append(descriptor)
        print(f"  {src}: {duration:.1f}s -> {len(rows)} cửa sổ")

    print(f"\nDựng truy vấn từ {len(manifest)} file biến đổi "
          f"(cửa sổ {WINDOW_S:g}s / bước {QUERY_HOP_S:g}s)...")
    q_meta, q_vectors = [], []
    for row in manifest:
        src = row["source_file"]
        if src not in sources:
            continue
        path = row["path"]
        if not os.path.isabs(path):
            path = os.path.join(config.BASE_DIR, path)
        if not os.path.exists(path):
            print(f"  thiếu file truy vấn: {path}")
            continue

        transformation = row["transformation"]
        factor = query_tempo_factor(row)
        crop_start = query_crop_start(row)
        pitch_steps = query_pitch_steps(row)
        rows, _ = describe_file(path, QUERY_HOP_S, query_timing)
        for start, end, descriptor in rows:
            q_meta.append({
                "source": src,
                "transformation": transformation,
                "orig_start": crop_start + start * factor,
                "orig_end": crop_start + end * factor,
                # Dịch cao độ k bán cung -> phải xoay truy vấn ngược lại -k để khớp
                "expected_oti": (-pitch_steps) % 12,
            })
            q_vectors.append(descriptor)

    if not q_meta:
        print("Không dựng được truy vấn nào.")
        return 1
    print(f"  -> {len(ref_meta)} cửa sổ reference, {len(q_meta)} cửa sổ truy vấn")

    reference_matrix = np.vstack(ref_vectors)

    # ----------------------------------------------------------------------
    # Chấm điểm
    # ----------------------------------------------------------------------
    hits = {k: 0 for k in K_VALUES}
    reciprocal, average_precisions = [], []
    search_ms, top1_records = [], []
    oti_correct, oti_total = 0, 0
    correct_scores, wrong_scores, held_out_scores = [], [], []
    per_transformation = {}

    for descriptor, query in zip(q_vectors, q_meta):
        t0 = time.perf_counter()
        ranked = search(descriptor, reference_matrix, top_k=len(ref_meta))
        search_ms.append((time.perf_counter() - t0) * 1000)

        relevant = [i for i, ref in enumerate(ref_meta) if is_relevant(ref, query)]
        relevant_set = set(relevant)

        rank = None
        found, precision_sum = 0, 0.0
        for position, candidate in enumerate(ranked, start=1):
            if candidate["index"] in relevant_set:
                if rank is None:
                    rank = position
                found += 1
                precision_sum += found / position
                if found == len(relevant_set):
                    break

        for k in K_VALUES:
            if rank is not None and rank <= k:
                hits[k] += 1
        reciprocal.append(1.0 / rank if rank else 0.0)
        average_precisions.append(
            precision_sum / len(relevant_set) if relevant_set else 0.0)

        top1 = ranked[0]
        top1_correct = top1["index"] in relevant_set
        top1_records.append((float(top1["similarity_score"]), top1_correct))
        (correct_scores if top1_correct else wrong_scores).append(
            float(top1["similarity_score"]))

        # Held-out (giao thức EXP-06): bỏ HẴN bản ghi nguồn khỏi reference. Điểm
        # cao nhất còn lại chính là điểm mà một bài NGOÀI CSDL sẽ đạt được — đây
        # là căn cứ duy nhất để nói τCover có an toàn hay không.
        held_out_scores.append(float(max(
            (c["similarity_score"] for c in ranked
             if ref_meta[c["index"]]["source"] != query["source"]),
            default=0.0)))

        # OTI có bắt đúng lượng dịch cao độ không — đo TRÊN CỬA SỔ ĐÚNG, tách
        # hẳn khỏi chuyện xếp hạng, để biết cơ chế xoay vòng có hoạt động không.
        oti_ok = False
        if relevant:
            best = max((c for c in ranked if c["index"] in relevant_set),
                       key=lambda c: c["similarity_score"])
            oti_ok = best["oti"] == query["expected_oti"]
            oti_total += 1
            oti_correct += int(oti_ok)

        bucket = per_transformation.setdefault(
            query["transformation"], {"n": 0, "hit@1": 0, "oti_ok": 0, "scores": []})
        bucket["n"] += 1
        bucket["hit@1"] += int(rank == 1)
        bucket["oti_ok"] += int(oti_ok)
        bucket["scores"].append(float(top1["similarity_score"]))

    n = len(q_meta)
    metrics = {
        **{f"recall@{k}": round(hits[k] / n, 4) for k in K_VALUES},
        "mrr": round(float(np.mean(reciprocal)), 4),
        "map": round(float(np.mean(average_precisions)), 4),
        "oti_accuracy": round(oti_correct / oti_total, 4) if oti_total else 0.0,
        "queries": n,
        "reference_windows": len(ref_meta),
        "descriptor_dimension": int(reference_matrix.shape[1]),
    }

    noise_scores = [float(search(descriptor, reference_matrix, top_k=1)[0]["similarity_score"])
                    for _, descriptor in noise_probe_descriptors()]

    # --- Sweep ngưỡng τCover ----------------------------------------------
    sweep = threshold_sweep(top1_records, held_out_scores, noise_scores)
    best, safe = pick_threshold(sweep)
    metrics["threshold_sweep"] = sweep
    metrics["window_level_recommended_tau_cover"] = best

    # Ngưỡng đưa vào cấu hình phải đo đúng điều kiện server; cấp cửa sổ chỉ là
    # phương án dự phòng khi chưa dựng chỉ mục cover.
    runtime = runtime_protocol(manifest)
    metrics["runtime_protocol"] = runtime
    chosen = runtime or {"recommended_tau_cover": best, "tau_cover_meets_fmr_target": safe}
    metrics["recommended_tau_cover"] = chosen["recommended_tau_cover"]
    metrics["tau_cover_meets_fmr_target"] = chosen["tau_cover_meets_fmr_target"]
    metrics["recommended_tau_cover_source"] = "runtime_protocol" if runtime else "window_level"

    metrics["score_distribution"] = {
        "top1_correct": {
            "count": len(correct_scores),
            "mean": round(float(np.mean(correct_scores)), 4) if correct_scores else None,
            "p05": round(float(np.percentile(correct_scores, 5)), 4) if correct_scores else None,
        },
        "top1_wrong": {
            "count": len(wrong_scores),
            "mean": round(float(np.mean(wrong_scores)), 4) if wrong_scores else None,
            "p95": round(float(np.percentile(wrong_scores, 95)), 4) if wrong_scores else None,
        },
        "held_out_best": {
            "count": len(held_out_scores),
            "mean": round(float(np.mean(held_out_scores)), 4),
            "p95": round(float(np.percentile(held_out_scores, 95)), 4),
            "max": round(float(np.max(held_out_scores)), 4),
        },
        "noise_best": {
            "count": len(noise_scores),
            "mean": round(float(np.mean(noise_scores)), 4),
            "max": round(float(np.max(noise_scores)), 4),
        },
    }
    metrics["latency_ms"] = {
        "descriptor_reference_mean": round(float(np.mean(ref_timing)), 1),
        "descriptor_query_mean": round(float(np.mean(query_timing)), 1),
        "search_mean": round(float(np.mean(search_ms)), 3),
        "search_p95": round(float(np.percentile(search_ms, 95)), 3),
    }
    metrics["per_transformation"] = {
        t: {"n": b["n"],
            "accuracy@1": round(b["hit@1"] / b["n"], 4),
            "oti_accuracy": round(b["oti_ok"] / b["n"], 4),
            "mean_top1_score": round(float(np.mean(b["scores"])), 4)}
        for t, b in sorted(per_transformation.items())
    }

    # ----------------------------------------------------------------------
    # In bảng
    # ----------------------------------------------------------------------
    mert = load_exp03_reference(len(ref_meta), len(q_meta))

    rows = []
    for family, names in FAMILIES.items():
        present = [t for t in names if t in metrics["per_transformation"]]
        if not present:
            continue
        total = sum(metrics["per_transformation"][t]["n"] for t in present)
        chroma_ok = sum(metrics["per_transformation"][t]["accuracy@1"]
                        * metrics["per_transformation"][t]["n"] for t in present)
        oti_ok = sum(metrics["per_transformation"][t]["oti_accuracy"]
                     * metrics["per_transformation"][t]["n"] for t in present)
        score = sum(metrics["per_transformation"][t]["mean_top1_score"]
                    * metrics["per_transformation"][t]["n"] for t in present) / total

        mert_cell = "—"
        if mert:
            mert_n = sum(mert[t]["n"] for t in present if t in mert)
            if mert_n:
                mert_hit = sum(mert[t]["accuracy@1"] * mert[t]["n"]
                               for t in present if t in mert)
                mert_cell = f"{mert_hit / mert_n * 100:.0f}%"

        rows.append([family, total, f"{chroma_ok / total * 100:.0f}%", mert_cell,
                     f"{oti_ok / total * 100:.0f}%", round(score, 3)])

    print_table(
        "EXP-07 — Chroma/OTI so với MERT, trên CÙNG bộ cửa sổ",
        rows,
        ["Nhóm biến đổi", "N", "Chroma@1", "MERT@1", "OTI đúng", "Điểm TB"],
    )

    print_table(
        "EXP-07 — Tổng hợp",
        [["Recall@1", metrics["recall@1"]], ["Recall@5", metrics["recall@5"]],
         ["MRR", metrics["mrr"]], ["mAP", metrics["map"]],
         ["OTI đúng (trên cửa sổ đúng)", metrics["oti_accuracy"]],
         ["Trích đặc trưng (ms/cửa sổ)", metrics["latency_ms"]["descriptor_query_mean"]],
         ["Tìm kiếm (ms)", metrics["latency_ms"]["search_mean"]]],
        ["Metric", "Giá trị"],
    )

    print_table(
        f"EXP-07 — Sweep τCover cấp cửa sổ (reference = {len(sources)} bài nguồn)",
        [[r["threshold"], r["precision"], r["recall"], r["f1"], r["accepted"],
          f"{r['false_match_rate'] * 100:.1f}%"]
         for r in sweep if r["threshold"] in SWEEP_PRINT],
        ["τCover", "Precision", "Recall", "F1", "Chấp nhận", "False Match"],
    )

    dist = metrics["score_distribution"]
    print(f"\nĐiểm Top-1 khi ĐÚNG : TB {dist['top1_correct']['mean']} "
          f"(P05 {dist['top1_correct']['p05']}, n={dist['top1_correct']['count']})")
    print(f"Điểm Top-1 khi SAI  : TB {dist['top1_wrong']['mean']} "
          f"(P95 {dist['top1_wrong']['p95']}, n={dist['top1_wrong']['count']})")
    print(f"\nτCover cấp cửa sổ: {best['threshold']} — P={best['precision']} "
          f"R={best['recall']} F1={best['f1']} "
          f"False Match Rate={best['false_match_rate'] * 100:.1f}% "
          f"({'ĐẠT' if safe else 'KHÔNG ĐẠT'} ngưỡng ≤5% của §16)")
    held = metrics["score_distribution"]["held_out_best"]
    print(f"Điểm cao nhất khi bài KHÔNG có trong CSDL: TB {held['mean']} "
          f"(P95 {held['p95']}, max {held['max']})")
    noise = metrics["score_distribution"]["noise_best"]
    print(f"Điểm cao nhất của mẫu nhiễu ({noise['count']} mẫu): TB {noise['mean']}, "
          f"max {noise['max']} — τCover phải cao hơn max này")
    if runtime:
        print_runtime(runtime)

    pitch = [metrics["per_transformation"][t] for t in FAMILIES["Dịch cao độ"]
             if t in metrics["per_transformation"]]
    if pitch and mert:
        pitch_n = sum(b["n"] for b in pitch)
        chroma_pitch = sum(b["accuracy@1"] * b["n"] for b in pitch) / pitch_n
        mert_pitch = sum(mert[t]["accuracy@1"] * mert[t]["n"]
                         for t in FAMILIES["Dịch cao độ"] if t in mert)
        mert_pitch /= sum(mert[t]["n"] for t in FAMILIES["Dịch cao độ"] if t in mert)
        print(f"\nĐiểm mù dịch cao độ: MERT {mert_pitch * 100:.1f}% -> "
              f"Chroma/OTI {chroma_pitch * 100:.1f}%")

    path = save_result(
        EXPERIMENT_ID,
        params={
            "protocol": "window-level retrieval, CÙNG bộ cửa sổ với EXP-03",
            **protocol_params(len(ref_meta), len(q_meta)),
            "k_values": list(K_VALUES),
            "source_files": sorted(sources),
            "feature": "chroma_cqt + Optimal Transposition Index (12 phép xoay)",
            "chroma_sr": CHROMA_SR,
            "descriptor_frames": reference_matrix.shape[1] // 12,
            "descriptor_normalization": "trừ trung bình từng khung, chuẩn hoá L2 từng khung rồi toàn vector",
            "similarity": "cosine trên descriptor đã chuẩn hoá L2, lấy max qua 12 phép xoay",
            "noise_probes": {"colors": list(NOISE_COLORS), "seeds": len(NOISE_SEEDS)},
            "runtime_protocol": {
                "query": f"{RUNTIME_DURATION_S:g} giây đầu của file truy vấn (offset 0)",
                "gallery": "toàn bộ chỉ mục cover (scripts/build_cover_index.py)",
                "held_out": ("bỏ bản trùng fingerprint, bản gần trùng và (với "
                             "audio_overlay) bài bị trộn chồng — xem "
                             "metrics.runtime_protocol.held_out_exclusions"),
            },
            "compared_against_exp03": bool(mert),
        },
        metrics=metrics,
        notes=(
            "KHÔNG phải thí nghiệm cover thật: repo chưa có bản thu cover nào (§8 yêu "
            "cầu subset SecondHandSongs, 100–200 composition × 2–5 version). Thí nghiệm "
            "này đo tính BẤT BIẾN VỚI DỊCH CAO ĐỘ và đổi tốc độ — cơ chế cốt lõi mà một "
            "hệ cover dựa vào, tức điều kiện CẦN chứ không phải điều kiện đủ. "
            "`recommended_tau_cover` lấy từ `runtime_protocol` khi đã có chỉ mục cover: "
            "truy vấn và reference cùng quy ước với server, và bài ngoài CSDL phải thắng "
            "cả chỉ mục chứ không chỉ vài chục bài nguồn. Ngưỡng cấp cửa sổ (reference = "
            f"{len(sources)} bài nguồn) lạc quan hơn vì có ít ứng viên để nhận nhầm."
        ),
    )
    print(f"\nĐã lưu kết quả: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
