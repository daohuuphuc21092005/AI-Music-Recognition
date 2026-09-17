"""
EXP-03 — Pooling Strategy (§15): mean pooling vs mean+std pooling.

Câu hỏi: gộp chuỗi hidden state của MERT theo thời gian bằng `mean` (768 chiều,
đúng như `embedding_service` đang chạy) có thua `mean + std` (1536 chiều) không?
Độ lệch chuẩn theo thời gian mang thông tin về ĐỘ BIẾN THIÊN của đoạn nhạc —
thứ mà mean làm phẳng hoàn toàn.

Thiết kế (khác EXP-02 ở chỗ này, cần đọc kỹ):
  EXP-02 xếp hạng ở MỨC BẢN GHI trên index có sẵn. Ở đây không dùng lại index
  đó được, vì nó chỉ chứa vector mean — muốn có mean+std thì bắt buộc phải chạy
  lại MERT trên chính tín hiệu. Chạy lại cho cả 1.000 bản ghi là thừa: câu hỏi
  của EXP-03 là so sánh HAI PHÉP GỘP, không phải đo lại độ phủ retrieval. Vì vậy
  EXP-03 tự dựng reference từ 60 bản ghi nguồn của tập truy vấn và chấm điểm ở
  MỨC CỬA SỔ THỜI GIAN:

  - Reference : mọi cửa sổ 15s / bước 10s của các file audio gốc — đúng quy ước
                của `scripts/build_embeddings.py`, nên kết luận chuyển thẳng
                sang index production được.
  - Truy vấn  : cửa sổ 15s / bước 5s cắt từ các file đã biến đổi trong
                `data/test_queries/` (lấy dày hơn chỉ để tăng số mẫu thống kê).
  - Nhãn đúng : mọi cửa sổ reference CÙNG bản ghi chồng lấn >= 50% với khoảng
                thời gian gốc mà truy vấn ánh xạ tới. `augment_audio.py` cắt từ
                giây CROP_START=30 nên ánh xạ là xác định; riêng nhóm tempo thì
                trục thời gian co giãn theo đúng hệ số time_stretch.

  Hai chiến lược dùng CHUNG một lượt suy luận MERT — chỉ khác đúng phép gộp,
  nên chênh lệch đo được là của pooling chứ không phải của model hay dữ liệu.

Metrics (§15): Recall@1, Recall@5, MRR, Latency.

Chạy trước: python scripts/augment_audio.py --from-db
"""
import base64
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import librosa
import numpy as np
import torch

from backend import config
from backend.services.embedding_service import get_device, load_model
from backend.services.retrieval_service import build_faiss_index, search_top_k
from experiments.common import (
    Checkpoint,
    OVERLAP_MIN,
    QUERY_HOP_S,
    REF_HOP_S,
    WINDOW_S,
    load_query_manifest,
    overlap_ratio,
    print_table,
    protocol_params,
    query_crop_start,
    query_tempo_factor,
    resolve_source_audio,
    save_result,
    windows,
)

EXPERIMENT_ID = "exp03_pooling"

K_VALUES = (1, 5)


# --------------------------------------------------------------------------
# Trích đặc trưng: một lượt MERT -> hai vector (mean và mean+std)
# --------------------------------------------------------------------------
def pool_both(chunk: np.ndarray, sr: int, processor, model, timing: dict):
    """Trả (vector mean 768 chiều, vector mean+std 1536 chiều), đều chuẩn hoá L2."""
    # Input phải nằm cùng thiết bị với model: load_model() đặt model lên GPU khi
    # có CUDA, và để input ở CPU thì conv1d đầu tiên ném RuntimeError.
    device = get_device()
    inputs = processor(chunk, sampling_rate=sr, return_tensors="pt").to(device)

    def synchronize():
        # GPU chạy bất đồng bộ: không chờ thì đồng hồ chỉ đo lúc XẾP lệnh
        if device == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state      # (1, T, 768)
    synchronize()
    timing.setdefault("forward_ms", []).append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    mean = hidden.mean(dim=1)
    v_mean = torch.nn.functional.normalize(mean, p=2, dim=1).squeeze().cpu().numpy()
    timing.setdefault("pool_mean_ms", []).append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    std = hidden.std(dim=1)
    v_ms = torch.nn.functional.normalize(torch.cat([mean, std], dim=1), p=2, dim=1)
    v_ms = v_ms.squeeze().cpu().numpy()
    timing.setdefault("pool_mean_std_ms", []).append((time.perf_counter() - t0) * 1000)

    return v_mean.astype("float32"), v_ms.astype("float32")


def encode_vectors(vectors) -> list:
    """
    Vector -> chuỗi base64 của float32 thô.

    Không ghi vector dạng số JSON: mỗi float sẽ tốn ~20 ký tự (repr float64 của
    một giá trị float32) và checkpoint phình lên vài trăm MB. Làm tròn cho ngắn
    thì lại đổi giá trị. base64 của chính các byte float32 vừa gọn hơn ~4 lần
    vừa khôi phục ĐÚNG TỪNG BIT — checkpoint không được phép làm lệch kết quả.
    """
    return [base64.b64encode(np.asarray(v, dtype="float32").tobytes()).decode()
            for v in vectors]


def decode_vectors(encoded) -> list:
    return [np.frombuffer(base64.b64decode(e), dtype="float32") for e in encoded]


def embed_file(path: str, hop: float, processor, model, timing: dict):
    """Trả danh sách (start, end, vector_mean, vector_mean_std) cho một file."""
    audio, sr = librosa.load(path, sr=config.MERT_SAMPLE_RATE, mono=True)
    duration = len(audio) / sr
    out = []
    for start, end in windows(duration, hop):
        chunk = audio[int(start * sr):int(end * sr)]
        if len(chunk) < sr:
            continue
        v_mean, v_ms = pool_both(chunk, sr, processor, model, timing)
        out.append((start, end, v_mean, v_ms))
    return out, duration


# --------------------------------------------------------------------------
# Chấm điểm
# --------------------------------------------------------------------------
def evaluate(query_vectors, ref_vectors, ref_meta, queries_meta):
    """Xếp hạng toàn bộ reference cho từng truy vấn -> Recall@K, MRR, latency."""
    matrix = np.vstack(ref_vectors)
    index = build_faiss_index(matrix)

    hits = {k: 0 for k in K_VALUES}
    rec_hits = 0
    reciprocal, search_ms, top1_scores = [], [], []

    per_transformation = {}
    for qv, qmeta in zip(query_vectors, queries_meta):
        t0 = time.perf_counter()
        scores, indices = search_top_k(index, qv, top_k=index.ntotal)
        search_ms.append((time.perf_counter() - t0) * 1000)

        rank = None
        for position, idx in enumerate(indices, start=1):
            ref = ref_meta[idx]
            if ref["source"] != qmeta["source"]:
                continue
            if overlap_ratio(ref["start"], ref["end"],
                             qmeta["orig_start"], qmeta["orig_end"]) >= OVERLAP_MIN:
                rank = position
                break

        # Đúng BẢN GHI (không cần đúng cửa sổ) — tương ứng cách API xếp hạng
        rec_ok = bool(indices) and ref_meta[indices[0]]["source"] == qmeta["source"]
        rec_hits += int(rec_ok)

        for k in K_VALUES:
            if rank is not None and rank <= k:
                hits[k] += 1
        reciprocal.append(1.0 / rank if rank else 0.0)
        top1_scores.append(float(scores[0]) if scores else 0.0)

        bucket = per_transformation.setdefault(
            qmeta["transformation"], {"n": 0, "hit@1": 0, "recording_hit": 0})
        bucket["n"] += 1
        bucket["hit@1"] += int(rank == 1)
        bucket["recording_hit"] += int(rec_ok)

    n = len(queries_meta)
    return {
        **{f"recall@{k}": round(hits[k] / n, 4) for k in K_VALUES},
        "mrr": round(float(np.mean(reciprocal)), 4),
        "recording_recall@1": round(rec_hits / n, 4),
        "mean_top1_similarity": round(float(np.mean(top1_scores)), 4),
        "search_ms_mean": round(float(np.mean(search_ms)), 3),
        "search_ms_p95": round(float(np.percentile(search_ms, 95)), 3),
        "dimension": int(matrix.shape[1]),
        "index_bytes": int(matrix.nbytes),
        "queries": n,
        "per_transformation": {
            t: {"n": b["n"],
                "accuracy@1": round(b["hit@1"] / b["n"], 4),
                "recording_accuracy@1": round(b["recording_hit"] / b["n"], 4)}
            for t, b in sorted(per_transformation.items())
        },
    }


def main() -> int:
    try:
        manifest = load_query_manifest()
    except FileNotFoundError as e:
        print(e)
        return 1

    # --- Reference: audio gốc còn giữ được -------------------------------
    sources = resolve_source_audio(manifest)
    if not sources:
        print("Không tìm thấy file audio gốc nào để dựng reference.")
        return 1

    processor, model = load_model()
    timing = {}

    print(f"Dựng reference từ {len(sources)} file audio gốc "
          f"(cửa sổ {WINDOW_S:g}s / bước {REF_HOP_S:g}s)...")
    # Checkpoint: EXP-03 chạy vài giờ vì phải suy luận MERT lại từ đầu cho cả
    # reference lẫn truy vấn. Khoá có tiền tố để hai giai đoạn không lẫn nhau.
    checkpoint = Checkpoint(EXPERIMENT_ID, {
        "window_s": WINDOW_S,
        "reference_hop_s": REF_HOP_S,
        "query_hop_s": QUERY_HOP_S,
        "overlap_min": OVERLAP_MIN,
        "sources": len(sources),
        "manifest_rows": len(manifest),
        "mert_model_version": config.MERT_MODEL_VERSION,
        # Độ trễ forward chỉ so được giữa các lượt cùng thiết bị
        "device": get_device(),
    })

    ref_meta, ref_mean, ref_mean_std = [], [], []
    for src, path in sources.items():
        key = "ref:" + src
        if not checkpoint.has(key):
            rows, duration = embed_file(path, REF_HOP_S, processor, model, timing)
            checkpoint.add(key, {
                "meta": [{"source": src, "start": start, "end": end}
                         for start, end, _vm, _vs in rows],
                "mean": encode_vectors([vm for _s, _e, vm, _vs in rows]),
                "mean_std": encode_vectors([vs for _s, _e, _vm, vs in rows]),
                "duration": duration,
            })
            print(f"  {src}: {duration:.1f}s -> {len(rows)} cửa sổ")

    for entry in checkpoint.records:
        if not entry["meta"] or "transformation" in entry["meta"][0]:
            continue                      # nhóm truy vấn, xử lý ở vòng sau
        ref_meta.extend(entry["meta"])
        ref_mean.extend(decode_vectors(entry["mean"]))
        ref_mean_std.extend(decode_vectors(entry["mean_std"]))

    # --- Truy vấn: các file đã biến đổi ----------------------------------
    print(f"\nDựng truy vấn từ {len(manifest)} file biến đổi "
          f"(cửa sổ {WINDOW_S:g}s / bước {QUERY_HOP_S:g}s)...")
    q_meta, q_mean, q_mean_std = [], [], []
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

        key = "q:" + row["path"]
        if checkpoint.has(key):
            continue

        transformation = row["transformation"]
        factor = query_tempo_factor(row)
        crop_start = query_crop_start(row)
        rows, _ = embed_file(path, QUERY_HOP_S, processor, model, timing)
        checkpoint.add(key, {
            "meta": [{
                "source": src,
                "transformation": transformation,
                # Ánh xạ về trục thời gian của bản gốc
                "orig_start": crop_start + start * factor,
                "orig_end": crop_start + end * factor,
            } for start, end, _vm, _vs in rows],
            "mean": encode_vectors([vm for _s, _e, vm, _vs in rows]),
            "mean_std": encode_vectors([vs for _s, _e, _vm, vs in rows]),
        })

    for entry in checkpoint.records:
        if not entry["meta"] or "transformation" not in entry["meta"][0]:
            continue                      # nhóm reference, đã nạp ở vòng trước
        q_meta.extend(entry["meta"])
        q_mean.extend(decode_vectors(entry["mean"]))
        q_mean_std.extend(decode_vectors(entry["mean_std"]))

    if not q_meta:
        print("Không dựng được truy vấn nào.")
        return 1

    print(f"  -> {len(ref_meta)} cửa sổ reference, {len(q_meta)} cửa sổ truy vấn")

    # --- Chấm điểm hai chiến lược ----------------------------------------
    results = {
        "mean": evaluate(q_mean, ref_mean, ref_meta, q_meta),
        "mean_std": evaluate(q_mean_std, ref_mean_std, ref_meta, q_meta),
    }
    for name, res in results.items():
        res["pool_ms_mean"] = round(float(np.mean(timing[f"pool_{name}_ms"])), 4)
    forward_ms = round(float(np.mean(timing["forward_ms"])), 1)

    print_table(
        "EXP-03 — Pooling Strategy (chấm ở mức cửa sổ thời gian)",
        [[name, res["dimension"], res["recall@1"], res["recall@5"], res["mrr"],
          res["recording_recall@1"], res["mean_top1_similarity"],
          res["pool_ms_mean"], res["search_ms_mean"]]
         for name, res in results.items()],
        ["Pooling", "Chiều", "Recall@1", "Recall@5", "MRR",
         "Recall@1 bản ghi", "Sim TB", "Pool (ms)", "Search (ms)"],
    )
    print(f"\nMERT forward trung bình {forward_ms} ms/cửa sổ — CHUNG cho cả hai "
          f"chiến lược, nên chênh lệch latency chỉ nằm ở pooling và search.")

    delta_r1 = results["mean_std"]["recall@1"] - results["mean"]["recall@1"]
    delta_r5 = results["mean_std"]["recall@5"] - results["mean"]["recall@5"]
    print(f"Chênh lệch mean+std so với mean: Recall@1 {delta_r1:+.4f}, "
          f"Recall@5 {delta_r5:+.4f}, index to gấp "
          f"{results['mean_std']['index_bytes'] / results['mean']['index_bytes']:.1f} lần.")

    path = save_result(
        EXPERIMENT_ID,
        params={
            "protocol": "window-level retrieval; reference = cửa sổ của audio gốc, "
                        "truy vấn = cửa sổ của file đã biến đổi",
            **protocol_params(len(ref_meta), len(q_meta)),
            "k_values": list(K_VALUES),
            "source_files": sorted(sources),
            "strategies": {"mean": 768, "mean_std": 1536},
            "device": get_device(),
            "similarity": "cosine (IndexFlatIP trên vector đã chuẩn hoá L2)",
            "shared_forward_pass": True,
        },
        metrics={
            **results,
            "forward_ms_mean": forward_ms,
            "delta_mean_std_minus_mean": {
                "recall@1": round(delta_r1, 4),
                "recall@5": round(delta_r5, 4),
                "mrr": round(results["mean_std"]["mrr"] - results["mean"]["mrr"], 4),
            },
        },
        notes=(
            "Reference gồm cửa sổ của 60 bản ghi nguồn, truy vấn là cửa sổ của "
            "1.140 file đã biến đổi. Con số đáng đọc là Recall@1/@5 ở MỨC CỬA SỔ: "
            "các cửa sổ chồng lấn của cùng một bài rất giống nhau nên đây là phép "
            "thử đủ khó để phân biệt hai chiến lược pooling. Cố ý KHÔNG dựng lại "
            "reference cho cả 1.000 bản ghi — EXP-03 so sánh hai phép gộp trên "
            "cùng một lượt suy luận MERT, nên chênh lệch đo được là của pooling "
            "chứ không phải của quy mô reference. Độ phủ retrieval ở mức bản ghi "
            "đã có EXP-02 và EXP-04 đo."
        ),
    )
    checkpoint.close(remove=True)
    print(f"\nĐã lưu kết quả: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
