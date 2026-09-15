---
name: experiment-benchmark
description: Playbook thực thi, đo lường và đánh giá 8 bài thí nghiệm chuẩn (EXP-01 đến EXP-08), quản trị versioning và đối soát ngưỡng nghiệm thu nội bộ.
---

# Skill: Đo lường & Thực nghiệm MIR (`experiment-benchmark`)

Skill này hướng dẫn quy trình khoa học để thực hiện, đo lường và lập báo cáo kết quả cho 8 thí nghiệm bắt buộc (`EXP-01` đến `EXP-08`) trong dự án AI Phân loại Nhạc Bản quyền.

---

## 1. Danh mục 8 Bài Thí nghiệm Chuẩn (Experiment Matrix)

| Mã EXP | Tên thí nghiệm | Mục tiêu cốt lõi | Chỉ số đo lường chính (Metrics) |
|---|---|---|---|
| **EXP-01** | Fingerprint Baseline | Đánh giá Chromaprint trên clean và slightly distorted audio | Precision, Recall, F1, FPR, Latency/track |
| **EXP-02** | MERT Deep Retrieval | Đo lường độ chính xác tìm kiếm vector ngữ nghĩa của MERT | Recall@1, Recall@5, Recall@10, MRR, mAP |
| **EXP-03** | Pooling Strategy | So sánh P1 (Mean) vs P2 (Mean+Std) vs P3 (Attention) | Retrieval accuracy vs Vector Dimension trade-off |
| **EXP-04** | Hybrid Cascade | So sánh Fingerprint vs MERT vs Cascade (`FP → MERT`) | Macro-F1, Overall Latency, Resource Consumption |
| **EXP-05** | Robustness Analysis | Đánh giá độ bền trước 7 loại biến đổi trên Robustness Set | Per-transformation degradation rate |
| **EXP-06** | Unknown Track Detection | Đánh giá khả năng từ chối nhận diện nhạc không có trong DB | False Match Rate (FMR), Reject Accuracy |
| **EXP-07** | Cover Identification | Đánh giá nhận diện biến thể cover (nếu có module Cover) | Top-K Cover Recall, Alignment Score |
| **EXP-08** | End-to-End Evaluation | Đánh giá toàn diện toàn bộ pipeline và phân loại 5 nhóm | Macro-F1 5 nhóm, Confusion Matrix, End-to-End Latency |

---

## 2. Quy trình Thực thi & Ngăn chặn Data Leakage

### Quy tắc Data Splitting Bắt buộc
- **Split theo `recording_id`**: Tuyệt đối không chia dữ liệu theo segment. Mọi segment của cùng một bản thu phải nằm trọn vẹn trong tập Train HOẶC tập Test.
- **Cover Identification**: Phải chia theo `composition_id`. Các phiên bản cover của cùng một bài hát không được phân tán ở cả hai tập.

### Quy trình Ghi nhận Experiment Log
Mỗi lần chạy thí nghiệm, bắt buộc lưu lại file kết quả JSON với cấu trúc:

```json
{
  "experiment_id": "EXP-04_hybrid_cascade_v1",
  "git_commit": "a1b2c3d4",
  "date": "2026-09-15T05:50:00Z",
  "dataset_version": "mvp_jamendo_fma_v1.0",
  "split_version": "split_by_recording_v1",
  "models": {
    "chromaprint": {"version": "fpcalc_1.5.1", "tau_fp": 0.85},
    "mert": {
      "model_name": "m-a-p/MERT-v1-95M",
      "pooling": "mean_std",
      "dimension": 1536,
      "tau_mert": 0.78
    }
  },
  "metrics": {
    "precision": 0.962,
    "recall": 0.915,
    "recall_at_5": 0.842,
    "false_match_rate": 0.038,
    "macro_f1": 0.835,
    "avg_latency_ms": 320
  }
}
```

---

## 3. Bảng Đối soát Ngưỡng Nghiệm thu Nội bộ (Acceptance Gate)

Sau khi hoàn tất mỗi thí nghiệm, đối chiếu kết quả với bảng tiêu chuẩn nội bộ:

```
┌──────────────────────────────────────┬──────────────────────┬─────────────┐
│ Tiêu chí đánh giá                    │ Chỉ số mục tiêu      │ Kết quả EXP │
├──────────────────────────────────────┼──────────────────────┼─────────────┤
│ 1. Clean Exact-Match (Chromaprint)   │ Precision ≥ 0.95     │    [...]    │
│                                      │ Recall ≥ 0.90        │    [...]    │
│ 2. Robust Retrieval (MERT-v1-95M)    │ Recall@5 ≥ 0.80      │    [...]    │
│ 3. Unknown Detection (Nhạc lạ)       │ False Match Rate ≤ 5%│    [...]    │
│ 4. End-to-End System (Cascade)       │ Macro-F1 ≥ 0.80      │    [...]    │
└──────────────────────────────────────┴──────────────────────┴─────────────┘
```

> [!CAUTION]
> Nếu một thí nghiệm không đạt ngưỡng tối thiểu:
> 1. Không tự ý hạ thấp tiêu chuẩn nghiệm thu.
> 2. Phân tích Confusion Matrix để tìm ra nhóm bị phân loại nhầm nhiều nhất.
> 3. Kiểm tra lại việc quét ngưỡng `τFP` hoặc `τMERT` trên tập Validation trước khi tinh chỉnh model.
