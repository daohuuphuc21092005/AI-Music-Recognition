---
description: Quy chuẩn quản trị thực nghiệm, versioning model/dataset, bộ dữ liệu thử nghiệm, data augmentation và ngưỡng nghiệm thu
globs:
  - "experiments/**"
  - "scripts/**"
  - "data/**"
---

# 06. Thực nghiệm & Đánh giá (Experiments & Evaluation Standards)

Dự án áp dụng phương pháp nghiên cứu thực nghiệm chặt chẽ (Empirical AI/MIR). Mọi kết luận về hiệu năng phải được chứng minh thông qua số liệu thực nghiệm có khả năng tái lập.

---

## 1. Quy chuẩn Versioning Model & Thực nghiệm

### Model Metadata (Không ghi chung chung "model = MERT")
Mọi cấu hình model phải lưu trữ đầy đủ các thuộc tính:
- `model_name`: Tên gốc (ví dụ: `m-a-p/MERT-v1-95M`).
- `checkpoint`: Checkpoint hoặc hash trọng số cụ thể.
- `model_version`: Phiên bản triển khai (ví dụ: `mert_v1.0`).
- `preprocessing_version`: Phiên bản tiền xử lý âm thanh (ví dụ: `norm_24k_mono_v1`).
- `embedding_dimension`: Số chiều vector (ví dụ: `768`).
- `pooling_method`: Chiến lược pooling (`mean`, `mean_std`, `attention`).
- `threshold`: Ngưỡng quyết định tối ưu (`tau_mert`).
- `date_created`: Ngày khởi tạo.

### Experiment Tracking
Mỗi thí nghiệm phải ghi nhận vào file cấu hình (CSV/JSON, không bắt buộc MLflow ở giai đoạn MVP):
`experiment_id`, `git_commit`, `dataset_version`, `split_version`, `model`, `parameters`, `threshold`, `metrics`, `date`.

---

## 2. Quy mô Dataset MVP (Mục tiêu thiết kế)

| Dataset nguồn | Vai trò trong hệ thống | Số lượng mục tiêu | Ghi chú quan trọng |
|---|---|---|---|
| **MTG-Jamendo** (CC) | Reference chính cho embedding & retrieval | 1.000 – 1.500 tracks | Nhạc có bản quyền Creative Commons rõ ràng |
| **FMA** (Free Music Archive) | Đa dạng hóa reference, negative examples | 500 – 1.000 tracks | Dùng để đánh giá độ bao phủ retrieval |
| **YouTube Audio Library** | Tham chiếu trường hợp LOW / CONDITIONAL | 300 – 500 tracks | Đầy đủ điều kiện attribution / non-attribution |
| **Creator Music** | Metadata mô phỏng cho Rule Engine | 50 – 100 records | **Tuyệt đối KHÔNG crawl audio**, chỉ mô phỏng metadata |
| **Public Domain** | Đánh giá phân biệt composition vs recording | 200 – 300 recordings | Bắt buộc kiểm tra kỹ trạng thái bản quyền bản thu |
| **Cover Dataset** (SHS subset) | Thử nghiệm nâng cao Cover Identification | 100 – 200 compositions | 2 – 5 phiên bản / composition |

> [!IMPORTANT]
> **Metadata tối thiểu cần giữ cho mỗi track**:
> `track_id`, `artist`, `title`, `duration`, `tags/genre`, `source`, `license metadata`.

---

## 3. Data Augmentation & Robustness Test Set

Để đánh giá khả năng chống chịu biến đổi của hệ thống trong thực tế, xây dựng một **Robustness Test Set cố định** từ các query gốc, áp dụng các phép biến đổi:
1. **Time crop**: Cắt ngẫu nhiên 5s, 10s, 15s, 30s tại các vị trí khác nhau của bài hát.
2. **Codec compression**: Nén lại sang định dạng MP3 (64k, 128k, 192k) và AAC (64k, 128k).
3. **Additive noise**: Thêm tạp âm môi trường (tiếng ồn quán cà phê, tiếng gió, nhiễu trắng) với SNR từ 0dB đến 20dB.
4. **Voice overlay**: Trộn giọng nói / lời bình (voice-over) đè lên nhạc với các tỷ lệ âm lượng khác nhau.
5. **Gain & EQ**: Tăng/giảm âm lượng (±3dB, ±6dB), lọc dải tần (low-pass, high-pass filter).
6. **Pitch shift**: Dịch cao độ ±1, ±2 semitone.
7. **Tempo change**: Thay đổi tốc độ phát từ 0.90x đến 1.10x (giữ nguyên pitch).

---

## 4. Danh mục 8 Thí nghiệm Bắt buộc (EXP-01 → EXP-08)

- **`EXP-01` — Fingerprint Baseline**:
  - Đánh giá Chromaprint trên tập dữ liệu clean và nhiễu nhẹ.
  - Đo lường: Precision, Recall, F1-Score, False Positive Rate (FPR), độ trễ (latency/track).
- **`EXP-02` — MERT Deep Retrieval**:
  - Đánh giá khả năng tìm kiếm tương đồng ngữ nghĩa âm nhạc của MERT.
  - Đo lường: Recall@1, Recall@5, Recall@10, MRR (Mean Reciprocal Rank), mAP.
- **`EXP-03` — Pooling Strategy**:
  - So sánh thực nghiệm giữa P1 (Mean Pooling) vs P2 (Mean + Std Pooling).
  - Đánh giá sự đánh đổi giữa kích thước vector (768 vs 1536) và độ chính xác retrieval.
- **`EXP-04` — Hybrid Cascade (Thí nghiệm cốt lõi của đồ án)**:
  - So sánh 3 kiến trúc: Fingerprint đơn lẻ vs MERT đơn lẻ vs Mô hình lai Cascade (`Fingerprint → MERT`).
  - Chứng minh mô hình Cascade đạt tốc độ cao của Fingerprint trên clean audio và độ bền vững của MERT trên modified audio.
- **`EXP-05` — Robustness Analysis**:
  - Đo lường độ suy giảm hiệu năng (Performance Degradation) của từng module trước từng loại biến đổi (Noise, Pitch, Tempo, Codec).
- **`EXP-06` — Unknown Track Detection (Từ chối nhận diện)**:
  - Kiểm thử khả năng từ chối nhận diện đối với các bản nhạc chưa từng có trong cơ sở dữ liệu.
  - Đảm bảo hệ thống không đưa ra nhận diện sai lệch khi gặp nhạc lạ.
- **`EXP-07` — Cover Identification** *(Nếu triển khai AI-3)*:
  - Đánh giá khả năng nhận diện các bản cover / biến thể giai điệu so với bản thu gốc.
- **`EXP-08` — End-to-End System Evaluation**:
  - Đánh giá toàn diện toàn bộ chu trình từ âm thanh thô đến quyết định bản quyền cuối cùng.
  - Đo lường: Macro-F1 trên 5 nhóm bản quyền, Confusion Matrix, tỷ lệ Unknown Detection Rate, tổng độ trễ End-to-End.

### Thí nghiệm mở rộng (ngoài 8 thí nghiệm bắt buộc)

- **`EXP-09` — License Learnability** (`experiments/exp09_license_learnability/run.py`):
  - Kiểm định "giấy phép có đoán được từ embedding MERT không" bằng split theo **nghệ sĩ** (nghệ sĩ trong test là mới) — đúng tình huống bài ngoài cơ sở dữ liệu.
  - Kết quả gốc: accuracy 0.382 < baseline lớp phổ biến 0.569. Bộ phân loại chỉ sinh quyền `PREDICTED`, và cổng dữ liệu quyền (`rights_gate`) hạ mọi kết luận từ nguồn này về `UNKNOWN`.
- **License Complexity Sweep** (`python scripts/train_license_classifier.py --sweep` → `experiments/results/exp09_license_complexity_sweep.json`):
  - Quét `C` của LogisticRegression trên thang log (0.001 → 100), cùng giao thức split theo nghệ sĩ; chỉ dùng nhãn từ nguồn thật (loại `SIMULATED`/`PREDICTED`).
  - Mỗi mức ghi: lỗi độ chênh ≈ 1 − macro-F1 train, lỗi phương sai ≈ khoảng cách train↔kiểm định, tổng lỗi = 1 − macro-F1 kiểm định. Chọn C có tổng lỗi thấp nhất (hoà → C nhỏ hơn).
  - Luôn báo cáo kèm baseline accuracy và baseline macro-F1. Chạy lại mỗi khi tập bản ghi có audio + nhãn thật đổi quy mô.

---

## 5. Ngưỡng Nghiệm thu Nội bộ (Internal Acceptance Thresholds)

Mục tiêu chất lượng cần đạt cho các thí nghiệm trước khi tích hợp vào sản phẩm:

| Tiêu chí đánh giá | Metric | Ngưỡng mục tiêu |
|---|---|---|
| **Clean Exact-Match** (Chromaprint) | Precision / Recall | **Precision ≥ 0.95**, **Recall ≥ 0.90** |
| **Robust Retrieval** (MERT) | Recall@5 | **Recall@5 ≥ 0.80** |
| **Unknown Detection** (Từ chối nhận diện) | False Match Rate (FMR) | **FMR ≤ 5%** |
| **End-to-End System** | Macro-F1 (trên verified subset) | **Macro-F1 ≥ 0.80** |

> Có thể điều chỉnh các ngưỡng trên sau Tuần 5 nếu số liệu baseline từ dataset thực tế cho thấy bài toán quá khó, nhưng mọi điều chỉnh đều phải lập biên bản ghi rõ lý do khoa học.
