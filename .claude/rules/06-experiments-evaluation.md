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
  - So sánh 5 hệ trên CÙNG bộ truy vấn: Fingerprint đơn lẻ, MERT đơn lẻ, Cover (chroma/OTI) đơn lẻ, Cascade `Fingerprint → MERT`, và Cascade production `Fingerprint → MERT → Cover` (khoá `cascade` trong kết quả, đúng thứ tự dừng sớm của `cascade_service`).
  - Chứng minh mô hình Cascade đạt tốc độ cao của Fingerprint trên clean audio, độ bền vững của MERT trên modified audio, và phần Cover cứu thêm được (`transformations_rescued_by_cover`).
- **`EXP-05` — Robustness Analysis**:
  - Đo lường độ suy giảm hiệu năng (Performance Degradation) của từng module trước từng loại biến đổi (Noise, Pitch, Tempo, Codec).
- **`EXP-06` — Unknown Track Detection (Từ chối nhận diện)**:
  - Kiểm thử khả năng từ chối nhận diện đối với các bản nhạc chưa từng có trong cơ sở dữ liệu.
  - Đảm bảo hệ thống không đưa ra nhận diện sai lệch khi gặp nhạc lạ.
- **`EXP-07` — Cover Identification** *(Nếu triển khai AI-3)*:
  - Đánh giá khả năng nhận diện các bản cover / biến thể giai điệu so với bản thu gốc.
  - Chấm ở **hai giao thức**, và `recommended_tau_cover` luôn lấy từ giao thức thứ hai:
    1. *Cấp cửa sổ* — cửa sổ 15s, reference chỉ là các bài nguồn của tập truy vấn. Dùng để so với MERT của EXP-03 trên CÙNG bộ cửa sổ (đo điểm mù dịch cao độ).
    2. *Điều kiện server* (`metrics.runtime_protocol`) — 30 giây đầu của file truy vấn, tìm trên TOÀN BỘ chỉ mục cover (`scripts/build_cover_index.py`, mọi bản ghi có audio thật). Bài ngoài CSDL phải thắng cả chỉ mục; held-out loại bản trùng fingerprint, bản gần trùng và (với `audio_overlay`) bài bị trộn chồng (`HeldOutProtocol`).
  - Ngưỡng chọn theo ràng buộc trước, F1 sau: không nhận mẫu nhiễu nào → FMR ≤ 0.005 → F1 cao nhất; không đạt 0.005 mới nới về trần 5% của §16. Siết hơn §16 vì Cover là tầng CUỐI của cascade, nhận nhầm ở đây ra thẳng kết luận về quyền.
  - **Quy mô reference đổi thì phải chạy lại**: cùng bộ truy vấn, τ = 0.70 cho FMR 4,9% trên 100 bài nguồn nhưng 17,4% trên chỉ mục 24.375 bài.
- **`EXP-08` — End-to-End System Evaluation**:
  - Đánh giá toàn diện toàn bộ chu trình từ âm thanh thô đến quyết định bản quyền cuối cùng.
  - Điều kiện `held_out` dùng `exclude_recording_ids` + `experiments.common.HeldOutProtocol` (bản trùng, bản gần trùng, bài bị trộn chồng) trên đúng đường chạy production. Script dừng hẳn nếu có truy vấn held-out trả về id đã bị loại. Bản cũ (`HeldOutSession` lọc SQL theo chuỗi) vừa crash với bộ đệm fingerprint vừa không che được tầng Cover.
  - Đo lường: Macro-F1 trên 5 nhóm bản quyền, Confusion Matrix, tỷ lệ Unknown Detection Rate, tổng độ trễ End-to-End.

### Thứ tự chạy và giao thức held-out chung
- `scripts/run_all_experiments.py`: hiệu chỉnh (01, 06, 03, 07) → ghi ngưỡng → kiểm ngưỡng runtime khớp `identity_gate` của `rules_v1.yaml` → tiêu thụ (02, 04, 05, 08, sweep license).
- EXP-02 tính theo khối (`CHUNK_ROWS`), không dựng ma trận N×N (8,9 GiB ở 48.750 vector); phân vị cặp khác bản ghi lấy từ histogram (sai số 1e-4). `tests/test_exp02_chunked.py` đối chiếu với cách tính N×N.
- EXP-05 từ chối ghép khi EXP-01 và EXP-04 không cùng bộ truy vấn (từng lệch 0/1.900 mà bảng vẫn in).
- Held-out của EXP-01 / EXP-07 (điều kiện server) / EXP-08 dùng `HeldOutProtocol`; EXP-01 ghi thêm `false_positive_rate_exact_only` theo giao thức cũ để thấy phần chênh.

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

| Tiêu chí đánh giá | Metric | Ngưỡng mục tiêu | Đo được ở 24.375 bài (2026-09-17) |
|---|---|---|---|
| **Clean Exact-Match** (Chromaprint) | Precision / Recall | **Precision ≥ 0.95**, **Recall ≥ 0.90** | ✅ P 1.000 · R 1.000 (EXP-01, 800 truy vấn không đổi trục thời gian/cao độ, τFP 0.30) |
| **Robust Retrieval** (MERT) | Recall@5 | **Recall@5 ≥ 0.80** | ❌ MERT **0.7921** (EXP-04 `retrieval_recall_at_k`, 1.900 truy vấn biến đổi, toàn chỉ mục) — thiếu hoàn toàn do dịch cao độ (0.185; bỏ pitch 0.954). Cover 0.8542, MERT∪Cover 0.9926 |
| **Unknown Detection** (Từ chối nhận diện) | False Match Rate (FMR) | **FMR ≤ 5%** | ✅ FP 0,21% · MERT 2,65% · Cover 0,37% · cả pipeline 0,00% (EXP-08, 760 lượt held-out) |
| **End-to-End System** | Macro-F1 (trên verified subset) | **Macro-F1 ≥ 0.80** | ✅ **0.9091**, đủ 4 lớp (LOW 114 · CONDITIONAL 475 · HIGH 171 · UNKNOWN 760) |

- **Không đọc EXP-02 như Recall@5 "robust"**: EXP-02 dùng cửa sổ chưa biến đổi của chính bài trong chỉ mục (0.9374) — cận trên. Tiêu chí §13 chấm bằng EXP-04.
- FPR của EXP-01 và FMR của EXP-07 đo trước `HeldOutProtocol` → cận trên (nhận ra bản gần trùng / bài bị trộn chồng vẫn bị đếm là nhận nhầm).
- EXP-08 chỉ phủ nhóm CREATIVE_COMMONS + fallback bằng audio thật; các nhóm còn lại do `tests/test_decision_rules.py` bảo đảm. Điểm mù lớn nhất còn lại: tempo chậm (EXP-05 8–10%; nguyên nhân: cửa sổ 30 s đầu của tầng Cover, xem `02-pipeline-architecture.md` §4).
- Robust Retrieval **chưa được điều chỉnh mục tiêu** — coi MERT∪Cover là đạt hay không là quyết định của chủ dự án, và phải ghi lý do.

> Có thể điều chỉnh các ngưỡng trên sau Tuần 5 nếu số liệu baseline từ dataset thực tế cho thấy bài toán quá khó, nhưng mọi điều chỉnh đều phải lập biên bản ghi rõ lý do khoa học.
