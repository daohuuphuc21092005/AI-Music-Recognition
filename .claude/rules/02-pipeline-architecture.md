---
description: Kiến trúc pipeline cascade xử lý âm thanh, module tiền xử lý và các tầng AI nhận diện (Chromaprint, MERT, Cover)
globs:
  - "backend/services/**"
  - "models/**"
  - "scripts/**"
---

# 02. Kiến trúc Pipeline Cascade (Cascade Pipeline Architecture)

Hệ thống hoạt động theo mô hình **Cascade nối tiếp tuần tự**, tuyệt đối không chạy song song 5 model nhận diện.

```
INPUT AUDIO/VIDEO
  → Audio Pre-processing (tách âm thanh, chuẩn hóa, lọc nhiễu)
  → Music Segment Detection
  → Audio Fingerprinting (Chromaprint / fpcalc)
       ├─ score ≥ τFP → EXACT / NEAR-EXACT MATCH
       └─ score < τFP → Music Embedding Retrieval (MERT-v1-95M)
                            ├─ similarity ≥ τMERT → NEAR-MATCH CANDIDATE
                            └─ similarity < τMERT → Cover/Version Identification (optional, advanced)
  → Track/Composition Resolver (recording_id, composition_id)
  → Rights Database Lookup
  → Rule Engine (cây quyết định 5 nhóm)
  → Copyright Usage Risk Assessment (LOW / CONDITIONAL / HIGH / UNKNOWN)
  → Evidence Object + Recommendation
```

---

## 1. Module tiền xử lý âm thanh (Audio Pre-processing)

- **M1 Audio Ingestion**:
  - Hỗ trợ định dạng đầu vào: audio (`.mp3`, `.wav`, `.m4a`), video (`.mp4`, `.mov`).
  - File video được tách track âm thanh tự động qua `FFmpeg`.
  - Luôn lưu giữ file gốc trong quá trình xử lý tác vụ tạm.

- **M2 Normalization**:
  - **Không ép tất cả các model dùng chung sampling rate**.
  - Chromaprint sử dụng quy trình tiền xử lý nội bộ của thư viện fpcalc.
  - Model MERT sử dụng chuẩn âm thanh đầu vào **24,000 Hz (24kHz)**, Mono channel.
  - Chuẩn hóa biên độ âm lượng (Peak Normalization hoặc Loudness LUFS phù hợp).

- **M3 Segmentation**:
  - Chia đoạn tín hiệu âm thanh với cửa sổ kích thước: 10 giây / 15 giây / 30 giây kèm overlap (ví dụ: overlap 50% hoặc 5s).
  - Tách đoạn trước khi sinh fingerprint hoặc embedding để tối ưu độ chính xác cục bộ.

---

## 2. AI-1 — Exact Audio Identification (Chromaprint / fpcalc)

- **Vai trò**: Tầng lọc đầu tiên, xử lý các bản ghi giống hệt hoặc gần như nguyên bản (near-identical audio).
- **Công cụ**: Chromaprint (`fpcalc`).
- **Ngưỡng quyết định `τFP`**:
  - Tuyệt đối **không được chọn tùy tiện**.
  - Bắt buộc phải xác định thông qua việc quét ngưỡng trên Validation Set để đạt sự cân bằng tối ưu giữa Precision, Recall và False Positive Rate (FPR).
- Nếu `score ≥ τFP`: Xác nhận `match_type = EXACT_MATCH` hoặc `NEAR_EXACT_MATCH`, chuyển thẳng đến Track Resolver, **bỏ qua tầng MERT**.
- **`τFP` hiện tại = 0.30, giữ theo quyết định của chủ dự án (2026-09-18)** dù quy tắc hiệu chỉnh (F1 tầng 1 cao nhất trong nhóm FPR ≤ 0.005) đề xuất 0.10: ở cấp hệ thống F1 cascade chỉ +0.0016, nhận sai 5 → 8, và dải ±0.15 của bộ lọc hash quanh 0.10 buộc 44% truy vấn quét toàn bộ. **Hạ τFP là quyết định của người**, không để `run_all_experiments.py` tự làm — `apply_calibrated_thresholds.py` chỉ tự động siết, nới phải có `--allow-loosen`. Nếu hạ τFP, phải đối chứng lại bộ lọc hash (và thu hẹp `BAND` nếu cần) trước.
- **Lọc ứng viên theo hash** (`config.FP_PREFILTER_*`): chỉ chấm đầy đủ 50 bản ghi có nhiều hash trùng tuyệt đối nhất, quay về quét toàn bộ khi điểm nằm trong ±0.15 quanh ngưỡng hiệu dụng hoặc truy vấn < 10 s (phép đối chứng chỉ phủ 10–33 s). Đối chứng hàm production trên 1.900 truy vấn: 0 lệch quyết định, 1,6% quay về quét toàn bộ, độ trễ TB 110 ms thay vì ~4,5 s. Đổi `TOP_K`/`BAND` thì phải đối chứng lại; EXP-01 (hiệu chỉnh τFP) luôn quét toàn bộ.

---

## 3. AI-2 — Deep Music Retrieval (MERT-v1-95M)

- **Điều kiện kích hoạt**: Chỉ chạy khi tầng Fingerprint trả về `score < τFP`.
- **Model chuẩn**: **`MERT-v1-95M`** (Pretrained Hugging Face). Không cần bản 330M trong giai đoạn MVP.
- **Chiến lược Pooling**:
  - Bắt buộc so sánh thực nghiệm tối thiểu 2 chiến lược:
    - **P1 (Mean Pooling)**: Lấy giá trị trung bình qua các time steps.
    - **P2 (Mean + Std Pooling)**: Ghép vector trung bình và độ lệch chuẩn.
    - **P3 (Attention Pooling)**: Chỉ thử nghiệm nếu còn thời gian sau khi hoàn thành P1/P2.
- **Vector Search Engine**:
  - Giai đoạn nghiên cứu / thí nghiệm: **FAISS** (FlatIP hoặc HNSW, tính Cosine Similarity).
  - Giai đoạn sản phẩm / backend: **Qdrant** (hoặc tích hợp qua Docker).
- **Ngưỡng tương đồng `τMERT`**:
  - Nếu `similarity ≥ τMERT`: Xác định ứng viên `NEAR_MATCH_CANDIDATE` và trích xuất `recording_id`.
  - Nếu `similarity < τMERT`: Chuyển sang AI-3 hoặc đánh dấu không nhận diện được.

---

## 4. AI-3 — Cover / Version Identification (Tùy chọn, nâng cao)

- **Điều kiện phát triển**: Chỉ được triển khai sau khi AI-1 và AI-2 đã hoạt động trơn tru và đạt ngưỡng nghiệm thu.
- **Baseline bắt buộc**: Biểu diễn CQT (Constant-Q Transform) / Chroma features kết hợp so khớp chuỗi thời gian (dynamic time warping / dynamic alignment).
- **Descriptor đang dùng** (`cover_service.normalize_frames`): chroma-CQT gộp về 64 khung, **trừ trung bình từng khung** rồi chuẩn hoá L2; so khớp bằng cosine lấy max qua 12 phép xoay (OTI). Không trừ trung bình thì chroma (luôn không âm) cho cosine cao sẵn giữa hai bài bất kỳ — nhiễu trắng từng đạt 0.985 > τCover.
- **Ngưỡng `τCover` phụ thuộc thang điểm của descriptor VÀ quy mô chỉ mục**: đổi descriptor hoặc đổi số bài trong chỉ mục thì bắt buộc dựng lại `cover_descriptors.npy` (`scripts/build_cover_index.py`, mọi bản ghi có audio thật) và chạy lại EXP-07 (có nhiễu làm mẫu âm) trước khi tin ngưỡng.
- **Cắt truy vấn theo nhiều hệ số nhịp độ** (`config.COVER_TEMPO_FACTORS`, mặc định 0.90–1.10 — dải của §11): reference là 30 giây đầu ở nhịp gốc co về 64 khung, nên chỉ khớp khi đoạn truy vấn chứa TRỌN đúng phần nội dung đó. Truy vấn nhịp f chứa nó trong `30 / f` giây đầu; `cover_service.query_descriptors` dựng một descriptor cho mỗi độ dài và lấy max theo từng ứng viên, evidence ghi `tempo_factor` đã thắng. Trước đây cắt cố định 30 s: tempo 0.90 đúng @1 chỉ 27%, nay 100%. Kiểm lại (kèm nhịp ngoài bảng hệ số): `experiments/exp07_cover/query_span_check.py --off-grid`.
- **`τCover` hiện tại = 0.90**, hiệu chỉnh lại sau khi đổi cách cắt, trên chỉ mục 24.375 bài đúng điều kiện server với `HeldOutProtocol`: Precision 0.9952, Recall 0.7663, nhận nhầm bài ngoài CSDL 0,11%, mẫu nhiễu cao nhất 0.7347; đúng @1 88,8%. **Đổi dải hệ số nhịp độ thì bắt buộc chạy lại EXP-07** — mỗi độ dài cắt thêm là thêm một cơ hội nhận nhầm.
- **Giới hạn còn lại**: đoạn truy vấn ngắn hơn 30 s nội dung (cắt 10/15 s) lệch trục thời gian với reference — đúng @1 chỉ 2% (so với reference cắt cùng khoảng thì điểm 1.000). Nhóm này Chromaprint đã bắt 100%, nên không phải điểm mù của cascade.
- **Phương án nâng cao**: Sử dụng model chuyên dụng như CoverHunter (hoặc tương đương) dưới dạng **pretrained inference**. Tuyệt đối **không huấn luyện từ đầu (train from scratch)**.

---

## 5. Track / Composition Resolver

- Nhận diện `recording_id` từ Fingerprint hoặc Embedding.
- Truy vấn bảng `recordings` và `compositions` để lấy metadata đầy đủ.
- Lấy toàn bộ bản ghi pháp lý liên quan từ bảng `rights`.
- Đóng gói thành **Evidence Object** trước khi chuyển qua Rule Engine.
