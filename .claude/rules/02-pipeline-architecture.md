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
- **`τCover` hiện tại = 0.90**, hiệu chỉnh trên chỉ mục 24.375 bài đúng điều kiện server (30 giây đầu của truy vấn): Precision 0.9946, nhận nhầm bài ngoài CSDL 0,4%, mẫu nhiễu cao nhất chỉ 0.7278. Ở quy mô này tầng Cover nhận đúng 83% truy vấn ở vị trí 1 và bắt đúng lượng dịch cao độ (OTI) 91% — đúng chỗ Chromaprint được 0%.
- **Phương án nâng cao**: Sử dụng model chuyên dụng như CoverHunter (hoặc tương đương) dưới dạng **pretrained inference**. Tuyệt đối **không huấn luyện từ đầu (train from scratch)**.

---

## 5. Track / Composition Resolver

- Nhận diện `recording_id` từ Fingerprint hoặc Embedding.
- Truy vấn bảng `recordings` và `compositions` để lấy metadata đầy đủ.
- Lấy toàn bộ bản ghi pháp lý liên quan từ bảng `rights`.
- Đóng gói thành **Evidence Object** trước khi chuyển qua Rule Engine.
