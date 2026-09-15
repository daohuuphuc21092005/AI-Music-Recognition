---
name: music-rights-pipeline
description: Playbook vận hành, kiểm thử và xử lý sự cố Cascade Pipeline (FFmpeg Preprocessing -> Chromaprint fpcalc -> MERT-v1-95M Deep Embedding -> FAISS/Qdrant Retrieval -> Rights Resolver).
---

# Skill: Cascade Music Rights Pipeline (`music-rights-pipeline`)

Skill này hướng dẫn quy trình chuẩn hóa để xây dựng, kiểm thử, tích hợp và gỡ lỗi (debug) cho **Cascade Audio Processing Pipeline** trong dự án AI Phân loại Nhạc Bản quyền.

---

## 1. Kiến trúc Cascade Nối tiếp (Execution Flow)

```
[Input File] 
    │
    ▼
(AudioService) ── FFmpeg trích xuất Mono WAV
    │
    ▼
(Audio Normalization)
    ├─ Chromaprint preprocessing (fpcalc)
    └─ MERT preprocessing (24,000 Hz, L2 Norm)
    │
    ▼
(FingerprintService) ── Tính score Chromaprint
    │
    ├── [score ≥ τFP] ──────────────────────────┐ (Exact Match)
    │                                           │
    └── [score < τFP]                           │
            │                                   │
            ▼                                   ▼
      (EmbeddingService)               (Track Resolver)
      MERT-v1-95M Inference                     │
            │                                   ▼
            ▼                            (Rights Lookup)
      (RetrievalService)                        │
      FAISS / Qdrant Top-K                      ▼
            │                             (Rule Engine)
            └── [similarity ≥ τMERT] ───────────┘
```

> [!IMPORTANT]
> **Quy tắc Cascade bất biến**: Tầng MERT Embedding chỉ được kích hoạt khi Chromaprint không đạt ngưỡng `τFP`. Tuyệt đối không chạy song song MERT cho các bài hát đã khớp chính xác với Fingerprint.

---

## 2. Quy trình Thực thi Chuẩn (Step-by-Step Execution)

### Bước 1: Tiền xử lý âm thanh (Audio Ingestion & Normalization)
1. Xác thực định dạng tệp: hỗ trợ `.mp3`, `.wav`, `.m4a`, `.mp4`, `.mov`.
2. Dùng FFmpeg chuyển đổi video/audio sang chuẩn WAV tạm:
   ```bash
   ffmpeg -i input_file -vn -acodec pcm_s16le -ac 1 -ar 24000 temp_mert.wav
   ```
3. Cắt đoạn (Segmentation): Chia các cửa sổ 10s, 15s hoặc 30s với overlap 50%.

### Bước 2: Trích xuất Dấu vân tay (Chromaprint / fpcalc)
1. Gọi `fpcalc` trích xuất raw fingerprint và thời lượng:
   ```bash
   fpcalc -json temp_audio.wav
   ```
2. So khớp hash bitwise (Bit error rate / raw match score) với bảng `fingerprints`.
3. So sánh với ngưỡng `τFP`:
   - Nếu `score ≥ τFP`: Gán `match_type = EXACT_MATCH`, chuyển thẳng tới Rights Lookup.
   - Nếu `score < τFP`: Ghi log và chuyển tiếp sang Tầng 3.

### Bước 3: Trích xuất Vector MERT (Deep Music Embedding)
1. Tải model `m-a-p/MERT-v1-95M` với tham số `torch_dtype=torch.float32` (hoặc `float16` trên GPU).
2. Kiểm tra sampling rate đầu vào: bắt buộc **24,000 Hz**.
3. Áp dụng chiến lược Pooling:
   - **P1 (Mean Pooling)**: Lấy trung bình embedding qua các hidden frames.
   - **P2 (Mean + Std Pooling)**: Ghép mean vector và std vector (chiều dài gấp đôi: 1536).
4. Chuẩn hóa vector: Thực hiện L2 Normalization (`vector / norm(vector)`).

### Bước 4: Tìm kiếm Tương đồng (Vector Retrieval)
1. Truy vấn cơ sở dữ liệu vector:
   - **FAISS**: Dùng `IndexFlatIP` (Inner Product trên vector đã L2-normalize = Cosine Similarity).
   - **Qdrant**: Dùng metric `Cosine`.
2. Lấy danh sách Top-K ứng viên (K=5 hoặc K=10).
3. Lọc ứng viên có `similarity ≥ τMERT`.

### Bước 5: Truy vấn Quyền & Đóng gói Evidence
1. Truy vấn thông tin pháp lý từ bảng `recordings`, `compositions`, và `rights`.
2. Đóng gói đối tượng **Evidence Object** chuẩn bị chuyển sang Rule Engine.

---

## 3. Checklist Gỡ lỗi Phổ biến (Troubleshooting Guide)

| Hiện tượng | Nguyên nhân gốc | Cách khắc phục |
|---|---|---|
| MERT trích xuất vector không chính xác | Audio không đúng sample rate 24kHz | Kiểm tra lại lệnh resample của FFmpeg hoặc `torchaudio.transforms.Resample(orig_sr, 24000)` |
| Lỗi CUDA Out of Memory (OOM) | Batch size trích xuất embedding quá lớn | Giảm batch size xuống 2-4 hoặc chuyển sang `device="cpu"` trong `.env` |
| Chromaprint không nhận diện được audio nén | Audio bị méo tần số hoặc cắt dải cao | Chuyển tiếp sang tầng MERT Retrieval theo đúng thiết kế cascade |
| Trả về traceback ra API client | Thiếu try/catch ở tầng Controller | Bao bọc service call bằng HTTPException với mã lỗi chuẩn (`MODEL_FAILURE`, `TIMEOUT`) |
| Quên giải phóng file tạm | AudioService không có hook cleanup | Sử dụng `tempfile.NamedTemporaryFile` hoặc `finally: os.remove(temp_path)` |
| Tầng 1 báo `UNAVAILABLE`, `reason_code=DATABASE_UNAVAILABLE` | Server không kết nối được PostgreSQL (vd. `DATABASE_URL` trỏ cổng 5433 của Docker cũ trong khi PostgreSQL chạy ở 5432) | Sửa `DATABASE_URL`; kiểm tra `/health` → `components.database` |
| Tầng 1 báo `UNAVAILABLE`, `reason_code=FPCALC_MISSING` | Tiến trình server không thấy `fpcalc` (PATH của server khác PATH của shell) | Đặt `FPCALC_PATH` trong `.env` hoặc để binary ở `~/bin` |
| Nhiễu / tạp âm bị trả `COVER_MATCH` | Descriptor chroma không trừ trung bình → cosine giữa hai bài bất kỳ đã cao sẵn | Dùng `cover_service.normalize_frames`; dựng lại `cover_descriptors.npy` và chạy lại EXP-07 (có mẫu nhiễu) để hiệu chỉnh τCover |
| Tầng 1 trả `EXACT_MATCH` cho bài không có audio | Fingerprint không sinh từ audio thật lọt vào bảng (gộp dữ liệu chỉ có metadata) | `python scripts/check_data_integrity.py` — mục "Fingerprint là đầu ra thật của fpcalc" phải PASS |
| Badge HIGH/CONDITIONAL từ giấy phép do model đoán | Thiếu cổng độ tin cậy dữ liệu quyền | `rights_gate` trong `configs/rules_v1.yaml` hạ về UNKNOWN; kết luận tạm ở `evidence.rule_engine.provisional_decision` |
| Mở rộng corpus xong nhưng ngưỡng không còn đúng | τFP / τMERT / τCover phụ thuộc quy mô reference | Chạy lại EXP-01 / EXP-06 / EXP-07 rồi cập nhật `.env`, `backend/config.py` và `min_identity_confidence_by_match_type` trong `rules_v1.yaml` cùng lúc |
