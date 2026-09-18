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
   - `search_fingerprint` KHÔNG quét cả bảng: lọc `FP_PREFILTER_TOP_K` (50) bản ghi nhiều hash trùng tuyệt đối nhất rồi chấm đầy đủ; quay về quét toàn bộ khi điểm trong ±0.15 quanh ngưỡng hoặc truy vấn < 10 s. Đối chứng 1.900 truy vấn: 0 lệch quyết định, 110 ms thay vì ~4,5 s. `evidence.fingerprint.prefilter.full_scan_reason` cho biết đã đi đường nào.
   - Hiệu chỉnh τFP (EXP-01) vẫn quét đầy đủ — cần phân bố điểm của mọi bản ghi.
3. So sánh với ngưỡng `τFP`:
   - Nếu `score ≥ τFP`: Gán `match_type = EXACT_MATCH`, chuyển thẳng tới Rights Lookup.
   - Nếu `score < τFP`: Ghi log và chuyển tiếp sang Tầng 3.

### Bước 3: Trích xuất Vector MERT (Deep Music Embedding)
1. Tải model `m-a-p/MERT-v1-95M` với tham số `torch_dtype=torch.float32` (hoặc `float16` trên GPU).
   - Thiết bị do `embedding_service.get_device()` quyết định: tự dò CUDA, ép bằng biến môi trường `DEVICE=cpu|cuda`. Đo trên GTX 1650 4 GB: **~1 s/bài trên GPU so với ~4 s/bài trên 6 luồng CPU** (phần giải mã audio vẫn nằm ở CPU nên đó là trần tốc độ).
   - Dựng embedding cho corpus lớn: gọi `python scripts/build_embeddings.py --all --limit 400 --no-rebuild` LẶP LẠI trong một vòng lặp ngoài (mã thoát `10` = còn việc, `0` = đã ghi CSV xong). Tiến trình MERT phình ~3 MB mỗi file, nên phải thoát hẳn sau mỗi lượt để hệ điều hành thu hồi bộ nhớ.
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
| Bản chậm (tempo < 1) không qua tầng Cover dù dịch cao độ vẫn bắt được | Reference là 30 s đầu co về 64 khung; cắt truy vấn cố định 30 s làm bản chậm mất phần cuối (tempo 0.90: đúng @1 27%) | `COVER_TEMPO_FACTORS` (mặc định 0.90–1.10) cắt truy vấn theo từng hệ số rồi lấy max: tempo 0.90 lên 100% @1. Đổi dải hệ số thì chạy lại EXP-07 — thêm độ dài cắt là thêm cơ hội nhận nhầm. Kiểm nhanh: `experiments/exp07_cover/query_span_check.py --off-grid` |
| Nhiễu / tạp âm bị trả `COVER_MATCH` | Descriptor chroma không trừ trung bình → cosine giữa hai bài bất kỳ đã cao sẵn | Dùng `cover_service.normalize_frames`; dựng lại `cover_descriptors.npy` và chạy lại EXP-07 (có mẫu nhiễu) để hiệu chỉnh τCover |
| Tầng 1 trả `EXACT_MATCH` cho bài không có audio | Fingerprint không sinh từ audio thật lọt vào bảng (gộp dữ liệu chỉ có metadata) | `python scripts/check_data_integrity.py` — mục "Fingerprint là đầu ra thật của fpcalc" phải PASS |
| Badge HIGH/CONDITIONAL từ giấy phép do model đoán | Thiếu cổng độ tin cậy dữ liệu quyền | `rights_gate` trong `configs/rules_v1.yaml` hạ về UNKNOWN; kết luận tạm ở `evidence.rule_engine.provisional_decision` |
| Mở rộng corpus xong nhưng ngưỡng không còn đúng | τFP / τMERT / τCover phụ thuộc quy mô reference | Chạy lại EXP-01 / EXP-06 / EXP-07 rồi cập nhật `.env`, `backend/config.py` và `min_identity_confidence_by_match_type` trong `rules_v1.yaml` cùng lúc |
| `CUDA error: an illegal memory access` giữa lượt dựng embedding | GPU Max-Q trục trặc nhất thời dưới tải dài (cùng file đó chạy lại bình thường) | Lỗi CUDA làm hỏng context cả tiến trình, bắt ngoại lệ rồi chạy tiếp là vô ích: `build_embeddings.py` ghi file lỗi vào checkpoint rồi thoát mã `10` để vòng lặp ngoài mở tiến trình mới. Cuối đợt dựng lại các bài đó với `DEVICE=cpu` |
| Ổ C cạn sau khi `docker compose build` (image backend ~9,7 GB) | Đĩa ảo `%LOCALAPPDATA%\Docker\wsl\disk\docker_data.vhdx` chỉ phình, không tự co (không sparse); đĩa dữ liệu trong máy ảo gắn KHÔNG có `discard`, nên xoá image/cache bên trong không trả chỗ cho Windows. Lần build lại sau khi máy khởi động lại còn mất cả cache lớp pip → tải lại torch, file phình tới 42,6 GB | 1) `docker builder prune -a -f`; 2) `wsl -d docker-desktop -u root -e fstrim -v /mnt/docker-desktop-disk` (bắt buộc — thiếu bước này diskpart không thu được gì); 3) `docker desktop stop`, thoát hẳn app Docker Desktop (nó tự bật lại và giữ file → diskpart báo "already attached"), `wsl --shutdown`, kiểm không còn `vmmemWSL`; 4) diskpart quyền quản trị: `select vdisk file=...docker_data.vhdx` / `attach vdisk readonly` / `compact vdisk` / `detach vdisk`. Lần 2026-09-18: 42,6 → 14,5 GB, CSDL nguyên vẹn |
| Tác vụ nền bị dừng với lý do thiếu RAM | Cơ chế bảo vệ của Claude Code cắt MỌI tác vụ nền khi RAM trống xuống thấp, không phân biệt to nhỏ | Chạy đúng MỘT việc nặng một lúc; việc dài nhiều giờ thì `Start-Process ... -WindowStyle Hidden` để tách khỏi phiên. Tiến độ luôn nằm trong checkpoint nên chạy lại là tiếp tục |
| `pip install torch==<ver>` báo "Requirement already satisfied" dù đang là bản CPU | Biến thể `+cpu` và `+cu130` cùng số hiệu phiên bản, pip coi là đã có | Chỉ đích danh biến thể: `pip install --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps "torch==<ver>+cu130"` |
