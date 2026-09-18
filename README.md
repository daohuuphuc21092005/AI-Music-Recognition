# 🎵 MUSIC RECOGNITION & RIGHTS INSPECTION PLATFORM

> Hệ thống AI nhận diện âm nhạc đa tầng (Hybrid Cascade) kết hợp Audio Fingerprinting và Deep Retrieval 768 chiều, phục vụ tra cứu bản quyền tự động.

---

## 📌 Tổng quan kiến trúc

```
INPUT (audio/video)
  → Tầng 1: Chromaprint fingerprint      → score ≥ τFP    → EXACT_MATCH
  → Tầng 2: MERT-v1-95M + FAISS (cosine) → sim   ≥ τMERT  → NEAR_MATCH
  → Tầng 3: chroma CQT + OTI (cover)     → sim   ≥ τCover → COVER_MATCH
                                          → dưới cả ba    → UNKNOWN (cần người duyệt)
  → Tra cứu rights → Rule Engine → rights_gate → Đánh giá rủi ro → Evidence
```

- **Tầng 1 (Exact Match):** Chromaprint. Fingerprint được giải nén bằng
  `backend/services/chromaprint_codec.py` (thuần Python, không cần thư viện native)
  và so khớp bằng thuật toán bit-error của Chromaprint đã vector hoá bằng numpy.
  Chỉ chấm đầy đủ 50 bản ghi có nhiều hash trùng nhất (quay về quét toàn bộ khi
  điểm sát ngưỡng): đối chứng 1.900 truy vấn 0 lệch quyết định, ~110 ms thay vì ~4,5 s.
- **Tầng 2 (Deep Retrieval):** MERT-v1-95M + FAISS `IndexFlatIP` trên vector đã
  chuẩn hoá L2. Điểm được **gộp theo `recording_id`** nên Top-K là K bản ghi khác
  nhau, không phải K đoạn của cùng một bài.
- **Tầng 3 (Cover):** `backend/services/cover_service.py`, chroma từ CQT + Optimal
  Transposition Index, tìm trên toàn bộ chỉ mục cover. Bất biến với dịch cao độ vì
  dịch k bán cung chỉ là xoay vòng vector chroma đi k bậc — đúng nhóm biến đổi mà
  Chromaprint và MERT cùng bó tay.
- **rights_gate:** kết luận dựa trên dữ liệu quyền kém tin cậy (giấy phép do mô
  hình SUY ĐOÁN, metadata MÔ PHỎNG chưa xác minh) bị hạ về `UNKNOWN`; kết luận tạm
  vẫn nằm trong evidence.
- **Backend:** FastAPI + PostgreSQL.

> Ba ngưỡng dưới đây hiệu chỉnh trên **corpus 24.375 bản ghi có audio thật
> (48.750 vector MERT), 1.900 truy vấn biến đổi từ 100 bài nguồn**.
>
> **τFP = 0.30** — EXP-01: trên 800 truy vấn không làm biến dạng trục thời gian/cao
> độ (cắt đoạn, nén MP3, nhiễu SNR 20, gain, EQ) **Precision = Recall = 1.0**; toàn
> tập Precision 0.9981, nhận nhầm bài ngoài CSDL **0/1.900** (`HeldOutProtocol`).
> Quy tắc "F1 cao nhất trong nhóm giữ FPR ≤ 0.005" nay đề xuất 0.10, nhưng **giữ
> 0.30** theo quyết định của chủ dự án: ở cấp hệ thống F1 chỉ +0.0016 mà nhận sai
> tăng 5 → 8 và bộ lọc hash phải quét toàn bộ 44% truy vấn (xem mục Nghiệm thu §13).
>
> **τMERT = 0.98** — EXP-06: False Match Rate **2,65%** (≤ 5% của §13). Chính τ = 0.97
> cũ cho FMR 6,12% ở quy mô này, dù chỉ 4,40% khi corpus còn 4.000 bản ghi.
>
> **τCover = 0.90** — EXP-07 đúng điều kiện server (cắt truy vấn theo từng hệ số nhịp
> độ 0.90–1.10, tìm trên cả 24.375 bài): Precision 0.9952, Recall 0.7663, FMR 0,11%,
> không nhận mẫu nhiễu nào (nhiễu cao nhất 0.7347).
>
> ⚠️ **Cả ba ngưỡng phụ thuộc QUY MÔ reference.** Cùng bộ truy vấn, τCover = 0.70
> cho FMR 4,9% trên 100 bài nhưng 17,4% trên 24.375 bài. **Mở rộng dữ liệu ⇒ bắt
> buộc hiệu chỉnh lại**, và phải sửa đồng thời `.env` lẫn `configs/rules_v1.yaml`
> — `scripts/run_all_experiments.py` dừng lại nếu hai nơi lệch nhau.
>
> ⚠️ **τFP còn phụ thuộc ĐỘ DÀI truy vấn.** Đoạn ngắn có ít offset để dò nên dễ gặp
> một offset "may mắn" — nhiễu trắng 8 giây từng đạt ~0.26. `min_score_for_duration()`
> tự nâng ngưỡng cho truy vấn ngắn (chỉ nâng, không bao giờ hạ dưới τFP).
>
> **Cổng định danh của Rule Engine tách ngưỡng theo TỪNG TẦNG** (`EXACT_MATCH: 0.30`,
> `NEAR_MATCH: 0.98`, `COVER_MATCH: 0.90`) chứ không dùng một con số chung — điểm
> Chromaprint, cosine MERT và cosine chroma **không cùng thang đo**.

---

## 🚀 Cài đặt

### 1. Môi trường Python

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. fpcalc (Chromaprint) — bắt buộc cho tầng 1

`fpcalc` **không cài được bằng pip**.

- **Windows:** tải `chromaprint-fpcalc-*-windows-x86_64.zip` tại
  <https://github.com/acoustid/chromaprint/releases>, giải nén và đặt `fpcalc.exe`
  vào một thư mục nằm trong `PATH` (ví dụ `%USERPROFILE%\bin`).
  *conda-forge không có gói `chromaprint` cho win-64.*
- **Linux/Docker:** `apt-get install -y libchromaprint-tools` (đã có sẵn trong Dockerfile).

Kiểm tra: `fpcalc -version`

### 3. FFmpeg — chỉ cần khi xử lý VIDEO

File audio (mp3/wav/flac/m4a) được `librosa` giải mã và resample trực tiếp,
không cần FFmpeg. Thiếu FFmpeg thì `/health` báo `ffmpeg: MISSING (chi audio)`
và hệ thống từ chối file video bằng mã lỗi rõ ràng.

### 4. Cấu hình `.env`

```bash
copy .env.example .env      # Windows
```

Mở `.env` và chọn một profile:

```ini
# A. PostgreSQL cài trực tiếp trên máy
DATABASE_URL=postgresql://postgres:<MAT_KHAU>@localhost:5432/music_rights_ai

# B. PostgreSQL bằng Docker Compose (cổng 5433 để không đụng bản cài sẵn)
# DATABASE_URL=postgresql://postgres:<MAT_KHAU>@localhost:5433/music_rights_ai
# POSTGRES_PASSWORD=<MAT_KHAU>          # bắt buộc, trùng mật khẩu ở dòng trên
```

---

## ▶️ Khởi chạy

```bash
# 1. Dựng lại FAISS index + bản đồ ID từ embeddings_master.csv
python scripts/rebuild_faiss_index.py

# 2. Tạo bảng và nạp dữ liệu tham chiếu vào PostgreSQL
python init_db.py

# 3. Kiểm tra toàn vẹn dữ liệu (khoá ngoại, số chiều vector, khớp index/ID map)
python scripts/check_data_integrity.py

# 4. Chạy server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: <http://localhost:8000/docs> · Health: <http://localhost:8000/health>

**Làm nóng lúc khởi động** (`WARMUP_ON_STARTUP`, mặc định bật): server trả lời ngay,
một luồng nền giải mã fingerprint tham chiếu, nạp MERT + suy luận thử và nạp chỉ mục
Cover; `/health` báo tiến độ ở khoá `warmup`. Đo 2026-09-18 (24.375 bản ghi, GTX 1650):
làm nóng ~24 s (fingerprint 19,7 s · MERT 3,1 s · Cover 1,5 s), sau đó truy vấn khớp
tầng 1 đầu tiên mất **112 ms thay vì 21,1 s**, MERT lần đầu 1,1 s thay vì 4,6 s. Truy
vấn gửi TRONG lúc làm nóng chờ bộ đệm đang nạp (có khoá) chứ không nạp lần hai.

### Chạy bằng Docker (backend + CSDL)

`.env` trên máy host cần `POSTGRES_PASSWORD` và `AUDIO_ROOT` (thư mục audio corpus,
đường dẫn tuyệt đối) — compose dừng ngay nếu thiếu, không có mật khẩu mặc định.

```bash
docker compose up -d --build        # API: http://127.0.0.1:8000 · CSDL: 127.0.0.1:5433
docker compose ps                   # backend chuyển "healthy" khi /health báo ONLINE
```

Khi volume CSDL còn rỗng (máy mới), nạp dữ liệu tham chiếu từ `data/processed/`:
`docker compose run --rm backend python init_db.py`.

- **Không có gì nhạy cảm trong image**: `.dockerignore` loại `.env*`, `data/`, audio
  test. `data/` mount chỉ đọc; audio corpus mount từ `AUDIO_ROOT` vào `/audio`.
- **Chạy bằng user thường** (`app`, uid 10001); mã nguồn thuộc root, chỉ
  `temp_uploads/` và cache model ghi được. Cổng API và CSDL chỉ mở trên `127.0.0.1`.
- **GPU**: torch trên PyPI cho Linux là bản CUDA nên image nặng vài GB; compose xin
  một GPU NVIDIA (`deploy.resources`). Máy không có GPU thì xoá khối `deploy` —
  `DEVICE=auto` tự chạy CPU.
- **MERT ghim revision** (`MERT_MODEL_REVISION`): model dùng `trust_remote_code`, không
  ghim thì container mới tải bản mới nhất cả trọng số lẫn mã, embedding lệch chỉ mục
  mà không báo lỗi. Trọng số tải một lần vào volume `hf_cache`.
- Tên project compose cố định `music-recognition-main` để dùng lại đúng container và
  volume CSDL đã có; đổi tên là compose tạo một CSDL rỗng mới. Container CSDL cũ được
  tạo với `restart: always`, nên lần `docker compose up` đầy đủ đầu tiên sẽ tạo lại nó
  (dữ liệu nằm trong volume, không mất); chỉ bật backend thì dùng `--no-deps`.

**Đã kiểm chứng (2026-09-17, GTX 1650):** image `music-rights-ai-backend` 9,7 GB, build
context 98 KB; trong image không có `.env`, `data/`, `.git`; tiến trình chạy bằng `app`
(uid 10001), không ghi được vào mã nguồn; `torch.cuda.is_available()` = True trong
container. `/health` ONLINE (CSDL qua hostname `db`, FAISS 48.750 vector, fpcalc,
ffmpeg, Rule Engine, chỉ mục Cover). Truy vấn tempo 0.90 qua `POST /api/v1/search`:
`COVER_MATCH` đúng bài nguồn, `tempo_factor` 0.9, 2,4 s. Truy vấn **đầu tiên** mất ~2
phút vì tải MERT (đúng revision ghim) và dựng bộ đệm fingerprint — đo TRƯỚC khi có làm
nóng lúc khởi động; container chưa được đo lại (lần tải MERT đầu tiên vẫn mất thời
gian, nhưng nay nằm trong luồng làm nóng thay vì trong truy vấn của người dùng).

---

## 🔌 API

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/health` | Trạng thái từng thành phần: database, faiss, chromaprint, ffmpeg, mert, rule_engine, cover · tiến độ làm nóng (`warmup`) |
| `POST` | `/api/v1/analyze` | Upload + `platform` / `commercial_use` / `monetization` → trả `job_id` (202) |
| `GET` | `/api/v1/jobs/{job_id}` | `QUEUED` / `PROCESSING` / `DONE` / `FAILED` |
| `GET` | `/api/v1/results/{job_id}` | Kết quả đầy đủ: identity, match, rights, assessment, evidence |
| `GET` | `/api/v1/tracks/{recording_id}` | Tra cứu bản ghi + quyền |
| `POST` | `/api/v1/feedback` | `Correct` / `Incorrect` / `Unsure` |
| `POST` | `/api/v1/search` | Bản đồng bộ của `/analyze` (tiện thử nhanh) |

Lỗi luôn trả về mã chuẩn hoá, **không bao giờ trả traceback**:
`FILE_TOO_LARGE`, `UNSUPPORTED_FORMAT`, `NO_AUDIO`, `NO_MUSIC`, `MODEL_FAILURE`,
`DATABASE_FAILURE`, `TIMEOUT`, `UNKNOWN_TRACK`, `LOW_CONFIDENCE`, `INVALID_REQUEST` (tham số sai,
422), `INTERNAL_ERROR` (lỗi không lường trước, 500).

### Rule Engine

Phân loại 5 nhóm bản quyền bằng **logic tường minh** đọc từ `configs/rules_v1.yaml`,
theo đúng thứ tự ưu tiên cố định:

```
Audio Library → Creator Music → Content ID → Creative Commons → Public Domain → Uncategorized
```

Trước đó là **cổng định danh**: chưa nhận diện đủ tin cậy thì không có quyền nào
để xét, phải trả `UNKNOWN` + `HUMAN_REVIEW_REQUIRED` — không bao giờ ép thành
`LOW` hay `HIGH`. Ngưỡng của cổng này lấy **theo tầng đã sinh ra kết quả**
(`min_identity_confidence_by_match_type` trong `rules_v1.yaml`), đúng bằng τFP và
τMERT, để cascade và Rule Engine không nói hai chuyện khác nhau. Mỗi kết quả kèm **ba loại độ tin cậy tách biệt**:
`identity_confidence`, `rights_confidence`, `decision_confidence`.

---

## 🖥️ Giao diện web

Mở <http://localhost:8000/> sau khi chạy server — frontend được **FastAPI phục vụ
tĩnh** từ thư mục `frontend/`, không cần web server riêng và **không có bước build**
(dự án chỉ build Python; thêm bundler chỉ để tô màu là cái giá không đáng trả).

Bốn màn hình theo §13:

| Màn hình | Nội dung |
|---|---|
| **1. Upload** | Kéo–thả hoặc chọn file · Nền tảng · Mục đích (phi thương mại / thương mại) · Bật kiếm tiền. Ba tuỳ chọn này là **đầu vào thật của Rule Engine**, không phải trang trí. Máy chủ thiếu FFmpeg (theo `/health`) thì chọn file video bị báo ngay và khoá nút phân tích, thay vì tải lên rồi mới nhận `MODEL_FAILURE`. |
| **2. Processing** | Bảy bước THẬT của pipeline, tô sáng theo trường `stage` mà backend ghi vào bảng `jobs` — **không có thanh chờ giả, không có "AI is thinking"**. Xong bước nào hiện độ trễ đo được của bước đó. |
| **3. Result** | Nhận diện · **căn cứ nhận diện** (điểm từng tầng so với ngưỡng của CHÍNH tầng đó — điểm Chromaprint, cosine MERT và điểm Cover không cùng thang) · Quyền & giấy phép (giấy phép do mô hình **suy đoán** hiện trong thẻ viền nét đứt, mỗi giá trị kèm "(suy đoán)", không tô xanh/đỏ như quyền tra được) · Mức rủi ro 🟢/🟡/🔴/⚪ · Khuyến nghị · **ba thanh độ tin cậy tách biệt** (identity / rights / decision; chưa định danh được thì thanh identity mờ đi và ghi rõ, vì đó là điểm cao nhất DƯỚI ngưỡng) · nút phản hồi Correct / Incorrect / Unsure. |
| **4. Evidence** | Điểm fingerprint và ngưỡng hiệu dụng (ghi rõ khi bị nâng cho truy vấn ngắn) · tầng 1 lọc theo hash hay quét toàn bộ, và vì sao · bảng Top-K của MERT · tầng Cover: lệch cao độ quy từ OTI (OTI 11 = truy vấn cao hơn 1 bán cung) và hệ số nhịp của đoạn cắt thắng · luật Rule Engine đã kích hoạt · nguồn metadata quyền và ngày xác minh · PD của tác phẩm và PD của bản thu **để riêng** (§2) · độ trễ từng bước · JSON gốc. |

Đã kiểm bằng ảnh chụp Chrome headless trên server thật (2026-09-18): desktop 1280 px,
điện thoại 390 px (không tràn ngang; nhãn nằm trên giá trị, bảng Top-K vẫn thấy cột
Similarity) và chế độ tối (chữ trên nút màu nhấn đạt tương phản ~7:1).

Để màn hình Processing hiển thị được tiến trình thật, backend có thêm cột
`jobs.stage` và một callback `on_stage` chạy xuyên `analyze_audio` →
`process_music_query`; `GET /api/v1/jobs/{job_id}` trả kèm trường `stage`.
Nếu CSDL đã tạo từ trước, chạy lại `python init_db.py` (hoặc
`ALTER TABLE jobs ADD COLUMN IF NOT EXISTS stage VARCHAR(40);`).

---

## 🔬 Thí nghiệm

```bash
python experiments/exp02_mert/run.py          # Retrieval: Recall@K, MRR, mAP
python experiments/exp06_unknown/run.py       # Unknown detection -> hiệu chỉnh τMERT
python scripts/augment_audio.py --from-db     # Dựng tập truy vấn biến đổi
python experiments/exp01_fingerprint/run.py   # Chromaprint baseline -> hiệu chỉnh τFP
python experiments/exp03_pooling/run.py       # mean vs mean+std pooling
python experiments/exp04_hybrid/run.py        # FP vs MERT vs Cascade (thí nghiệm chính)
python experiments/exp05_robustness/run.py    # Tổng hợp độ bền theo phép biến đổi
python experiments/exp07_cover/run.py         # Chroma/CQT + OTI -> hiệu chỉnh τCover
python experiments/exp08_end_to_end/run.py    # Trọn pipeline -> Macro-F1, confusion matrix
python experiments/exp09_license_learnability/run.py  # Bản quyền có học được từ âm thanh?
```

Hoặc chạy cả chuỗi đúng thứ tự phụ thuộc — một lệnh đi qua bốn pha: **hiệu chỉnh**
(EXP-01, 06, 03, 07) → ghi ngưỡng vào `.env` → **hỏi lại `backend.config` trong
tiến trình mới** và so với `identity_gate` của `configs/rules_v1.yaml` (lệch thì
dừng) → **tiêu thụ** (EXP-02, 04, 05, 08, sweep license):

```bash
python scripts/run_all_experiments.py
python scripts/run_all_experiments.py --only exp04,exp05   # một phần, vẫn đúng thứ tự
```

Bước hỏi lại tồn tại vì một lỗi có thật: `.env` từng có khoá trùng nên script báo
"đã ghi" mà runtime vẫn chạy ngưỡng cũ suốt một ngày.

**Giao thức "bài không có trong CSDL" (held-out)** dùng chung
`experiments.common.HeldOutProtocol` và chạy trên đúng đường production qua tham
số `exclude_recording_ids` của cascade. Nó loại bản trùng fingerprint, bản gần trùng
(ví dụ "Digg It!" bản 12" và 7" của cùng nghệ sĩ) và — với truy vấn `audio_overlay` —
cả bài bị trộn chồng, vì bài đó cũng nằm trong CSDL: thiếu bước này thì 2/4 "nhận
nhầm" held-out của EXP-01 thực ra là hệ thống nhận ĐÚNG.

EXP-03/04/08 chạy hàng giờ nên có **checkpoint**: bị ngắt giữa chừng (máy hết RAM
chẳng hạn) thì chạy lại sẽ tiếp tục từ đúng chỗ cũ. Chữ ký cấu hình nằm ở dòng đầu
file checkpoint — đổi ngưỡng hay đổi tập truy vấn thì checkpoint cũ tự bị vứt, chứ
không trộn kết quả của hai cấu hình vào một bảng.

Kết quả lưu ở `experiments/results/*.json` kèm `git_commit`, phiên bản dữ liệu,
tham số và ngưỡng (§14).

### Trạng thái 9 thí nghiệm (§15 yêu cầu 8; EXP-09 phát sinh từ một câu hỏi thiết kế)

Số liệu EXP-01 → EXP-08 đo trên **corpus FMA thật ở quy mô hiện tại**: 24.375 bản
ghi có audio, 48.750 vector MERT, chỉ mục cover 24.375 bài; 1.900 truy vấn biến
đổi từ 100 bài nguồn (EXP-08 thêm 190 truy vấn từ 10 bài CC0). Ngưỡng đúng cấu
hình server: τFP 0.30 · τMERT 0.98 · τCover 0.90; Tầng 1 lọc ứng viên theo hash,
tầng Cover cắt truy vấn theo hệ số nhịp độ 0.90–1.10.

| # | Thí nghiệm | Kết quả chính |
|---|---|---|
| EXP-01 | Fingerprint Baseline | **τFP = 0.30**: 800 truy vấn sạch P = R = 1.000; cả 1.900 truy vấn P 0.9981 · nhận nhầm held-out 0,00% |
| EXP-02 | MERT Retrieval | Recall@1 0.8787 · Recall@5 0.9374 · MRR 0.9059 (leave-one-out, truy vấn chưa biến đổi) |
| EXP-03 | Pooling Strategy | `mean+std` chỉ +1,13pp Recall@1 nhưng index ×2 → **giữ `mean`** |
| EXP-04 | FP vs MERT vs Cover vs Cascade | **Cascade production F1 0.9352** (R 0.8805) > Cover 0.8659 > FP→MERT 0.7278 > FP 0.7140 > MERT 0.3915 |
| EXP-05 | Robustness | Nhờ tầng Cover: dịch cao độ 0,25% → **71%**, đổi nhịp 0% → **81%**; còn yếu nhất: chồng âm 65% |
| EXP-06 | Unknown Detection | → chốt **τMERT = 0.98** (FMR 2,65%) |
| EXP-07 | Cover Identification | → chốt **τCover = 0.90** (P 0.9952 · R 0.7663 · FMR 0,11%); dịch cao độ và đổi nhịp 0.90–1.10 đúng @1 100% |
| EXP-08 | End-to-End | Macro-F1 **0.9518** · FMR held-out **0,00%** · nhận đúng bài có trong CSDL 87,9% |
| EXP-09 | Học bản quyền từ âm thanh? | Tín hiệu rất yếu, không dùng được (đo ở corpus 4.000) |

### Nghiệm thu §13 — đạt cả bốn tiêu chí

| Tiêu chí | Mục tiêu | Đo được | Nguồn | |
|---|---|---|---|---|
| Clean exact-match | P ≥ 0.95, R ≥ 0.90 | **P 1.000 · R 1.000** trên 800 truy vấn (30 s, cắt 10/15 s, MP3 128k/64k, EQ, gain, nhiễu SNR 20) | EXP-01 | ✅ |
| Robust retrieval | Recall@5 ≥ 0.80 | **0.9937** cấp hệ thống (MERT ∪ Cover) · MERT đơn lẻ 0.7921 · Cover đơn lẻ 0.8947 | EXP-04 | ✅ |
| Unknown FMR | ≤ 5% | Chromaprint 0,00% · MERT 2,65% · Cover 0,11% · **cả pipeline 0,00%** (760 lượt held-out) | EXP-01/06/07/08 | ✅ |
| End-to-end | Macro-F1 ≥ 0.80 | **0.9518**, đủ 4 lớp có mẫu (LOW 114 · CONDITIONAL 475 · HIGH 171 · UNKNOWN 760) | EXP-08 | ✅ |

**Robust retrieval chấm ở cấp hệ thống — và vì sao phải nói rõ điều này.** MERT đơn
lẻ đạt 0.7921, thiếu 0.008; toàn bộ phần thiếu đến từ dịch cao độ (MERT mã hoá cao
độ tuyệt đối: 0.185, bỏ nhóm đó ra thì 0.954). Chủ dự án quyết định chấm trên hợp
top-5 của MERT và Cover: §13 chỉ ghi "Robust retrieval", tầng Cover là một phần của
cascade production và được thêm vào đúng vì điểm mù này, và cả hai danh sách đều hiện
trong evidence. Mục tiêu 0.80 giữ nguyên. Cách chấm được chọn **sau** khi đã thấy MERT
trượt, nên con số MERT đơn lẻ luôn được báo cạnh bên (biên bản ở
`.claude/rules/06-experiments-evaluation.md` §5).

**Giữ τFP = 0.30 dù quy tắc hiệu chỉnh đề xuất 0.10.** EXP-01 đo lại dưới
`HeldOutProtocol` cho tỉ lệ nhận nhầm của tầng 1 là 0,37% / 0,11% / 0 ở τ 0.10 / 0.15 /
≥ 0.20 (giao thức cũ đếm cả việc nhận ra bài bị trộn chồng là nhận nhầm: 2,26% / 1,63%
/ 0,68%). Quy tắc chỉ nhìn F1 của tầng 1 nên ra 0.10. Mô phỏng cả cascade (điểm quét
đầy đủ của EXP-01 + quyết định MERT/Cover của EXP-04, tái lập đúng EXP-04 ở 0.30):

| τFP | F1 cascade | Nhận sai | Nhận nhầm bài lạ | Bộ lọc hash quét toàn bộ |
|---|---|---|---|---|
| **0.30** | 0.9352 | 5 | 0 | 1,6% |
| 0.20 | 0.9361 | 5 | 0 | 5,4% |
| 0.10 | 0.9368 | 8 | 1 | 44% |

Phần tầng 1 bỏ sót ở 0.30 đã được MERT/Cover bắt lại, nên hạ ngưỡng chỉ đổi an toàn
và tốc độ lấy vài truy vấn. `scripts/apply_calibrated_thresholds.py` nay chỉ tự động
**siết** ngưỡng; hạ ngưỡng phải có `--allow-loosen`.

### EXP-08 — trọn pipeline: đạt ngưỡng nghiệm thu

380 truy vấn từ 20 bài nguồn chọn phân tầng theo giấy phép (3 bài mỗi loại; tập
truy vấn chỉ có 2 bài CC_BY_ND; bài CC0 lấy từ 10 bài sinh thêm), mỗi truy vấn chạy
2 điều kiện (`known`, `held_out`) × 2 ngữ cảnh (phi thương mại, thương mại) =
**1.520 lượt**, đúng đường chạy production.

| Mức rủi ro | Precision | Recall | F1 | Số mẫu |
|---|---|---|---|---|
| LOW | 1.000 | 0.9825 | 0.9912 | 114 |
| CONDITIONAL | 1.000 | 0.8547 | 0.9217 | 475 |
| HIGH | 0.9936 | 0.9064 | 0.9480 | 171 |
| UNKNOWN | 0.8983 | 1.000 | 0.9465 | 760 |

| Điều kiện | Macro-F1 | Accuracy | n |
|---|---|---|---|
| `held_out` (bài ngoài CSDL) | **1.000** | 1.000 | 760 |
| `known` (bài có trong CSDL) | 0.9536 | 0.8855 | 760 |

- **Sai sót gần như chỉ có một kiểu: nhận diện thất bại thì ra `UNKNOWN`** (86/92
  lần). Hệ thống không đoán bừa — Precision của LOW / CONDITIONAL / HIGH là 1.000 /
  1.000 / 0.9936. Trong 92 lần thất bại: 48 dịch cao độ, 30 đổi nhịp, 14 chồng âm.
- **Lỗi mức rủi ro duy nhất không phải `UNKNOWN`**: một truy vấn `audio_overlay` (bài
  CC_BY_ND trộn với bài phi thương mại, dùng thương mại) ra HIGH thay vì CONDITIONAL,
  vì hệ thống nhận ra bài bị trộn chồng. Cả 6 lần nhận ra bản ghi khác ở điều kiện
  `known` đều là bài bị trộn chồng. Với một bản trộn thì HIGH mới là rủi ro thật; giới
  hạn thực sự là pipeline chỉ báo **một** bài cho mỗi truy vấn.
- Độ trễ trung bình 1,4 s (P50 1,7 s, P95 2,5 s) — phần lớn là MERT (~1 s) vì điều
  kiện held-out luôn đi hết ba tầng. Chromaprint P50 41 ms; tra quyền 1,6 ms; Rule
  Engine 0,1 ms.

So với lượt trước trên cùng bộ truy vấn (cắt Cover cố định 30 s, Tầng 1 quét toàn bộ):
Macro-F1 0.9091 → **0.9518**, nhận đúng bài có trong CSDL 80,3% → **87,9%**, số lần
thất bại vì tempo chậm 70 → 12, độ trễ trung bình 5,8 s → **1,4 s**. Lượt này dừng một
lần ở truy vấn 338/380 vì lỗi GPU nhất thời (`CUDA error: an illegal memory access`,
backend trả đúng `MODEL_FAILURE`) và chạy tiếp từ checkpoint.

**Phạm vi — đọc trước khi trích dẫn con số.** Chỉ nhóm CREATIVE_COMMONS và nhánh
fallback đi được bằng audio thật (FMA). AUDIO_LIBRARY, CREATOR_MUSIC,
COMMERCIAL_CONTENT_ID và PUBLIC_DOMAIN không có audio trong CSDL; các nhánh đó được
bảo đảm bằng `tests/test_decision_rules.py`, không phải bằng EXP-08. Nhãn đúng của
điều kiện `known` sinh bằng Rule Engine với danh tính đúng, nên EXP-08 đo **sai sót
nhận diện lan sang mức rủi ro ra sao**, không đo tính đúng pháp lý của luật.

### Ngưỡng phụ thuộc QUY MÔ REFERENCE — đo được, không phải suy đoán

Cùng bộ truy vấn, **cùng một ngưỡng**, chỉ đổi số bản ghi trong reference (FPR ở đây
theo giao thức held-out CŨ — chỉ loại bản trùng hệt — vì các cột 1.000/4.000 đo như vậy;
đo theo `HeldOutProtocol` thì ở 24.375 bài τFP 0.10 chỉ còn 0.0037, 0.15 còn 0.0011):

| Ngưỡng | Chỉ số | @1.000 | @4.000 | @24.375 |
|---|---|---|---|---|
| τFP = 0.10 | FPR (EXP-01) | 0.0044 | 0.0095 | **0.0226** |
| τFP = 0.15 | FPR (EXP-01) | — | 0.0032 | **0.0163** |
| τMERT = 0.97 | FMR (EXP-06) | 3,35% | 4,4% | **6,12%** — vượt trần 5% |
| — | Recall@1 (EXP-02) | 0.9515 | 0.9195 | 0.8787 |

Tầng Cover cũng vậy: τ = 0.70 cho FMR 4,9% khi chỉ tìm trong 100 bài nguồn nhưng
**17,4%** khi tìm trên chỉ mục 24.375 bài.

Nên cả ba ngưỡng đã hiệu chỉnh lại ở 24.375 bài: τFP **0.15 → 0.30**, τMERT
**0.97 → 0.98**; τCover chốt **0.90** trên đúng điều kiện server (giảm so với 0.97
cũ là do đổi descriptor — trừ trung bình từng khung — chứ không phải do quy mô).
Bài học: ngưỡng hiệu chỉnh trên corpus nhỏ không chỉ thiếu chính xác mà **lệch có
hệ thống về phía lạc quan** — corpus nhỏ hiếm khi chứa cặp gây nhầm. Mở rộng
corpus thêm nữa thì **bắt buộc** hiệu chỉnh lại.

### Ngưỡng phụ thuộc ĐỘ DÀI TRUY VẤN — một dương tính giả có thật

Điểm Chromaprint là tỉ lệ bit trùng ở offset căn chỉnh tốt nhất. Đoạn càng ngắn
thì càng ít offset để dò, nên càng dễ gặp một offset "may mắn". Đo bằng nhiễu
trắng (chắc chắn không có trong CSDL) trên reference 4.000, 10 seed mỗi độ dài:

| Độ dài | Điểm nền TB | Max | Ngưỡng áp dụng (τFP 0.30) |
|---|---|---|---|
| 5 s | 0.2632 | **0.3684** | **0.4237** |
| 8 s | 0.1535 | 0.2558 | 0.3000 |
| 10 s | 0.1237 | 0.2034 | 0.3000 |
| 15 s | 0.0990 | 0.1800 | 0.3000 |
| 20 s | 0.0757 | 0.1286 | 0.3000 |
| 30 s | 0.0593 | 0.0814 | 0.3000 |

**Đoạn 5 giây bất kỳ — kể cả nhiễu trắng thuần — đạt tới 0.37, vượt τFP = 0.30.**
Hồi τFP còn 0.15 thì cả đoạn 8 giây (0.26) cũng đã lọt. Một ngưỡng cố định vì thế
không thể vừa an toàn cho đoạn ngắn vừa không quá khắt khe với đoạn dài.
`fingerprint_service.min_score_for_duration()` nâng ngưỡng theo độ dài (chỉ nâng,
không bao giờ hạ dưới τFP — với τFP 0.30 thì chỉ còn tác dụng dưới khoảng 8 giây),
và response mang cờ `threshold_raised_for_short_query` để việc này không diễn ra
âm thầm.

Hai điều cần biết: τFP = 0.30 hiệu chỉnh bằng EXP-01 trên tập truy vấn 10–30 giây,
nên **không chuyển thẳng sang truy vấn ngắn hơn được**; và bảng nhiễu nền trên đo với
reference 4.000 bản ghi. Đo lại trên 24.375 bài (cùng 10 seed) cho đúng các giá trị
lớn nhất đó ở 30 s (0.0814) và 20 s (0.1286) — reference lớn gấp 6 lần không làm điểm
nền của nhiễu cao lên.

### EXP-07 — tầng cover lấp đúng điểm mù

Chấm ở hai giao thức. **Cấp cửa sổ** — 6.059 cửa sổ 15 giây, reference là 200 cửa
sổ của 100 bài nguồn, đúng bộ mà EXP-03 dùng cho MERT nên so thẳng được:

| Nhóm biến đổi | N | Chroma@1 | MERT@1 | OTI đúng |
|---|---|---|---|---|
| **Dịch cao độ** | 1.372 | **68%** | 49% | 91% |
| Đổi tốc độ | 1.400 | 45% | **94%** | 90% |
| Cắt đoạn | 543 | 67% | **99%** | 94% |
| Nén codec | 686 | 69% | **99%** | 92% |
| Nhiễu | 1.029 | 69% | **93%** | 92% |
| Biên độ / EQ | 686 | 69% | **99%** | 92% |
| Chồng âm | 343 | 59% | **78%** | 86% |

Chroma/OTI **thắng MERT ở đúng nhóm dịch cao độ** — chính là điểm mù mà cả
Chromaprint lẫn MERT đều bó tay. Đây là căn cứ bằng số cho việc nối tầng cover vào
cascade, chứ không phải suy đoán.

**Điều kiện server** (`metrics.runtime_protocol`) — 1.900 file truy vấn, tìm trên toàn
chỉ mục cover 24.375 bài, held-out theo `HeldOutProtocol`: đúng ở vị trí 1 **88,8%**.
τCover = **0.90** cho Precision 0.9952, Recall 0.7663, nhận nhầm bài ngoài CSDL
**0,11%**, mẫu nhiễu cao nhất chỉ 0.7347.

| Biến đổi | Đúng @1 | Biến đổi | Đúng @1 |
|---|---|---|---|
| pitch ±1, ±2 | **100%** | tempo 0.90 / 0.95 / 1.05 / 1.10 | **100%** |
| MP3, EQ, gain, nhiễu | 100% | chồng âm | 84% |
| original 30 s | 100% | cắt 10 s / 15 s | **2%** |

**Cắt truy vấn theo nhiều hệ số nhịp độ.** Descriptor là 30 giây đầu co giãn về đúng
64 khung, nên chỉ bất biến nhịp độ khi đoạn truy vấn chứa **trọn** phần nội dung của
reference. Bản chậm 0.90 chứa nó trong 33,3 giây đầu; server trước đây cắt cố định 30
giây nên mất 10% nội dung cuối. Nay `cover_service` cắt theo từng hệ số trong
`COVER_TEMPO_FACTORS` (0.90–1.10, dải biến đổi nhịp độ của §11) và lấy điểm cao nhất.
Kiểm lại trên 100 bài nguồn, chỉ đổi đúng cách cắt
(`python experiments/exp07_cover/query_span_check.py --off-grid`, ~17 phút):

| Truy vấn | Cắt cố định 30 s (cũ) | Nhiều hệ số (hiện tại) |
|---|---|---|
| tempo 0.90 | @1 27% · qua τ 1% | **@1 100% · qua τ 83%** |
| tempo 0.95 | @1 64% · qua τ 2% | **@1 100% · qua τ 77%** |
| tempo 1.05 / 1.10 | @1 100% · qua τ 77% / 81% | không đổi |
| *tempo 0.92 — không có trong bảng hệ số* | @1 43% · qua τ 2% | **@1 100% · qua τ 82%** |
| *tempo 1.07 — không có trong bảng hệ số* | @1 100% · qua τ 76% | không đổi |

Hai dòng cuối có mặt vì các hệ số 0.90/0.95/1.05/1.10 trùng đúng mức tempo của tập
kiểm thử — số ở trên có thể chỉ phản ánh việc "đoán trúng lưới". Nhịp 0.92 (tự đổi
nhịp bằng `librosa.effects.time_stretch`) nằm giữa hai hệ số mà vẫn lên 100% @1.

Thử nhiều độ dài cũng là thêm cơ hội nhận nhầm, nên τCover được hiệu chỉnh lại chứ
không giữ nguyên theo quán tính: ngưỡng vẫn ra 0.90, Recall 0.6837 → **0.7663**, nhận
nhầm 0,37% → 0,11% (lần cũ đo trước `HeldOutProtocol` nên là cận trên — không so
thẳng được), mẫu nhiễu cao nhất 0.7278 → 0.7347.

Đoạn cắt 10/15 giây vẫn trượt vì lệch trục thời gian thuần tuý: so với reference cắt
**cùng khoảng** thì điểm 1.000 (thấp nhất 0.984), so với reference 30 giây chỉ 0.40.
Nhóm này Chromaprint đã bắt 100% nên không phải điểm mù của cascade.

Một điểm dễ đọc nhầm: MERT@1 nhóm pitch = 49% mà trong cascade MERT gần như không
nhận truy vấn pitch nào. Không mâu thuẫn — ở đây chấm **xếp hạng**, còn cascade áp
**ngưỡng τMERT = 0.98**: truy vấn dịch cao độ có điểm MERT khoảng 0.87–0.92, xếp
đúng nhưng bị từ chối. **Điểm mù đó đến từ ngưỡng, không phải từ năng lực model** —
mà ngưỡng buộc phải chặt để giữ FMR ≤ 5%.

#### EXP-07b — Luật chấp nhận theo khoảng cách hạng 1 – hạng 2 (chỉ đo, CHƯA bật)

Tầng Cover cũng gặp đúng chuyện đó: trên chỉ mục 24.375 bài, nó xếp đúng bài ở hạng 1
cho **400/400** truy vấn dịch cao độ và 400/400 đổi nhịp (EXP-04), nhưng 116 và 82
truy vấn bị loại vì điểm tuyệt đối < τCover 0.90. Khoảng cách giữa hạng 1 và hạng 2
tách hai nhóm rất rõ: bài có trong CSDL mà điểm dưới τ có khoảng cách trung vị
**0.315** (5% thấp nhất 0.09), còn bài lạ (held-out) chỉ **0.009** (95% dưới 0.044).
`python experiments/exp07_cover/margin_rule.py --sources 100` (~25 phút, cùng điều kiện
server với EXP-07) đo luật `s1 ≥ τ_abs HOẶC (s1 ≥ τ_low VÀ s1 − s2 ≥ δ)`:

| Luật (1.900 truy vấn) | Precision | Recall | Nhận nhầm bài lạ | Nhiễu | Dịch cao độ | Đổi nhịp | Chồng âm |
|---|---|---|---|---|---|---|---|
| Production: `s1 ≥ 0.90` | 0.9952 | 0.7663 | 2 (0,11%) | 0 | 71% | 79,5% | 57% |
| `s1 ≥ 0.90` hoặc `(s1 ≥ 0.70` và `s1 − s2 ≥ 0.08)` | 0.9905 | **0.8821** | **2 (0,11%)** | 0 | **99,5%** | **100%** | 78% |

Hai lượt nhận nhầm ở cả hai dòng là **cùng** hai đoạn cắt 10/15 giây mà luật hiện
tại vốn đã nhận nhầm — nhánh khoảng cách không thêm lượt nào trên 1.900 truy vấn
held-out. Precision giảm vì nhóm chồng âm: cả 14 lần "nhận sai bài" đều là nhận ra
**bài bị trộn chồng** (có thật trong audio và trong CSDL; pipeline chỉ báo một bài
mỗi truy vấn). Số đọc chính là **kiểm tra chéo** — chọn luật trên một nửa bài nguồn,
chấm trên nửa kia: recall 0.7432 → **0.8779** (nhận nhầm 0 → 0) và 0.7895 → **0.8863**
(0,21% → 0,53%, 5/950 — nửa này bộ chọn lấy δ 0.05 chứ không phải 0.08). δ = 0.08
là lựa chọn SAU KHI đã thấy số, nên bảng trên không phải ước lượng khách quan.
**Production chưa đổi**: như τFP, đổi luật chấp nhận là quyết định của chủ dự án, và
bật thì phải chạy lại EXP-04/05/08.


### EXP-09 — Bản quyền có học được từ âm thanh không?

Thí nghiệm này không nằm trong §15. Nó phát sinh từ một câu hỏi thiết kế đáng giá:
*nếu gắn nhãn bản quyền vào đặc trưng rồi train, hệ thống có đoán được bài chưa
từng thấy không?* §2 đã bác bỏ bằng lập luận ("license không phải một acoustic
class"); đây là lần đầu nó được **đo**.

Ba giao thức, khác nhau ở đúng một chỗ — cách chia dữ liệu. Số liệu trên corpus
4.000 bản ghi / 1.387 nghệ sĩ:

| Bài toán | Model | C. Baseline | A. Ngẫu nhiên | B. Theo nghệ sĩ | Rò rỉ (A−B) | B vượt baseline |
|---|---|---|---|---|---|---|
| license_type | logistic | 0.3827 / 0.079 | 0.306 / 0.265 | 0.2357 / 0.155 | 0.070 | −0.147 / +0.076 |
| license_type | **mlp** | 0.3827 / 0.079 | 0.456 / 0.264 | **0.408** / 0.144 | 0.048 | **+0.025** / +0.065 |
| commercial_use | logistic | 0.8387 / 0.456 | 0.6645 / 0.563 | 0.651 / 0.532 | 0.014 | −0.188 / +0.076 |
| commercial_use | mlp | 0.8387 / 0.456 | 0.8433 / 0.519 | 0.837 / 0.465 | 0.006 | −0.002 / +0.008 |
| **artist (đối chứng)** | mlp | 0.0389 / 0.000 | **0.4455** / 0.390 | **0.000** / 0.000 | **+0.4455** | −0.039 |

*(accuracy / macro-F1; B = nghệ sĩ trong test chưa từng xuất hiện lúc train)*

**Kết luận, và nó thay đổi so với lần chạy trên corpus 1.000.** Ở quy mô nhỏ,
không model nào vượt baseline. Với 4.000 bản ghi, MLP **vượt baseline 2,5 điểm
accuracy và 6,5 điểm macro-F1** ở giao thức B. Sai số chuẩn ở cỡ mẫu này khoảng
0,008 nên mức đó vào khoảng 3σ — là tín hiệu thật, không phải nhiễu.

Nhưng ba điều vẫn giữ nguyên, và chúng mới là điều quyết định:

1. **40,8% accuracy trên 7 lớp không dùng được** cho một quyết định về quyền —
   sai gần 6 trên 10 lần.
2. **Dòng đối chứng còn mạnh hơn trước.** Cùng embedding đó dự đoán *nghệ sĩ* đạt
   0.4455 so với baseline 0.0389 — **gấp 11 lần** — rồi vẫn rơi về **0.000** khi
   nghệ sĩ là người mới. Trong corpus, **95,2% nghệ sĩ chỉ có một loại giấy
   phép**, nên ở giao thức A model chỉ cần nhận ra *ai hát* rồi tra bảng.
3. Tín hiệu yếu đó nhiều khả năng đến từ tương quan **thể loại ↔ giấy phép** (một
   biến trung gian), chứ không phải hệ thống "đọc được bản quyền" từ sóng âm.

Đây là bằng chứng thực nghiệm cho đóng góp **C3**: nhận diện âm thanh và đánh giá
quyền sử dụng là hai bài toán khác nhau, và cái thứ hai không giải được bằng cái
thứ nhất. Giấy phép là thuộc tính pháp lý gắn với hợp đồng — cùng một file WAV có
thể là CC-BY hôm nay và độc quyền thương mại tháng sau mà không đổi một bit.


### Bộ phân loại giấy phép — được tích hợp, kèm cảnh báo

Theo yêu cầu của chủ dự án sau khi đã xem số liệu EXP-09, bộ phân loại vẫn được
train và **tham gia quyết định mức rủi ro** cho bài ngoài cơ sở dữ liệu:

| Giấy phép dự đoán | Phi thương mại | Thương mại |
|---|---|---|
| CC0 | LOW | LOW |
| CC_BY | CONDITIONAL | CONDITIONAL |
| CC_BY_NC_SA | CONDITIONAL | **HIGH** |

Ba chốt an toàn được giữ, và người đọc báo cáo cần biết chúng tồn tại:

1. `identity_confidence` vẫn là điểm MERT thật, **không** thay bằng xác suất của
   classifier — §2 cấm gộp hai loại độ tin cậy.
2. `rights_confidence` bị trừ **0.60** cho nguồn `PREDICTED` (nặng hơn cả metadata
   mô phỏng), nên `decision_confidence` của mọi dự đoán chỉ còn **0.2**.
3. Response mang cờ `rights.predicted = true`, để giao diện không hiển thị quyền
   suy đoán giống quyền tra được từ nguồn thật.

> ⚠️ **Số liệu EXP-09 không ủng hộ việc dùng đầu ra này để kết luận về bản quyền.**
> Nó được ghi lại ở đây để người đọc tự đánh giá, chứ không bị giấu đi.

### Đặc trưng sản xuất — thứ THỰC SỰ đo được từ âm thanh

Đối lập với bản quyền, dấu vết **mastering** thì nằm thật trong tín hiệu và đo
được trực tiếp, không cần nhãn và không cần train:

| Chỉ số | Ý nghĩa |
|---|---|
| `rms_dbfs` | Độ to trung bình, so với mức −14 dBFS mà các nền tảng chuẩn hoá về |
| `crest_factor_db` | Đỉnh trên RMS; dưới 10 dB là dấu hiệu nén dải động mạnh |
| `dynamic_range_db` | Chênh lệch độ to theo thời gian |
| `clipping_ratio` | Tỉ lệ mẫu chạm trần — dấu vết limiter đẩy kịch |

Đo trên corpus: `00_0003.mp3` (RMS −22,0 / crest 20,6) → 0.162 *"giống bản thu
mộc"*; `00_0001.mp3` (RMS −15,4 / crest 11,3) → 0.601 *"không kết luận được"*.

**Nó KHÔNG phải chỉ báo bản quyền và không được phép đổi mức rủi ro** — nhạc
Creative Commons cũng master chuyên nghiệp, còn nhiều bản thu thương mại (cổ điển,
jazz mộc) cố ý giữ dải động rộng. Vì vậy nó chỉ đi vào phần Evidence, và mọi kết
quả đều mang theo cảnh báo đó.


### Số liệu chi tiết (corpus 24.375 bản ghi · 1.900 truy vấn biến đổi từ 100 bài nguồn)

**EXP-01 — Chromaprint theo từng phép biến đổi** (τFP = 0.30, reference 24.375
fingerprint, 100 mẫu mỗi biến đổi)

| Nhóm | Khớp đúng | Điểm TB | Nhóm | Khớp đúng | Điểm TB |
|---|---|---|---|---|---|
| original 30s | 100/100 | 1.000 | pitch ±1 | 0/200 | 0.012 |
| crop 10s / 15s | 200/200 | 1.000 | pitch ±2 | 0/200 | 0.013 |
| MP3 128k / 64k | 200/200 | 0.999 | tempo 0.90 | 0/100 | 0.025 |
| EQ lowpass 4k | 100/100 | 1.000 | tempo 0.95 | 0/100 | 0.040 |
| gain −12 dB | 100/100 | 1.000 | tempo 1.05 | 0/100 | 0.044 |
| noise SNR 20/10 dB | 200/200 | 0.971 | tempo 1.10 | 0/100 | 0.031 |
| noise SNR 5 dB | 99/100 | 0.822 | voice overlay | 57/100 | 0.467 |

Đây là câu trả lời bằng số cho "*Fingerprint thất bại ở đâu?*": mọi biến đổi giữ
nguyên trục thời gian và cao độ đều đạt **99–100%**; toàn bộ nhóm pitch và tempo
sụp về **0%** với điểm ~0.01–0.04. Đó là giới hạn bản chất của fingerprinting, và
chính là lý do tồn tại của tầng MERT và tầng Cover. Độ trễ trung bình **4,5 giây**
mỗi truy vấn — phần lớn là so với toàn bảng 24.375 fingerprint.

**EXP-02 — MERT Retrieval** (leave-one-out trên 48.750 vector, 1,19 tỉ cặp khác bản
ghi, tính theo khối)

| Recall@1 | Recall@5 | Recall@10 | MRR | mAP |
|---|---|---|---|---|
| 0.8787 | 0.9374 | 0.9479 | 0.9059 | 0.9036 |

Truy vấn ở đây là một cửa sổ **chưa biến đổi** của chính bài có trong chỉ mục, nên
Recall@5 = 0.9374 là **cận trên**; tiêu chí "Robust retrieval" của §13 được chấm
bằng EXP-04 trên truy vấn đã biến đổi. Similarity cùng bản ghi 0.9645 so với khác
bản ghi 0.7771. Ở τMERT = 0.98 vẫn còn 1.196 cặp khác bản ghi vượt ngưỡng (1,0 phần
triệu), trong đó 283 cặp ≥ 0.999 — mức chỉ gặp khi hai bản ghi gần như cùng một
audio.

**EXP-04 — Năm hệ thống trên cùng 1.900 truy vấn** (thí nghiệm chính; τFP 0.30 ·
τMERT 0.98 · τCover 0.90 — đúng cấu hình server, kể cả bộ lọc hash ở Tầng 1 và cách
cắt truy vấn theo nhịp độ ở tầng Cover)

| Hệ thống | Precision | Recall | F1 | Nhận sai | Latency TB |
|---|---|---|---|---|---|
| A. Chromaprint | 0.9981 | 0.5558 | 0.7140 | 2 | 147 ms |
| B. MERT | 0.9957 | 0.2437 | 0.3915 | 2 | 946 ms |
| C. Cover (chroma/OTI) | 0.9952 | 0.7663 | 0.8659 | 7 | 688 ms |
| D. Cascade FP → MERT | 0.9982 | 0.5726 | 0.7278 | 2 | 587 ms |
| **E. Cascade production** (FP → MERT → Cover) | 0.9970 | **0.8805** | **0.9352** | 5 | 906 ms |

- Tầng Cover nâng Recall của cascade từ **0.5726 lên 0.8805** mà Precision gần như
  giữ nguyên. 1.058 truy vấn dừng ở Chromaprint, 32 ở MERT, 810 đi tới Cover.
- **Cả 5 "nhận sai" của cascade production đều là truy vấn `audio_overlay` nhận ra
  bài bị trộn chồng** — bài đó cũng nằm trong CSDL, nhưng EXP-04 lấy nhãn đúng là
  bài nguồn nên vẫn chấm SAI. Không có lần nào nhận ra một bài không liên quan.
- MERT đơn lẻ Recall chỉ 0.2437 vì τMERT = 0.98 rất chặt — cần chặt như vậy để
  FMR ≤ 5% ở 24.375 bài (EXP-06).
- Độ trễ: cascade P50 57 ms (dừng ở Tầng 1), P95 2,3 s (đi tới Cover). Tầng 1 lọc
  ứng viên theo hash — P50 41 ms, 30/1.900 truy vấn (1,6%) quay về quét toàn bộ vì
  điểm sát ngưỡng. Lượt trước quét toàn bộ mọi truy vấn: Tầng 1 mất ~4,8 s, cascade
  trung bình 5.411 ms.

So với lượt trước (Tầng 1 quét toàn bộ, Cover cắt cố định 30 s): Recall cascade 0.8047
→ 0.8805, F1 0.8905 → 0.9352; từng biến đổi đổi nhịp của cascade 8/10/79/82% →
**83/79/79/82%** (tempo 0.90/0.95/1.05/1.10).

**Recall@5 với truy vấn đã biến đổi, tìm trên toàn chỉ mục** — tiêu chí "Robust
retrieval" của §13, chấm ở cấp hệ thống (`system_recall@5`), tính trên top-5 bất kể
ngưỡng:

| Nhóm | N | MERT | Cover | **Hệ thống (MERT ∪ Cover)** |
|---|---|---|---|---|
| Dịch cao độ | 400 | **0.185** | 1.000 | 1.000 |
| Đổi tốc độ | 400 | 0.945 | 1.000 | 1.000 |
| Còn lại (cắt, codec, nhiễu, EQ, chồng âm) | 1.100 | 0.957 | 0.818 | 0.989 |
| **Toàn bộ** | 1.900 | 0.7921 | 0.8947 | **0.9937** |

MERT đơn lẻ thiếu 0.008 so với mục tiêu 0.80, toàn bộ do dịch cao độ (MERT mã hoá cao
độ tuyệt đối; bỏ nhóm đó ra thì 0.954). Cover trượt ở chiều ngược lại — đoạn cắt 10/15
giây (0.06) — và ở đó MERT bắt đủ. Hai tầng bù đúng điểm mù của nhau.

**EXP-05 — Độ bền theo nhóm biến đổi** (tổng hợp lại từ EXP-01 và EXP-04; mỗi ô là
tỉ lệ nhận ĐÚNG sau khi áp ngưỡng)

| Nhóm | N | Chromaprint | MERT | Cover | Cascade production | Suy giảm điểm FP |
|---|---|---|---|---|---|---|
| Cắt đoạn | 300 | **100%** | 76% | 33% | **100%** | — |
| Nén codec | 200 | **100%** | 50% | 100% | **100%** | −0.001 |
| Biên độ / EQ | 200 | **100%** | 30% | 100% | **100%** | +0.000 |
| Nhiễu | 300 | 99,7% | 8% | 99% | **100%** | −0.079 |
| Đổi tốc độ | 400 | 0% | 8% | 80% | **81%** | −0.970 |
| Dịch cao độ | 400 | 0% | 0,25% | 71% | **71%** | −0.992 |
| Chồng âm | 100 | 57% | 17% | 57% | **65%** | −0.538 |

- **Dịch cao độ và đổi tốc độ** từng là điểm mù của cả Chromaprint lẫn MERT; tầng Cover
  đưa cascade lên **71%** và **81%** (đổi tốc độ từ 45% trước khi cắt truy vấn theo
  nhịp độ).
- **Còn yếu nhất: chồng âm (65%) và dịch cao độ (71%)** — với dịch cao độ, Cover luôn
  xếp đúng bài ở top-5 (1.000) nhưng điểm nhiều truy vấn rơi dưới τCover = 0.90.
- Cột MERT thấp không phải vì MERT xếp hạng kém: điểm MERT chỉ giảm 0.03–0.06 khi
  biến đổi, nhưng thế là đủ rơi dưới τMERT = 0.98.


## 🧪 Kiểm thử

```bash
python -m pytest -q -rs
```

Test cần PostgreSQL hoặc `fpcalc` sẽ **skip kèm lý do** thay vì báo đỏ.
Hai bộ test đáng chú ý:

- `tests/test_chromaprint_codec.py` — chứng minh bộ giải nén thuần Python cho ra
  đúng từng phần tử so với `fpcalc -raw`.
- `tests/test_fingerprint_match.py` — chứng minh bản so khớp vector hoá cho điểm
  giống hệt `acoustid._match_fingerprints` khi giới hạn cùng dải offset.
- `tests/test_decision_rules.py` — 65 case phủ mọi nhánh Rule Engine (§9).
- `tests/test_cover_service.py` — 19 case, trong đó phần cốt lõi chứng minh dịch
  cao độ k bán cung cho OTI đúng bằng `(-k) mod 12`, dùng hợp âm tổng hợp có tần
  số biết trước nên kết quả là xác định, không phụ thuộc phase vocoder.

### Một phát hiện quan trọng về Chromaprint

`acoustid._match_fingerprints` chỉ dò lệch **±120 item ≈ ±15 giây**, nên một đoạn
cắt từ giữa bài **không thể khớp**. EXP-01 đo được trên cùng một cặp fingerprint:
điểm **0.0498** với cửa sổ ±120 item, và **0.8688** khi dò toàn bộ offset (điểm
khớp thật nằm ở offset −242 item = đúng giây 30 của bài). Vì vậy matcher của hệ
thống mặc định dò toàn bộ offset (`FP_MAX_ALIGN_OFFSET=0`), cài đặt bằng phép
cộng đường chéo với `np.bincount` nên vừa chính xác vừa nhanh (~1.4 ms/cặp).

---

## 🛠️ Script tiện ích

| Script | Công dụng |
|---|---|
| `scripts/rebuild_faiss_index.py` | Dựng lại FAISS index + `faiss_id_map.json` từ CSV |
| `scripts/check_data_integrity.py` | Kiểm tra khoá ngoại, số chiều vector, khớp index/ID map |
| `scripts/add_test_track.py` | Nạp một file nhạc thật vào CSDL để thử end-to-end |
| `scripts/check_fingerprint.py` | Chạy thử riêng tầng 1 trên một file |
| `scripts/recognize_track.py` | Nhận diện bằng SQL thuần, tiện soi CSDL |
| `scripts/evaluate_benchmark.py` | Đo độ trễ pipeline trên index thật |
| `scripts/generate_test_queries.py` | Sinh kịch bản test transformation vào bảng `test_queries` |
| `scripts/build_embeddings.py` | Sinh embedding MERT từ audio thật và nạp vào reference |
| `scripts/augment_audio.py` | Dựng tập truy vấn biến đổi (crop/nén/nhiễu/pitch/tempo) |
| `scripts/enrich_rights_metadata.py` | Sinh rights phủ đủ 5 nhóm Rule Engine (metadata mô phỏng) |

---

## ⚠️ Giới hạn hiện tại của dữ liệu

Cơ sở dữ liệu có **158.117 bản ghi**, nhưng chỉ **24.375** trong đó có audio thật trên
đĩa. Phân biệt hai loại này là điều kiện để đọc đúng mọi con số thí nghiệm:

| Nguồn | Bản ghi | Audio trên đĩa | Fingerprint / embedding / cover | Giấy phép | Nguồn quyền |
|---|---|---|---|---|---|
| FMA (medium) | **24.375** | ✅ | ✅ 24.375 / 48.750 vector / 24.375 | CC thật do FMA công bố | xác minh (`metadata_verified`) |
| dataset_G nhạc Việt | 100.000 | ❌ (URL Spotify giả) | ❌ | COMMERCIAL, CC, PD, Creator Music, cover | `SIMULATED` |
| MTG-Jamendo | 29.881 | ❌ (URL tải) | ❌ | CC_BY / CC_BY_NC suy từ cờ tải về | `SIMULATED` |
| Spotify Web API | 3.361 | ❌ | ❌ | COMMERCIAL | chưa xác minh hãng phát hành |
| YouTube Audio Library | 400 | ❌ | ❌ | AUDIO_LIBRARY | `SIMULATED` (có `verified_at`) |
| Creator Music | 100 | ❌ | ❌ | CREATOR_MUSIC | `SIMULATED` (có `verified_at`) |

Mọi thí nghiệm nhận diện (EXP-01 → EXP-08) chỉ chạm tới 24.375 bài FMA. Audio thật
**không nằm trong repo** — đặt ở `AUDIO_ROOT`, dựng lại bằng `scripts/fetch_fma.py`
rồi `scripts/ingest_corpus.py`.

**Dữ liệu mô phỏng bị cổng quyền chặn, không bị xoá.** Nhóm AUDIO_LIBRARY,
CREATOR_MUSIC và CONTENT_ID không có nguồn mở nào cấp được — chúng là chính sách nội
bộ của nền tảng, không phải giấy phép công bố kèm bản thu. Metadata được giữ để cả 5
nhánh Rule Engine có dữ liệu tra cứu, nhưng:

- **Không bản ghi nào không có audio mà lại có fingerprint** — đó chính là lỗi đã làm
  hỏng ground truth ở phiên bản đầu.
- Nguồn quyền mang tiền tố `SIMULATED`; chưa có `verified_at` thì `rights_confidence`
  còn 0.40 < 0.50 và `rights_gate` hạ kết luận về `UNKNOWN` (dataset_G, Jamendo). Metadata
  mô phỏng đã có `verified_at` (YouTube Audio Library, Creator Music) giữ 0.70 để làm
  ca kiểm thử cho Rule Engine.
- `metadata_verified = TRUE` chỉ có ở 24.375 bài FMA.

**Chất lượng phần dữ liệu thật:**

- Fingerprint: **24.299 giá trị duy nhất / 24.375**. Phần trùng là bản thu trùng thật
  trong FMA (cùng audio dưới nhiều `recording_id`) — Chromaprint cho chúng cùng
  fingerprint là ĐÚNG. Thí nghiệm gộp chúng thành lớp tương đương; EXP-08 ghi
  `resolved_to_other_id_in_same_fingerprint_class = 0`.
- Giấy phép: `backend/services/license_mapping.py` suy cờ quyền từ chính điều khoản CC,
  đối chứng độc lập với bốn cờ FMA tự tách sẵn (`allow_commercial_use`,
  `allow_derivatives`, `require_attribution`, `require_share_alike`).
- Tập truy vấn: **2.090 truy vấn** — 1.900 từ 100 bài nguồn × 19 phép biến đổi, cộng
  190 từ 10 bài CC0 cho EXP-08.

Phân bố giấy phép của 24.375 bài FMA:

| Giấy phép | Số lượng | Giấy phép | Số lượng |
|---|---|---|---|
| CC_BY_NC_ND | 10.124 | CC_BY_SA | 867 |
| CC_BY_NC_SA | 9.697 | CC_BY_ND | 209 |
| CC_BY | 1.689 | CC0 | 117 |
| CC_BY_NC | 1.672 | | |

**Còn thiếu, và không tự lấp bằng cách đoán:**

- **Chỉ nhánh CREATIVE_COMMONS và fallback của Rule Engine đi được bằng audio thật.**
  15.033 dòng PUBLIC_DOMAIN đều thuộc dataset_G mô phỏng; FMA không có bản thu mang
  Public Domain Mark. 117 bài CC0 đi vào nhánh 4 và cho `LOW` — đúng nghiệp vụ, nhưng
  nhánh 5 và nhánh `RECORDING_PERMISSION_REQUIRED` (tác phẩm PD, bản thu còn bản quyền)
  chỉ được unit test phủ. Cần nguồn khác (Musopen, archive.org). **Không xếp CC0 vào
  nhóm 5 chỉ để bảng thống kê đủ 5 nhóm.**
- **Chưa có bản thu cover nào**, nên EXP-07 vẫn là thí nghiệm bất biến cao độ/nhịp độ
  trên chính bản thu gốc, không phải cover identification đúng nghĩa. Cần subset
  SecondHandSongs (§8).
- 10.032 dòng `COVER_MECHANICAL_LICENSE` (dataset_G) không thuộc nhóm nào → nhánh
  `uncategorized`, `UNKNOWN` kèm lý do nêu đích danh loại giấy phép.

Chạy `python scripts/check_data_integrity.py` để xem báo cáo cập nhật
(hiện tại: **17 PASS / 2 WARN / 0 FAIL**).

---


## 📂 Cấu trúc dự án

<!-- TREE:START -->
<!-- TREE:END -->

---
*Phần cấu trúc thư mục ở trên được GitHub Action cập nhật tự động.*
