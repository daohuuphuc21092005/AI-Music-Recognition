# 🎵 MUSIC RECOGNITION & RIGHTS INSPECTION PLATFORM

> Hệ thống AI nhận diện âm nhạc đa tầng (Hybrid Cascade) kết hợp Audio Fingerprinting và Deep Retrieval 768 chiều, phục vụ tra cứu bản quyền tự động.

---

## 📌 Tổng quan kiến trúc

```
INPUT (audio/video)
  → Tầng 1: Chromaprint fingerprint      → score ≥ τFP   → EXACT_MATCH
  → Tầng 2: MERT-v1-95M + FAISS (cosine) → sim   ≥ τMERT → NEAR_MATCH
                                          → sim   < τMERT → UNKNOWN (cần người duyệt)
  → Tra cứu rights → Đánh giá rủi ro → Evidence
```

- **Tầng 1 (Exact Match):** Chromaprint. Fingerprint được giải nén bằng
  `backend/services/chromaprint_codec.py` (thuần Python, không cần thư viện native)
  và so khớp bằng thuật toán bit-error của Chromaprint đã vector hoá bằng numpy.
- **Tầng 2 (Deep Retrieval):** MERT-v1-95M + FAISS `IndexFlatIP` trên vector đã
  chuẩn hoá L2. Điểm được **gộp theo `recording_id`** nên Top-K là K bản ghi khác
  nhau, không phải K đoạn của cùng một bài.
- **Tầng 3 (Cover — CHƯA nối vào cascade):** `backend/services/cover_service.py`,
  chroma từ CQT + Optimal Transposition Index. Bất biến với dịch cao độ vì dịch
  k bán cung chỉ là xoay vòng vector chroma đi k bậc. τCover = 0.97 hiệu chỉnh
  bằng EXP-07, và EXP-07 cho thấy tầng này **thắng MERT ở đúng nhóm dịch cao độ**
  (66% so với 54%) — tức nó lấp được điểm mù còn lại của cascade.
- **Backend:** FastAPI + PostgreSQL.

> Ba ngưỡng dưới đây hiệu chỉnh trên **corpus 4.000 bản ghi thật, 1.900 truy vấn
> biến đổi**. Chi tiết ở mục Thí nghiệm.
>
> **τFP = 0.15** — EXP-01: Precision 0.9963, FPR 0.0032. Chọn theo quy tắc tường
> minh "F1 cao nhất trong nhóm giữ FPR ≤ 0.005", vì trong hệ thống bản quyền nhận
> nhầm tốn kém hơn bỏ sót (bỏ sót còn được tầng MERT xử lý tiếp).
>
> **τMERT = 0.97** — EXP-06: False Match Rate **4,4%**, vừa đạt ngưỡng §16 (≤ 5%).
> Giá trị cũ 0.95 từng được báo cáo là "FMR 1,87%" trên corpus nhỏ; đo lại trên
> 4.000 bản ghi thì chính ngưỡng đó cho **14,7%**.
>
> **τCover = 0.97** — EXP-07: Precision 0.9864, FMR 1,4%.
>
> ⚠️ **Cả ba ngưỡng phụ thuộc QUY MÔ reference.** Corpus tăng gấp 4 thì ở cùng
> τFP = 0.10, FPR tăng từ 0.0044 lên 0.0095. **Mở rộng dữ liệu ⇒ bắt buộc hiệu
> chỉnh lại**, và phải sửa đồng thời `.env` lẫn `configs/rules_v1.yaml`.
>
> ⚠️ **τFP còn phụ thuộc ĐỘ DÀI truy vấn.** Đoạn 8 giây bất kỳ — kể cả nhiễu
> trắng thuần — đạt điểm Chromaprint ~0.16, vượt τFP = 0.15. `min_score_for_duration()`
> tự nâng ngưỡng cho truy vấn ngắn (chỉ nâng, không bao giờ hạ dưới τFP).
>
> **Cổng định danh của Rule Engine tách ngưỡng theo TỪNG TẦNG** (`EXACT_MATCH: 0.15`,
> `NEAR_MATCH: 0.97`) chứ không dùng một con số chung — điểm Chromaprint (tỉ lệ bit
> trùng) và cosine similarity của MERT **không cùng thang đo**.

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
# DATABASE_URL=postgresql://postgres:postgrespassword@localhost:5433/music_rights_ai
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

Chạy bằng Docker: `docker compose up --build` (API ở cổng 8000, DB ở cổng 5433).

---

## 🔌 API

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/health` | Trạng thái từng thành phần: database, faiss, chromaprint, ffmpeg, mert, rule_engine |
| `POST` | `/api/v1/analyze` | Upload + `platform` / `commercial_use` / `monetization` → trả `job_id` (202) |
| `GET` | `/api/v1/jobs/{job_id}` | `QUEUED` / `PROCESSING` / `DONE` / `FAILED` |
| `GET` | `/api/v1/results/{job_id}` | Kết quả đầy đủ: identity, match, rights, assessment, evidence |
| `GET` | `/api/v1/tracks/{recording_id}` | Tra cứu bản ghi + quyền |
| `POST` | `/api/v1/feedback` | `Correct` / `Incorrect` / `Unsure` |
| `POST` | `/api/v1/search` | Bản đồng bộ của `/analyze` (tiện thử nhanh) |

Lỗi luôn trả về mã chuẩn hoá, **không bao giờ trả traceback**:
`FILE_TOO_LARGE`, `UNSUPPORTED_FORMAT`, `NO_AUDIO`, `NO_MUSIC`, `MODEL_FAILURE`,
`DATABASE_FAILURE`, `TIMEOUT`, `UNKNOWN_TRACK`, `LOW_CONFIDENCE`.

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
| **1. Upload** | Kéo–thả hoặc chọn file · Nền tảng · Mục đích (phi thương mại / thương mại) · Bật kiếm tiền. Ba tuỳ chọn này là **đầu vào thật của Rule Engine**, không phải trang trí. |
| **2. Processing** | Bảy bước THẬT của pipeline, tô sáng theo trường `stage` mà backend ghi vào bảng `jobs` — **không có thanh chờ giả, không có "AI is thinking"**. Xong bước nào hiện độ trễ đo được của bước đó. |
| **3. Result** | Nhận diện · Quyền & giấy phép · Mức rủi ro 🟢/🟡/🔴/⚪ · Khuyến nghị · **ba thanh độ tin cậy tách biệt** (identity / rights / decision) · nút phản hồi Correct / Incorrect / Unsure. |
| **4. Evidence** | Điểm fingerprint và ngưỡng · bảng Top-K của MERT · luật Rule Engine đã kích hoạt · nguồn metadata quyền và ngày xác minh · PD của tác phẩm và PD của bản thu **để riêng** (§2) · độ trễ từng bước · JSON gốc. |

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

Hoặc chạy cả chuỗi đúng thứ tự phụ thuộc (EXP-05 đọc kết quả EXP-01+04, EXP-07 đọc
EXP-03) — chạy sai thứ tự thì hai thí nghiệm đó âm thầm đối chiếu với số liệu của
lần chạy trước:

```bash
python scripts/run_all_experiments.py                    # lượt 1: đo ngưỡng
python scripts/apply_calibrated_thresholds.py --write    # ghi τ vào .env
python scripts/run_all_experiments.py --only exp04,exp05,exp08   # lượt 2
```

EXP-03/04/08 chạy hàng giờ nên có **checkpoint**: bị ngắt giữa chừng (máy hết RAM
chẳng hạn) thì chạy lại sẽ tiếp tục từ đúng chỗ cũ. Chữ ký cấu hình nằm ở dòng đầu
file checkpoint — đổi ngưỡng hay đổi tập truy vấn thì checkpoint cũ tự bị vứt, chứ
không trộn kết quả của hai cấu hình vào một bảng.

Kết quả lưu ở `experiments/results/*.json` kèm `git_commit`, phiên bản dữ liệu,
tham số và ngưỡng (§14).

### Trạng thái 9 thí nghiệm (§15 yêu cầu 8; EXP-09 phát sinh từ một câu hỏi thiết kế)

Toàn bộ số liệu dưới đây đo trên **corpus FMA thật**: 4.000 bản ghi có audio,
8.000 vector MERT, 1.900 truy vấn biến đổi từ 100 bản ghi nguồn.

| # | Thí nghiệm | Kết quả chính |
|---|---|---|
| EXP-01 | Fingerprint Baseline | → chốt **τFP = 0.15** (P=0.9963, FPR=0.0032) |
| EXP-02 | MERT Retrieval | Recall@1 **0.9195** · Recall@5 **0.9656** · MRR 0.9407 |
| EXP-03 | Pooling Strategy | `mean+std` chỉ +1,15pp Recall@1 nhưng index ×2 → **giữ `mean`** |
| EXP-04 | FP vs MERT vs Hybrid | **Cascade F1 0.7587** > FP 0.7193 > MERT 0.5751 |
| EXP-05 | Robustness | Điểm mù còn lại: **dịch cao độ** (0/401) |
| EXP-06 | Unknown Detection | → chốt **τMERT = 0.97** (FMR 4,4%) |
| EXP-07 | Cover Identification | **Chroma thắng MERT ở nhóm pitch: 66% vs 54%** |
| EXP-08 | End-to-End | Macro-F1 **0.785** · FMR **0,68%** · Unknown detection **99,32%** |
| EXP-09 | Học bản quyền từ âm thanh? | Tín hiệu rất yếu, không dùng được |

### EXP-08 — kết quả không đạt ngưỡng nghiệm thu, và vì sao

| Chỉ số §16 | Mục tiêu | Đo được | |
|---|---|---|---|
| Macro-F1 | ≥ 0.80 | **0.785** | ❌ |
| False Match Rate | ≤ 5% | **0,68%** | ✅ |

Tách theo điều kiện thì nguyên nhân lộ ra ngay:

| Điều kiện | Macro-F1 | Accuracy | n |
|---|---|---|---|
| `held_out` (bài ngoài CSDL) | **0.9966** | 0.9932 | 3.800 |
| `known` (bài có trong CSDL) | 0.7631 | **0.6155** | 3.800 |

Hệ thống **từ chối bài lạ gần như hoàn hảo**, nhưng 1.470 truy vấn `known` không
nhận diện được — đúng nhóm pitch/tempo mà EXP-05 đã chỉ ra. Macro-F1 tụt dưới
0.80 vì khâu NHẬN DIỆN, không phải vì Rule Engine sai: khi nhận diện thất bại thì
chỉ 0,61% trường hợp vẫn ra đúng mức rủi ro, tức hệ thống chuyển sang `UNKNOWN`
chứ không đoán bừa — đúng nguyên tắc §2.

`resolved_to_other_id_in_same_fingerprint_class = 0`: ba cặp bản thu trùng nhau
trong corpus không gây nhầm lẫn nào.

Theo §16 ("*sau tuần 5 có thể điều chỉnh target nếu baseline cho thấy dataset khó
— phải ghi lý do*"), lý do được ghi ở đây: tập truy vấn cố ý gồm 8/19 nhóm biến
đổi nằm ngoài khả năng của fingerprinting. Chấm trên tập đó rồi đòi Macro-F1 ≥
0.80 là đòi tầng nhận diện làm việc mà kiến trúc đã dành cho module cover.

### Ngưỡng phụ thuộc QUY MÔ REFERENCE — đo được, không phải suy đoán

Khi corpus tăng từ 1.000 lên 4.000 bản ghi, ở **cùng một ngưỡng**:

| Ngưỡng | Chỉ số | @1.000 bản ghi | @4.000 bản ghi |
|---|---|---|---|
| τFP = 0.10 | FPR | 0.0044 | **0.0095** (gấp 2,2×) |
| τMERT = 0.97 | FMR | 3,35% | **4,4%** (sát trần §16) |
| — | Recall@1 (EXP-02) | 0.9515 | 0.9195 |

Nên τFP phải nâng **0.10 → 0.15**. Bài học: ngưỡng hiệu chỉnh trên corpus nhỏ
không chỉ thiếu chính xác mà **lệch có hệ thống về phía lạc quan** — corpus nhỏ
hiếm khi chứa cặp gây nhầm. Mở rộng corpus thêm nữa thì **bắt buộc** hiệu chỉnh lại.

### Ngưỡng phụ thuộc ĐỘ DÀI TRUY VẤN — một dương tính giả có thật

Điểm Chromaprint là tỉ lệ bit trùng ở offset căn chỉnh tốt nhất. Đoạn càng ngắn
thì càng ít offset để dò, nên càng dễ gặp một offset "may mắn". Đo bằng nhiễu
trắng (chắc chắn không có trong CSDL) trên reference 4.000, 10 seed mỗi độ dài:

| Độ dài | Điểm nền TB | Max | Ngưỡng áp dụng |
|---|---|---|---|
| 5 s | 0.2632 | 0.3684 | 0.4237 |
| 8 s | 0.1535 | **0.2558** | 0.2942 |
| 10 s | 0.1237 | 0.2034 | 0.2339 |
| 15 s | 0.0990 | 0.1800 | 0.2070 |
| 20 s | 0.0757 | 0.1286 | 0.1500 |
| 30 s | 0.0593 | 0.0814 | 0.1500 |

**Đoạn 8 giây bất kỳ — kể cả nhiễu trắng thuần — đạt ~0.16, vượt τFP = 0.15.**
Một ngưỡng cố định vì thế không thể vừa an toàn cho đoạn ngắn vừa không quá khắt
khe với đoạn dài. `fingerprint_service.min_score_for_duration()` nâng ngưỡng theo
độ dài (chỉ nâng, không bao giờ hạ dưới τFP), và response mang cờ
`threshold_raised_for_short_query` để việc này không diễn ra âm thầm.

τFP = 0.15 hiệu chỉnh bằng EXP-01 trên tập truy vấn 30 giây, nên **không chuyển
thẳng sang truy vấn ngắn được** — đó là lý do hàm này tồn tại.

### EXP-07 — tầng cover lấp đúng điểm mù

| Nhóm biến đổi | N | Chroma@1 | MERT@1 | OTI đúng |
|---|---|---|---|---|
| **Dịch cao độ** | 1.384 | **66%** | 54% | 92% |
| Đổi tốc độ | 1.400 | 37% | **93%** | 88% |
| Cắt đoạn | 546 | 67% | **100%** | 94% |
| Nén codec | 692 | 68% | **100%** | 93% |
| Nhiễu | 1.038 | 66% | **95%** | 93% |
| Biên độ / EQ | 692 | 68% | **99%** | 93% |

Chroma/OTI **thắng MERT ở đúng nhóm dịch cao độ** — chính là điểm mù mà EXP-05
báo là cả hai tầng đều bó tay. Đây là căn cứ bằng số cho việc nối tầng cover vào
cascade, chứ không phải suy đoán.

Một điểm dễ đọc nhầm: EXP-04 báo MERT cứu được 0–1% nhóm pitch, còn ở đây MERT@1
= 54%. Không mâu thuẫn — EXP-07 chấm **xếp hạng** ở mức cửa sổ, EXP-04 áp **ngưỡng
τMERT = 0.97**. MERT *vẫn xếp đúng* hơn nửa số truy vấn pitch, chỉ là điểm rơi
dưới 0.97 nên bị từ chối. **Điểm mù ở EXP-05 đến từ ngưỡng, không phải từ năng
lực model** — mà ngưỡng đó buộc phải chặt để giữ FMR ≤ 5%.

τCover đề xuất **0.97** (P=0.9864, R=0.4405, FMR 1,4% — đạt §16).


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


### Số liệu chi tiết (corpus 4.000 bản ghi · 1.900 truy vấn biến đổi)

**EXP-01 — Chromaprint theo từng phép biến đổi** (100 mẫu mỗi nhóm)

| Nhóm | Khớp đúng | Điểm TB | Nhóm | Khớp đúng | Điểm TB |
|---|---|---|---|---|---|
| original 30s | 100/100 | 1.000 | pitch ±1 | 0/100 | 0.009 |
| crop 10s / 15s | 100/100 | 1.000 | pitch ±2 | 0/100 | 0.009 |
| MP3 128k / 64k | 100/100 | 0.999 | tempo 0.90 | 2/100 | 0.028 |
| EQ lowpass 4k | 100/100 | 1.000 | tempo 0.95 | 4/100 | 0.040 |
| gain −12 dB | 100/100 | 0.9995 | tempo 1.05 | 3/100 | 0.041 |
| noise SNR 20/10 dB | 100/100 | 0.987 | tempo 1.10 | 4/100 | 0.032 |
| noise SNR 5 dB | 100/100 | 0.784 | voice overlay | 74/100 | 0.456 |

Đây là câu trả lời bằng số cho §20 "*Fingerprint thất bại ở đâu?*": mọi biến đổi
giữ nguyên trục thời gian và cao độ đều đạt **100%**; toàn bộ nhóm pitch và tempo
sụp về **~0%** với điểm ~0.01–0.04. Đó là giới hạn bản chất của fingerprinting,
và chính là lý do tồn tại của tầng MERT.

**EXP-02 — MERT Retrieval** (leave-one-out trên 8.000 vector, 32,0 triệu cặp âm)

| Recall@1 | Recall@5 | Recall@10 | MRR | mAP |
|---|---|---|---|---|
| 0.9195 | **0.9656** | 0.9715 | 0.9407 | 0.9383 |

Recall@5 = 0.9656 vượt xa ngưỡng §16 (≥ 0.80). Similarity cùng bản ghi 0.9667 so
với khác bản ghi 0.7765.

**EXP-04 — Ba hệ thống trên cùng 1.900 truy vấn** (thí nghiệm chính, §15)

| Hệ thống | Precision | Recall | F1 | Nhận sai | Latency TB |
|---|---|---|---|---|---|
| A. Chromaprint | 0.9963 | 0.5629 | 0.7193 | 4 | 829 ms |
| B. MERT | 0.9974 | 0.4040 | 0.5751 | 2 | 4.754 ms |
| **C. Cascade** | 0.9957 | **0.6128** | **0.7587** | 5 | 3.009 ms |

Cascade cho Recall và F1 cao nhất, đồng thời **nhanh hơn MERT đơn thuần 1,6 lần**
— vì tầng 1 bắt được phần lớn truy vấn và dừng sớm, không phải chạy MERT.

**EXP-05 — Độ bền theo nhóm biến đổi**

| Nhóm | N | Chromaprint | MERT | Cascade | Suy giảm điểm FP |
|---|---|---|---|---|---|
| Cắt đoạn | 300 | **100%** | 96% | **100%** | — |
| Nén codec | 200 | **100%** | 90% | **100%** | −0.001 |
| Biên độ / EQ | 200 | **100%** | 54% | **100%** | −0.000 |
| Nhiễu | 300 | 99% | 21% | **99%** | −0.099 |
| Chồng âm | 100 | 69% | 34% | **70%** | −0.544 |
| Đổi tốc độ | 400 | 1% | 23% | **24%** | −0.965 |
| Dịch cao độ | 401 | 0% | 0% | **0%** | −0.991 |

Hai tầng bù trừ nhau đúng như giả thuyết C1: điểm Chromaprint sụt **−0.991** khi
dịch cao độ trong khi MERT chỉ sụt **−0.068**; ngược lại với nhiễu thì Chromaprint
giữ 99% còn MERT rơi xuống 21%.

**Dịch cao độ là điểm mù của cả hai tầng (0/401).** Trên corpus 1.000 nó còn được
2%; nâng τMERT lên 0.97 để giữ FMR ≤ 5% đã xoá nốt phần đó. Đây là đánh đổi có
thật giữa "dám từ chối bài lạ" và "bắt được biến đổi mạnh" — và EXP-07 cho thấy
tầng cover lấp được đúng chỗ này (Chroma 66% so với MERT 54% ở nhóm pitch).


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

Reference database gồm **4.922 bản ghi**, chia làm hai loại rất khác nhau — phân
biệt được hai loại này là điều kiện để đọc đúng mọi con số thí nghiệm:

| Loại | Số lượng | Có audio | Fingerprint | Embedding | Giấy phép |
|---|---|---|---|---|---|
| Corpus thật (FMA) | **4.000** | ✅ | ✅ 4.000 | ✅ 8.000 vector | **thật**, do FMA công bố |
| Metadata mô phỏng | 922 | ❌ | ❌ | ❌ | `SIMULATED` |

Audio thật **không nằm trong repo** (3,9 GB) — đặt ở `AUDIO_ROOT`, dựng lại bằng
`scripts/fetch_fma.py --shards 8 --sample 4000` rồi `scripts/ingest_corpus.py`.

Quy mô này đạt mục tiêu §8 (2.400–4.100 bản ghi tham chiếu).

**Vì sao vẫn giữ 922 bản ghi mô phỏng.** Ba nhóm đầu của Rule Engine —
`AUDIO_LIBRARY` (400), `CONTENT_ID` (422), `CREATOR_MUSIC` (100) — không có nguồn
mở nào cấp được: chúng là chính sách nội bộ của nền tảng, không phải giấy phép
công bố kèm bản thu. Giữ dạng metadata để cả 5 nhánh Rule Engine vẫn có dữ liệu,
nhưng **tuyệt đối không cho chúng fingerprint hay embedding** — bản ghi không có
audio mà có fingerprint chính là lỗi đã làm hỏng ground truth ở phiên bản trước.
Chúng giữ tiền tố `SIMULATED` ở `source` nên `compute_rights_confidence` trừ điểm
đúng.

**Đã khắc phục so với phiên bản trước:**

- Fingerprint: **3.997/4.000 giá trị duy nhất**. Trước đây 2.650 dòng chỉ có 2.016
  giá trị vì fingerprint của `test.mp3` bị gán cho 38 `recording_id` khác nhau.
  Ba cặp trùng còn lại là **bản thu trùng thật trong FMA** (cùng tiêu đề, nghệ sĩ,
  thời lượng, nằm ở các shard khác nhau) — Chromaprint cho chúng cùng fingerprint
  là ĐÚNG vì chúng đúng là cùng audio. EXP-08 xác nhận chúng không gây nhầm lẫn
  nào (`resolved_to_other_id_in_same_fingerprint_class = 0`).
- Tập truy vấn: **1.900 truy vấn từ 100 bản ghi nguồn × 19 phép biến đổi**, thay
  cho 38 truy vấn từ đúng 2 file audio.
- Giấy phép: 4.000 bản ghi mang giấy phép Creative Commons **thật**.
  `backend/services/license_mapping.py` suy các cờ quyền từ chính điều khoản CC,
  và được đối chứng độc lập với bốn cờ mà FMA tự tách sẵn
  (`allow_commercial_use`, `allow_derivatives`, `require_attribution`,
  `require_share_alike`) — **khớp 100% trên 4.207 track**.

Phân bố giấy phép của 4.000 bản ghi thật:

| Giấy phép | Số lượng | Giấy phép | Số lượng |
|---|---|---|---|
| CC_BY_NC_SA | 1.531 | CC_BY_SA | 200 |
| CC_BY_NC_ND | 1.511 | CC0 | 40 |
| CC_BY | 377 | CC_BY_ND | 28 |
| CC_BY_NC | 313 | | |

**Còn thiếu, và không tự lấp bằng cách đoán:**

- **Nhánh 5 `PUBLIC_DOMAIN` của Rule Engine chưa có dữ liệu thật.** FMA-small không
  có bản thu nào mang Public Domain Mark. 40 track CC0 đi vào nhánh 4
  (`CREATIVE_COMMONS` khai báo `CC0` trong `match`) và cho `LOW / FREE_TO_USE` — đúng
  về nghiệp vụ, nhưng nghĩa là nhánh 5 hiện chỉ được unit test phủ, không có dữ liệu
  thật nào chạm tới. §8 đặt mục tiêu 200–300 bản ghi PD; cần nguồn khác (Musopen,
  archive.org). **Không xếp CC0 vào nhóm 5 chỉ để bảng thống kê đủ 5 nhóm.**
- Hệ quả trực tiếp: nhánh `RECORDING_PERMISSION_REQUIRED` (tác phẩm PD nhưng bản thu
  còn bản quyền) — minh hoạ tiêu biểu nhất cho §2 — chưa kiểm được bằng dữ liệu thật.
- **Chưa có bản thu cover nào**, nên EXP-07 vẫn là thí nghiệm bất biến cao độ/nhịp độ
  trên chính bản thu gốc, không phải cover identification đúng nghĩa. Cần subset
  SecondHandSongs (§8).
- Tầng 2 (MERT) chỉ tìm được trong 4.000 bản ghi có embedding, không phải cả 4.922.

Chạy `python scripts/check_data_integrity.py` để xem báo cáo cập nhật
(hiện tại: **11 PASS / 4 WARN / 0 FAIL**).

---


## 📂 Cấu trúc dự án

<!-- TREE:START -->
<!-- TREE:END -->

---
*Phần cấu trúc thư mục ở trên được GitHub Action cập nhật tự động.*
