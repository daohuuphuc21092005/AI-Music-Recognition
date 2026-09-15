> **Đề cương gốc của dự án — lưu nguyên văn, không chỉnh sửa.**
>
> Tách khỏi `CLAUDE.md` ngày 2026-09-14 (bản tại commit `afe090f`). Giữ nguyên số mục để
> mọi trích dẫn `§N` trong code, test, `configs/rules_v1.yaml`, README và kết quả thí
> nghiệm vẫn trỏ đúng chỗ. Hướng dẫn làm việc hiện hành nằm ở `CLAUDE.md`; chỗ nào đề
> cương lệch thực tế (cây thư mục §6, `data_dictionary.xlsx` ở §7...), `CLAUDE.md` và
> code là nguồn đúng.

---

# CLAUDE.md

Tài liệu hướng dẫn cho Claude (hoặc bất kỳ AI coding agent nào) khi làm việc trong repository **`music-rights-ai`** — Hệ thống AI nhận diện & phân loại nhạc bản quyền (*AI-based Music Identification and Copyright Usage Risk Assessment System*).

Đọc file này trước khi viết code, thiết kế schema, hoặc đề xuất kiến trúc mới trong repo. Các quyết định dưới đây đã được **chốt** — không tự ý thay đổi trừ khi người dùng yêu cầu rõ ràng.

---

## 1. Tổng quan dự án

Hệ thống nhận đầu vào là **audio hoặc video**, sau đó:

1. Phát hiện đoạn có chứa nhạc.
2. Xác định đoạn nhạc có trùng bản ghi đã có trong database tham chiếu không.
3. Nếu không trùng hoàn toàn → tìm các bản ghi tương đồng cao (retrieval).
4. Khi cần → nhận diện khả năng đây là cover/remix/phiên bản khác.
5. Xác định `recording_id` và/hoặc `composition_id`.
6. Tra cứu thông tin giấy phép & trạng thái quyền sử dụng.
7. Áp dụng rule nghiệp vụ (Rule Engine — **không phải ML**).
8. Trả về: loại khớp, độ tin cậy, trạng thái quyền, mức rủi ro, điều kiện sử dụng, evidence, khuyến nghị.
9. Phân loại vào **5 nhóm bản quyền**: Audio Library, Creator Music, Commercial/Content ID, Creative Commons, Public Domain (+ nhóm ngoại lệ Uncategorized/UGC).

### Ba đóng góp khoa học cần thể hiện trong code & báo cáo
- **C1:** Kiến trúc cascade Audio Fingerprinting + Deep Music Embedding để nhận dạng nhạc.
- **C2:** Protocol đánh giá robustness (crop, compression, noise, pitch, tempo, voice overlay) + unknown-track detection.
- **C3:** Framework tách biệt rõ **nhận diện âm thanh** (identity) khỏi **đánh giá quyền sử dụng** (rights/risk) — Audio Evidence + Rights Metadata + Rule Engine.

---

## 2. Nguyên tắc bất di bất dịch (không được vi phạm khi code)

- **Không phân loại ngẫu nhiên / song song.** Rule Engine phải tuân theo thứ tự ưu tiên cố định: `Audio Library → Creator Music → Content ID → Creative Commons → Public Domain → Uncategorized`.
- **`composition_id` ≠ `recording_id`.** Composition (tác phẩm) và Recording (bản thu cụ thể) là hai thực thể tách biệt, một composition có thể có nhiều recording. Không bao giờ gộp hai khái niệm này trong code, schema hay API.
- **Không suy luận `composition public domain ⇒ recording public domain`.** Hai trường `composition_public_domain` và `recording_public_domain` phải được lưu và xét độc lập.
- **Không được ép `UNKNOWN` thành `LOW` hoặc `HIGH`.** Khi thiếu thông tin (không nhận diện được, similarity thấp, license mâu thuẫn, metadata cũ...) → luôn trả `UNKNOWN`, kèm `HUMAN_REVIEW_REQUIRED`.
- **AI không hoạt động như hộp đen.** Mọi output phân loại đều phải kèm: dữ liệu chứng minh (evidence object) + cơ chế đảo ngược/giải thích. Không trả kết quả risk mà không có `decision_reason` và các score gốc.
- **Không dùng một số điểm duy nhất làm "copyright probability".** Phải tách riêng `identity_confidence`, `rights_confidence`, `decision_confidence`.
- **Không dùng CLAP (hay bất kỳ zero-shot classifier nào) làm bộ phân loại bản quyền trực tiếp** (vd. prompt `"This is copyrighted music"`). License không phải một acoustic class — CLAP chỉ dùng như experiment phụ để so sánh embedding.
- **Không train/test bằng cách split ngẫu nhiên theo segment.** Bắt buộc split theo `recording_id`; với cover detection, split theo `composition_id`. Các version của cùng composition không được nằm rải rác cả train lẫn test.
- **Không trả traceback cho frontend.** Lỗi phải được map sang mã lỗi chuẩn hoá (xem mục 8).
- **Quy tắc ưu tiên tiến độ quan trọng nhất:** Không phát triển Cover Detection, RAG, LLM, cloud phức tạp, hay fine-tuning model **trước khi** pipeline lõi `Fingerprint → MERT → Rights → Result` đã chạy ổn định end-to-end. Đây là MVP cốt lõi, ưu tiên tuyệt đối.

---

## 3. Phạm vi MVP

**MVP bắt buộc:**
```
Upload audio/video
→ tìm đoạn nhạc
→ nhận diện trong reference database
→ Top-K retrieval nếu cần
→ tra cứu rights metadata
→ đánh giá LOW / CONDITIONAL / HIGH / UNKNOWN
→ hiển thị evidence
```

**MVP chưa cần làm** (không tự ý mở rộng phạm vi): nhận diện hàng triệu bài, realtime livestream, sao chép Content ID của YouTube, kết luận pháp lý, tự động crawl toàn Internet, huấn luyện foundation model từ đầu, xử lý mọi loại mashup/remix phức tạp.

---

## 4. Kiến trúc & Pipeline tổng quát

```
INPUT AUDIO/VIDEO
  → Audio Pre-processing (tách luồng âm thanh, chuẩn hoá, lọc nhiễu)
  → Music Segment Detection (khoanh vùng đoạn có nhạc)
  → Audio Fingerprinting (Chromaprint)
      ├─ Exact/Near-exact match → Track/Composition Resolver
      └─ No exact match → Music Embedding Retrieval (MERT)
                            → Cover/Version Identification (nếu cần)
                            → Track/Composition Resolver
  → Rights Database (tra cứu sở hữu & giấy phép)
  → Rule Engine (cây quyết định phân loại 5 nhóm bản quyền)
  → Copyright Usage Risk Assessment (LOW / CONDITIONAL / HIGH / UNKNOWN)
  → Evidence + Recommendation
```

**Quy tắc cascade AI:**
```
QUERY → Fingerprint (Chromaprint)
  ├─ score ≥ τFP → EXACT/NEAR_EXACT
  └─ score < τFP → MERT embedding → Top-K
        ├─ similarity ≥ τMERT → near-match
        └─ similarity < τMERT → Cover module → candidate? → composition : UNKNOWN
```

Ngưỡng `τFP`, `τMERT` phải được xác định bằng validation set (cân bằng Precision/Recall/FPR), **không tự chọn threshold trước khi có dữ liệu**.

---

## 5. Tech stack đã chốt

Không thay thế các thành phần dưới đây trừ khi có yêu cầu rõ ràng:

| Thành phần | Công nghệ |
|---|---|
| Backend framework | FastAPI |
| Database quan hệ | PostgreSQL |
| Vector search | FAISS (giai đoạn nghiên cứu) → Qdrant (giai đoạn sản phẩm) |
| Cache/queue (optional) | Redis |
| Exact/near-exact identification | Chromaprint (`fpcalc`) |
| Deep music retrieval | MERT-v1-95M (không cần bản 330M cho MVP) |
| Cover/version identification (nâng cao) | CQT/Chroma baseline, sau đó CoverHunter/CSI (ưu tiên pretrained inference, không train từ đầu) |
| Containerization | Docker |
| Audio extraction | FFmpeg |

---

## 6. Cấu trúc repository

```
music-rights-ai/
│
├── backend/
│   ├── api/
│   ├── schemas/
│   ├── services/
│   │   ├── audio_service.py
│   │   ├── fingerprint_service.py
│   │   ├── embedding_service.py
│   │   ├── retrieval_service.py
│   │   ├── cover_service.py
│   │   ├── rights_service.py
│   │   └── decision_service.py
│   ├── database/
│   └── main.py
│
├── models/
│   ├── fingerprint/
│   ├── mert/
│   └── cover/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── metadata/
│   └── test_queries/
│
├── experiments/
│   ├── exp01_fingerprint/
│   ├── exp02_mert/
│   ├── exp03_pooling/
│   ├── exp04_hybrid/
│   ├── exp05_robustness/
│   ├── exp06_unknown/
│   ├── exp07_cover/
│   └── exp08_end_to_end/
│
├── frontend/
├── tests/
├── docker/
├── configs/
├── scripts/
└── README.md
```

Khi thêm code mới, **đặt đúng vị trí theo cấu trúc trên** — không tạo thư mục song song cùng chức năng.

---

## 7. Data model (PostgreSQL)

Các bảng bắt buộc — giữ nguyên tên field khi implement schema/migration:

- **`compositions`**: `composition_id (UUID)`, `title`, `composer`, `year`, `public_domain_status (enum: verified/possible/no/unknown)`, `source`, `verified_at`
- **`recordings`**: `recording_id (UUID)`, `composition_id`, `title`, `artist`, `album`, `release_year`, `duration`, `source_dataset`, `source_track_id`, `audio_path`, `metadata_verified`
- **`rights`**: `rights_id`, `recording_id`, `composition_id`, `license_type`, `copyright_status`, `attribution_required`, `commercial_use_allowed`, `modification_allowed`, `monetization_allowed`, `revenue_share_required`, `territory`, `platform`, `valid_from`, `valid_until`, `source`, `source_url`, `verified_at`
- **`fingerprints`**: `fingerprint_id`, `recording_id`, `algorithm`, `fingerprint`, `duration`, `created_at`
- **`embeddings`**: `embedding_id`, `recording_id`, `segment_start`, `segment_end`, `model`, `model_version`, `dimension`, `vector`, `created_at`
- **`test_queries`**: `query_id`, `recording_id`, `composition_id`, `transformation`, `snr`, `pitch_shift`, `tempo_factor`, `codec`, `bitrate`, `segment_start`, `duration`, `expected_match_type`
- **`analysis_results`**: `job_id`, `query_id`, `recording_candidate`, `composition_candidate`, `fingerprint_score`, `embedding_score`, `cover_score`, `match_type`, `risk_level`, `confidence`, `decision_reason`, `latency_ms`, `model_version`, `timestamp`

Không thêm/bớt field cốt lõi trong các bảng này mà không cập nhật `data_dictionary.xlsx` tương ứng.

---

## 8. Nguồn dữ liệu (reference database) & quy mô mục tiêu

| Dataset | Vai trò | Quy mô đề xuất |
|---|---|---|
| MTG-Jamendo | Dataset mở chính để build/test embedding & retrieval | 1.000–1.500 tracks |
| FMA | Bổ sung đa dạng, negative examples, retrieval eval | 500–1.000 tracks |
| YouTube Audio Library | Case LOW/CONDITIONAL rõ ràng | 300–500 tracks |
| Creator Music | **Không dùng để train** — chỉ metadata mô phỏng cho Rule Engine, không crawl audio | 50–100 metadata records |
| Public Domain | Case PUBLIC_DOMAIN — phải phân biệt composition status vs recording status | 200–300 recordings |
| Cover dataset (SecondHandSongs subset) | Giai đoạn nâng cao, không cần train từ đầu | 100–200 compositions, 2–5 versions/composition |

Tổng tham chiếu mục tiêu: **2.400–4.100** bản ghi (không bắt buộc đạt đúng, đây là target).

**Tiền xử lý:** M1 Audio Ingestion (FFmpeg cho video) → M2 Normalization (không ép cùng sampling rate cho mọi model; MERT dùng 24kHz) → M3 Segmentation (10s/15s/30s có overlap).

**Data augmentation cho test set:** original, crop, MP3/AAC compression, noise, voice overlay, gain, EQ, pitch shift (±1,±2 semitone), tempo (0.9–1.10), re-encoding — giữ một **Robustness Test Set cố định**, không bắt buộc dùng hết để train.

---

## 9. Rule Engine — logic phân loại 5 nhóm bản quyền

Rule Engine là **logic tường minh (if/else), không phải model ML**. Thứ tự đánh giá cố định:

1. **Audio Library**: `is_in_audio_library = TRUE` → `attribution_required = FALSE` ⇒ `LOW`, không điều kiện; `= TRUE` ⇒ `CONDITIONAL`, `ATTRIBUTION_REQUIRED`.
2. **Creator Music**: `license_purchased = TRUE & VALID` ⇒ `LOW`, giữ 100% doanh thu; `revenue_share_agreed = TRUE` ⇒ `CONDITIONAL`, `REVENUE_SHARE_APPLIED`.
3. **Commercial/Content ID**: `policy_action = MONETIZE_CLAIM` ⇒ `CONDITIONAL`, `REVENUE_REDIRECTED`; `policy_action = BLOCK_OR_TAKEDOWN` ⇒ `HIGH`, `VIDEO_BLOCKED_OR_STRIKE`.
4. **Creative Commons**: `CC_BY` + metadata hợp lệ ⇒ `CONDITIONAL`, `ATTRIBUTION_REQUIRED`; `CC_BY_NC` + `commercial_use = TRUE` ⇒ `HIGH`, `NON_COMMERCIAL_VIOLATION`.
5. **Public Domain**: composition & recording đều PD ⇒ `LOW`, tự do sử dụng; composition PD nhưng recording copyrighted ⇒ `CONDITIONAL_OR_HIGH`, `RECORDING_PERMISSION_REQUIRED`.
6. **Ngoại lệ**: `recording_status = UNKNOWN` hoặc `confidence < THRESHOLD_MIN` ⇒ `UNKNOWN`, `HUMAN_REVIEW_REQUIRED`; không match bất kỳ database nào ⇒ `USER_GENERATED_CONTENT`, `LOW`, `MONETIZABLE_UNTIL_RETROACTIVE_CLAIM`.

Rules nên được viết dưới dạng file cấu hình (`rules_v1.yaml`/`json`), có unit test riêng (`unit_test_rules`), tối thiểu 50–100 test case bao phủ mọi nhánh trên.

---

## 10. Các service backend bắt buộc

- **`AudioService`**: validate file, extract audio, normalize, segment, cleanup.
- **`FingerprintService`**: generate fingerprint, search fingerprint DB, trả score.
- **`EmbeddingService`**: load MERT, batch inference, pooling, normalize.
- **`RetrievalService`**: Top-K vector search, thresholding, ranking.
- **`CoverService`** (optional): CQT/chroma, cover similarity.
- **`RightsService`**: query recording/composition/rights, validate metadata.
- **`DecisionService`**: apply rules, sinh risk, condition, evidence.

---

## 11. API specification

```
POST /api/v1/analyze
  input: file, platform, commercial_use, monetization

GET /api/v1/jobs/{job_id}
  response status: QUEUED | PROCESSING | DONE | FAILED

GET /api/v1/results/{job_id}
  response ví dụ:
  {
    "status": "DONE",
    "identity": {"track": "...", "artist": "...", "recording_id": "...", "composition_id": "..."},
    "match": {"type": "EXACT", "confidence": 0.97},
    "rights": {"license": "CC_BY", "attribution_required": true},
    "assessment": {"risk": "CONDITIONAL", "reason": "Attribution required"}
  }

GET /api/v1/tracks/{recording_id}

POST /api/v1/feedback
  input: Correct | Incorrect | Unsure
```

---

## 12. Error handling & logging

**Mã lỗi bắt buộc xử lý:** `FILE_TOO_LARGE`, `UNSUPPORTED_FORMAT`, `NO_AUDIO`, `NO_MUSIC`, `MODEL_FAILURE`, `DATABASE_FAILURE`, `TIMEOUT`, `UNKNOWN_TRACK`, `LOW_CONFIDENCE`. Không bao giờ trả traceback thô cho frontend.

**Mỗi request phải log:** `job_id`, `timestamp`, `file duration`, `fingerprint latency`, `embedding latency`, `vector-search latency`, `database latency`, `total latency`, `result`, `confidence`, `error`, `model version`. Không lưu file người dùng lâu dài nếu không cần thiết.

---

## 13. Frontend — 4 màn hình MVP tối thiểu

1. **Upload**: drag&drop/choose file, chọn Platform (YouTube/Other), Purpose (Non-commercial/Commercial), Monetization (Yes/No), nút ANALYZE.
2. **Processing**: hiển thị progress từng bước thực (audio extracted, fingerprint generated, searching DB, deep similarity, checking rights) — **không hiển thị "AI is thinking"**.
3. **Result**: 4 phần — Identification (song/artist/recording/composition/match type/confidence), Rights (license/attribution/commercial use/monetization/source/last verified), Assessment (🟢LOW/🟡CONDITIONAL/🔴HIGH/⚪UNKNOWN), Recommendation (câu giải thích ngôn ngữ tự nhiên dựa trên evidence).
4. **Evidence**: fingerprint score, embedding Top-K, matched track, metadata source, license source, verification date, decision rules triggered.

Trang `/admin` (tuỳ chọn cho giảng viên/admin): xem database, thêm recording, chỉnh rights, rebuild fingerprints/embeddings, xem log, export experiment result.

---

## 14. Model versioning & experiment tracking

Không ghi đơn giản `model = MERT`. Phải ghi đầy đủ: `model_name`, `checkpoint`, `model_version`, `preprocessing_version`, `embedding_dimension`, `pooling_method`, `threshold`, `date_created`.

Mỗi experiment lưu: `experiment_id`, `git_commit`, `dataset_version`, `split_version`, `model`, `parameters`, `threshold`, `metrics`, `date` (CSV/JSON đủ dùng, MVP không bắt buộc MLflow).

---

## 15. Các thí nghiệm bắt buộc (Experiments)

| # | Tên | Nội dung chính | Metrics |
|---|---|---|---|
| EXP-01 | Fingerprint Baseline | Chromaprint trên original/crop/compression/noise | Precision, Recall, F1, FPR, Latency |
| EXP-02 | MERT Retrieval | Query vs reference DB | Recall@1/5/10, MRR, mAP |
| EXP-03 | Pooling Strategy | Mean vs Mean+Std | Recall@1/5, Latency |
| EXP-04 | Fingerprint vs MERT vs Hybrid | Chromaprint / MERT / cascade — **thí nghiệm chính của đồ án** | Precision, Recall, Recall@5, FPR, Latency |
| EXP-05 | Robustness | Codec/noise/crop/pitch/tempo/voice overlay | Biểu đồ transformation vs performance |
| EXP-06 | Unknown Track Detection | Query ngoài reference DB — **bắt buộc** | False Match Rate, Unknown Recall/Precision |
| EXP-07 | Cover Identification | Chroma/CQT vs MERT vs cover model | Recall@1/5, MRR, mAP |
| EXP-08 | End-to-End System | Full pipeline | Macro-F1, class-wise P/R, Confusion Matrix, Unknown Detection Rate, Latency |

---

## 16. Ngưỡng nghiệm thu kỹ thuật (mục tiêu nội bộ, không phải chuẩn ngành)

- Clean exact-match: `Precision ≥ 0.95`, `Recall ≥ 0.90`
- Robust retrieval: `Recall@5 ≥ 0.80`
- Unknown test: `False Match Rate ≤ 5%`
- End-to-end verified subset: `Macro-F1 ≥ 0.80`

Sau tuần 5 có thể điều chỉnh target nếu baseline cho thấy dataset khó — **phải ghi lý do** trong experiment log.

---

## 17. Roadmap 12 tuần & Milestones

| Tuần | Trọng tâm | Milestone |
|---|---|---|
| 1 | Chốt yêu cầu, kiến trúc, data schema, API draft | — |
| 2 | Data pipeline: metadata, `recording_id`/`composition_id`, rights schema | M1 |
| 3 | Pre-processing: ingestion, normalization, segmentation | M1 (cuối tuần) |
| 4 | Fingerprinting baseline (Chromaprint) + EXP-01 | M2 |
| 5 | MERT embedding + FAISS + Top-K + EXP-02 | — |
| 6 | Tối ưu retrieval (pooling, threshold, Top-K) — đóng băng MERT config v1 & τMERT | M3 |
| 7 | Hybrid cascade (Chromaprint→MERT) + EXP-04 — **milestone quan trọng nhất** | M4 |
| 8 | Robustness + Unknown detection (EXP-05, EXP-06) | — |
| 9 | Rights DB + Rule Engine hoàn thiện, 50–100 test case | M5 |
| 10 | Backend + Frontend integration, demo upload→result→evidence | M6 |
| 11 | Cover module (nếu tiến độ tốt) hoặc EXP-08 + ổn định hệ thống (nếu chậm) — **không hy sinh MVP để làm Cover AI** | M7 |
| 12 | Docker, deploy, freeze code, report, video demo, slides | M8 |

---

## 18. Phân công gợi ý (nhóm 4 người)

- **Data & Rights**: datasets, metadata, rights database, license taxonomy, rule engine.
- **AI/MIR**: Chromaprint, MERT, FAISS, retrieval, cover module.
- **Backend/MLOps**: FastAPI, PostgreSQL, job management, Docker, logging, deployment.
- **Frontend/Evaluation**: UI/UX, API integration, test queries, experiments, visualization, system testing.

Tất cả thành viên phải hiểu toàn bộ pipeline, không chỉ phần của mình.

---

## 19. Hình/bảng bắt buộc trong báo cáo kỹ thuật

**Hình:** Overall system architecture, Audio preprocessing pipeline, Hybrid identification architecture, Database ERD, Rights decision workflow, Backend architecture, User workflow, Deployment architecture.

**Bảng:** Dataset summary, Rights taxonomy, Database schema, Audio transformations, Fingerprinting performance, MERT retrieval performance, Hybrid comparison, Robustness test, Unknown-track detection, End-to-end performance, Latency analysis, Error analysis.

---

## 20. Tiêu chí hoàn thành dự án

Dự án hoàn thành khi thực hiện được end-to-end: upload MP3/MP4 → tách âm thanh → fingerprint → (nếu không đủ) MERT retrieval → Top-K → xác định recording/composition → tra cứu rights → Rule Engine phân loại & đánh giá → hiển thị LOW/CONDITIONAL/HIGH/UNKNOWN → hiển thị Evidence.

Báo cáo phải trả lời bằng **số liệu thực nghiệm**: Hybrid tốt hơn baseline trong điều kiện nào? Fingerprint thất bại ở đâu? MERT cải thiện ở đâu? Pitch/tempo ảnh hưởng thế nào? Hệ thống có từ chối được bài ngoài DB không? False positive xảy ra ở đâu? Quyết định LOW/HIGH dựa trên evidence nào?

---

## 21. Tóm tắt kiến trúc đã đóng băng

```
Chromaprint       → Exact/Near-exact Identification
MERT              → Robust Music Retrieval
CQT/Chroma        → Cover Identification (optional)
FAISS/Qdrant      → Vector Search
PostgreSQL        → Metadata + Rights
Rule Engine       → Rights Decision
FastAPI           → Backend
Web Application   → User Interface
Docker            → Deployment
```

> **Ghi nhớ khi hỗ trợ code cho dự án này:** luôn ưu tiên hoàn thiện pipeline lõi `Fingerprint → MERT → Rights → Result` trước khi thêm bất kỳ tính năng nâng cao nào (cover detection, LLM, RAG, cloud phức tạp, fine-tuning).
