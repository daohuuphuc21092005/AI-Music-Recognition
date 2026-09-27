# CLAUDE.md

Tài liệu định hướng cho Claude Code khi làm việc trong repo này. Đây là dự án **Hệ thống AI Phân loại Nhạc Bản quyền** (AI-based Music Identification and Copyright Usage Risk Assessment System) — một đồ án/prototype MIR (Music Information Retrieval) kết hợp Audio Fingerprinting, Deep Music Embedding, Rights Metadata và Rule-based Reasoning.

## 1. Bài toán

Input: file audio hoặc video có chứa âm thanh.
Output cho mỗi đoạn nhạc: `match_type`, `confidence`, `recording_id`/`composition_id`, `license`/`rights status`, `risk_level` (LOW/CONDITIONAL/HIGH/UNKNOWN), `condition` sử dụng, `evidence`, `recommendation`.

Hệ thống phân loại nhạc vào **5 nhóm bản quyền** theo thứ tự ưu tiên **cố định, tuần tự** (không chạy song song, không random):

```
Audio Library → Creator Music → Content ID (Commercial) → Creative Commons → Public Domain
→ (nếu không khớp điều kiện nào) UNCATEGORIZED / USER-GENERATED CONTENT
```

## 2. Nguyên tắc bất biến (không được vi phạm khi code)

1. **Composition ID ≠ Recording ID.** Composition = tác phẩm/giai điệu (quyền tác giả). Recording = một bản thu cụ thể (quyền bản ghi/phát sóng). Một composition có nhiều recording. **Không bao giờ suy luận `composition public domain ⇒ recording public domain`.**
2. **Thứ tự phân loại là cây quyết định tuần tự**, không phải multi-label song song: kiểm tra Audio Library trước, rồi mới đến Creator Music, Content ID, Creative Commons, Public Domain, cuối cùng mới fallback UNCATEGORIZED.
3. **Không kết luận từ kết quả nghi ngờ.** Mọi quyết định phân loại phải đi kèm chỉ số tin cậy, tách riêng: `identity_confidence`, `rights_confidence`, `decision_confidence`.
4. **Không hộp đen.** Mọi output phân loại phải kèm: dữ liệu chứng minh (evidence) + cơ chế đảo ngược/giải thích được (decision rules triggered).
5. **UNKNOWN không được ép thành LOW hoặc HIGH.** Dùng UNKNOWN khi: không nhận diện được, similarity thấp, license không xác định, metadata mâu thuẫn, composition xác định được nhưng recording thì không, hoặc dữ liệu quyền đã cũ.
6. **Rule Engine là logic if/else thuần, KHÔNG phải ML.** Không dùng model để quyết định risk/category — chỉ dùng model cho identity/similarity, quyết định rights/risk luôn qua rule engine tường minh.
7. **Không dùng CLAP (hay bất kỳ model nào) làm "copyright classifier".** License/risk không phải một acoustic class; không suy ra trạng thái pháp lý từ zero-shot prompt kiểu "this is copyrighted music".
8. **Split dữ liệu theo `recording_id`, không theo segment.** Với cover detection, split theo `composition_id`. Các đoạn cùng track/cùng composition tuyệt đối không được nằm rải rác cả train lẫn test.
9. **Ưu tiên MVP core trước.** Không phát triển Cover Detection, RAG, LLM, cloud phức tạp, hay fine-tuning model trước khi pipeline lõi `Fingerprint → MERT → Rights → Rule Engine → Result` chạy ổn định end-to-end. Đây là ràng buộc quan trọng nhất khi ước lượng phạm vi công việc.

## 3. Kiến trúc pipeline (cascade, không phải 5 model song song)

```
INPUT AUDIO/VIDEO
  → Audio Pre-processing (tách âm thanh, chuẩn hóa, lọc nhiễu)
  → Music Segment Detection
  → Audio Fingerprinting (Chromaprint)
       ├─ score ≥ τFP → EXACT/NEAR-EXACT MATCH
       └─ score < τFP → Music Embedding Retrieval (MERT)
                            ├─ similarity ≥ τMERT → near-match candidate
                            └─ similarity < τMERT → Cover/Version Identification (optional, advanced)
  → Track/Composition Resolver (recording_id, composition_id)
  → Rights Database lookup
  → Rule Engine (cây quyết định 5 nhóm, xem mục 6)
  → Copyright Usage Risk Assessment (LOW/CONDITIONAL/HIGH/UNKNOWN)
  → Evidence + Recommendation
```

### Module tiền xử lý
- **M1 Audio Ingestion**: input MP3/WAV/M4A/MP4/MOV; video → FFmpeg → audio; giữ file gốc.
- **M2 Normalization**: KHÔNG ép mọi model dùng chung sampling rate (Chromaprint có preprocessing riêng; MERT dùng 24kHz).
- **M3 Segmentation**: window 10s/15s/30s có overlap.

### AI-1 — Exact Audio Identification (bắt buộc, baseline)
- Chromaprint/fpcalc. Không phải fingerprinting tổng quát cho mọi biến đổi — chỉ cho near-identical audio.
- Threshold `τFP` KHÔNG được chọn tùy tiện — phải tìm qua validation set để cân bằng Precision/Recall/FPR.

### AI-2 — Deep Music Retrieval (kích hoạt khi fingerprint_score < τFP)
- Model: **MERT-v1-95M** (không cần bản 330M trong MVP).
- Pooling: so sánh tối thiểu Mean pooling (P1) vs Mean+Std pooling (P2). Attention pooling (P3) chỉ nếu còn thời gian.
- Vector search: **FAISS** giai đoạn nghiên cứu → **Qdrant** giai đoạn sản phẩm (cosine similarity).

### AI-3 — Cover/Version Identification (nâng cao, optional)
- Chỉ triển khai sau khi AI-1 và AI-2 chạy ổn định.
- Baseline bắt buộc: CQT/Chroma + sequence similarity/dynamic alignment.
- Nâng cao (tùy tiến độ): CoverHunter hoặc tương đương — ưu tiên pretrained inference, KHÔNG train từ đầu.

### Rights Resolver → Rule Engine
Sau khi có `recording_id`, truy vấn `recordings` + `compositions` + `rights`, dựng **Evidence Object** (xem ví dụ JSON trong schema `analysis_results`), rồi áp Rule Engine.

## 4. Tech stack đã chốt

| Thành phần | Lựa chọn |
|---|---|
| Backend framework | FastAPI |
| Database | PostgreSQL |
| Vector search | FAISS (nghiên cứu) → Qdrant (sản phẩm) |
| Cache/queue | Redis (optional, khi async processing) |
| Fingerprinting | Chromaprint / fpcalc |
| Embedding | MERT-v1-95M |
| Cover detection | CQT/Chroma baseline (+ CoverHunter optional) |
| Deployment | Docker |
| Experiment tracking | CSV/JSON — KHÔNG bắt buộc MLflow ở MVP |

## 5. Cấu trúc repository (mục tiêu)

```
music-rights-ai/
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
├── models/
│   ├── fingerprint/
│   ├── mert/
│   └── cover/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── metadata/
│   └── test_queries/
├── experiments/
│   ├── exp01_fingerprint/ ... exp08_end_to_end/
├── frontend/
├── tests/
├── docker/
├── configs/
├── scripts/
└── README.md
```

### Service bắt buộc (backend)
`AudioService` (validate/extract/normalize/segment/cleanup) · `FingerprintService` · `EmbeddingService` (load MERT, batch inference, pooling, normalize) · `RetrievalService` (top-K, threshold, ranking) · `CoverService` (optional) · `RightsService` · `DecisionService` (apply rules, risk, conditions, evidence).

## 6. Database schema (bắt buộc)

- **compositions**: `composition_id (UUID)`, `title`, `composer`, `year`, `public_domain_status (verified/possible/no/unknown)`, `source`, `verified_at`
- **recordings**: `recording_id (UUID)`, `composition_id (FK)`, `title`, `artist`, `album`, `release_year`, `duration`, `source_dataset`, `source_track_id`, `audio_path`, `metadata_verified (bool)`
- **rights**: `rights_id`, `recording_id (FK)`, `composition_id (FK)`, `license_type`, `copyright_status`, `attribution_required`, `commercial_use_allowed`, `modification_allowed`, `monetization_allowed`, `revenue_share_required`, `territory`, `platform`, `valid_from`, `valid_until`, `source`, `source_url`, `verified_at`
- **fingerprints**: `fingerprint_id`, `recording_id`, `algorithm`, `fingerprint`, `duration`, `created_at`
- **embeddings**: `embedding_id`, `recording_id`, `segment_start`, `segment_end`, `model`, `model_version`, `dimension`, `vector`, `created_at`
- **test_queries**: `query_id`, `recording_id`, `composition_id`, `transformations`, `snr`, `pitch_shift`, `tempo_factor`, `codec`, `bitrate`, `segment_start`, `duration`, `expected_match_type`
- **analysis_results**: `job_id`, `query_id`, `recording_candidate`, `composition_candidate`, `fingerprint_score`, `embedding_score`, `cover_score`, `match_type`, `risk_level`, `confidence`, `decision_reason`, `latency_ms`, `model_version`, `timestamp`

## 7. Rule Engine — 5 nhóm bản quyền + fallback

Đánh giá **tuần tự**, dừng ở nhóm đầu tiên khớp:

1. **AUDIO_LIBRARY**: `is_in_audio_library=TRUE` → risk LOW nếu không cần attribution, CONDITIONAL nếu cần attribution.
2. **CREATOR_MUSIC**: `is_in_creator_music=TRUE` + `license_purchased=TRUE` → LOW (giữ 100% doanh thu); hoặc `revenue_share_agreed=TRUE` → CONDITIONAL (revenue share).
3. **COMMERCIAL_CONTENT_ID**: `is_matched_content_id=TRUE` + `policy_action=MONETIZE_CLAIM` → CONDITIONAL (revenue redirected); + `policy_action=BLOCK_OR_TAKEDOWN` → HIGH.
4. **CREATIVE_COMMONS**: exact match + `CC_BY` + rights hợp lệ → CONDITIONAL (attribution required); `commercial_use=TRUE` + `CC_BY_NC` → HIGH (vi phạm non-commercial).
5. **PUBLIC_DOMAIN**: composition PD + recording PD → LOW; composition PD nhưng recording copyrighted → `PUBLIC_DOMAIN_COMPOSITION_ONLY`, risk CONDITIONAL/HIGH.
6. **Fallback**: `recording_status=UNKNOWN` hoặc `confidence < THRESHOLD_MIN` → UNCATEGORIZED, risk UNKNOWN, `HUMAN_REVIEW_REQUIRED`; nếu không khớp DB nào → `USER_GENERATED_CONTENT`, risk LOW, `MONETIZABLE_UNTIL_RETROACTIVE_CLAIM`.

Rules nên lưu dạng file cấu hình (`rules_v1.yaml`/json), có unit test riêng (tối thiểu 50–100 case).

## 8. API spec (MVP)

| Method | Path | Ghi chú |
|---|---|---|
| POST | `/api/v1/analyze` | Input: file, platform, commercial_use, monetization |
| GET | `/api/v1/jobs/{job_id}` | QUEUED / PROCESSING / DONE / FAILED |
| GET | `/api/v1/results/{job_id}` | identity / match / rights / assessment |
| GET | `/api/v1/tracks/{recording_id}` | |
| POST | `/api/v1/feedback` | Correct / Incorrect / Unsure |

**Error codes bắt buộc xử lý**: `FILE_TOO_LARGE`, `UNSUPPORTED_FORMAT`, `NO_AUDIO`, `NO_MUSIC`, `MODEL_FAILURE`, `DATABASE_FAILURE`, `TIMEOUT`, `UNKNOWN_TRACK`, `LOW_CONFIDENCE`. Không trả traceback ra frontend.

**Logging mỗi request**: `job_id`, `timestamp`, `file duration`, `fingerprint/embedding/vector-search/database/total latency`, `result`, `confidence`, `error`, `model version`. Không lưu file người dùng lâu dài nếu không cần.

## 9. Model & experiment versioning

Không ghi `model = MERT`. Phải ghi đầy đủ: `model_name`, `checkpoint`, `model_version`, `preprocessing_version`, `embedding_dimension`, `pooling_method`, `threshold`, `date_created`.

Mỗi experiment lưu: `experiment_id`, `git_commit`, `dataset_version`, `split_version`, `model`, `parameters`, `threshold`, `metrics`, `date`.

## 10. Dataset MVP (quy mô mục tiêu, không bắt buộc chính xác)

| Dataset | Vai trò | Số lượng |
|---|---|---|
| MTG-Jamendo (CC) | reference chính cho embedding/retrieval | 1.000–1.500 tracks |
| FMA (CC) | đa dạng reference, negative examples, retrieval eval | 500–1.000 tracks |
| YouTube Audio Library | case LOW/CONDITIONAL rõ ràng | 300–500 tracks |
| Creator Music | chỉ metadata mô phỏng cho Rule Engine, **không crawl audio** | 50–100 records |
| Public Domain | phải phân biệt composition status vs recording status | 200–300 recordings |
| Cover dataset (SecondHandSongs subset) | giai đoạn nâng cao | 100–200 compositions, 2–5 version/composition |

Metadata giữ tối thiểu: `track_id`, `artist`, `title`, `duration`, `tags/genre`, `source`, `license metadata`.

## 11. Data augmentation cho test set

Mỗi query gốc sinh nhiều biến thể để đánh giá robustness: crop, MP3/AAC compression, noise, voice overlay, gain, EQ, pitch shift (±1, ±2 semitone), tempo (0.9–1.10), re-encoding. Giữ một **Robustness Test Set cố định**, không nhất thiết dùng hết transformation để train.

## 12. Thí nghiệm bắt buộc (EXP-01 → EXP-08)

`EXP-01` Fingerprint baseline (Precision/Recall/F1/FPR/Latency) · `EXP-02` MERT retrieval (Recall@1/5/10, MRR, mAP) · `EXP-03` Pooling strategy (Mean vs Mean+Std) · `EXP-04` Fingerprint vs MERT vs Hybrid cascade — **thí nghiệm chính của đồ án** · `EXP-05` Robustness theo transformation · `EXP-06` Unknown track detection (bắt buộc — hệ thống phải biết từ chối nhận diện) · `EXP-07` Cover identification (nếu có module cover) · `EXP-08` End-to-end system (Macro-F1, class-wise P/R, confusion matrix, unknown detection rate, latency).

## 13. Ngưỡng nghiệm thu nội bộ (mục tiêu, không phải chuẩn ngành)

- Clean exact-match: Precision ≥ 0.95, Recall ≥ 0.90
- Robust retrieval: Recall@5 ≥ 0.80
- Unknown test: False Match Rate ≤ 5%
- End-to-end verified subset: Macro-F1 ≥ 0.80

Có thể điều chỉnh sau tuần 5 nếu baseline cho thấy dataset khó, nhưng phải ghi lý do.

## 14. Roadmap 12 tuần (milestone chính)

`M1` Data Ready (cuối T3) → `M2` Exact Matching Ready (cuối T4) → `M3` Deep Retrieval Ready (cuối T6) → `M4` Hybrid AI Ready (cuối T7, milestone quan trọng nhất) → `M5` Rights Intelligence Ready (cuối T9) → `M6` Full Application Ready (cuối T10) → `M7` Final Evaluation (T11) → `M8` Deployment (T12).

Khi ước lượng công việc "còn thiếu gì", đối chiếu với milestone hiện tại thay vì làm lan man sang các phần ở tuần sau.

## 15. Giao diện (4 màn hình MVP)

1. **Upload**: drag&drop, chọn platform/purpose/monetization, nút ANALYZE.
2. **Processing**: progress theo bước thật (audio extracted → fingerprint → searching DB → deep similarity → checking rights); không hiển thị "AI is thinking" mơ hồ.
3. **Result**: 4 phần — Identification / Rights / Assessment (🟢 LOW 🟡 CONDITIONAL 🔴 HIGH ⚪ UNKNOWN) / Recommendation.
4. **Evidence**: fingerprint score, embedding Top-K, matched track, metadata/license source, verification date, decision rules triggered.

Trang `/admin` (tùy chọn): xem DB, thêm recording, chỉnh rights, rebuild fingerprint/embedding, xem log, export kết quả experiment.

## 16. Phân công gợi ý (nhóm 4 người)

Data & Rights (dataset, metadata, rights DB, rule engine) · AI/MIR (Chromaprint, MERT, FAISS, retrieval, cover) · Backend/MLOps (FastAPI, PostgreSQL, job management, Docker, logging, deploy) · Frontend/Evaluation (UI/UX, API integration, test queries, experiments, visualization). Tất cả đều phải hiểu pipeline tổng thể.

## 17. Ngoài phạm vi MVP (không làm sớm)

Nhận diện hàng triệu bài, realtime livestream, sao chép Content ID của YouTube, kết luận pháp lý, auto-crawl toàn Internet, huấn luyện foundation model từ đầu, xử lý mọi loại mashup/remix phức tạp, Cover Detection/RAG/LLM/fine-tuning trước khi core pipeline hoạt động end-to-end.
