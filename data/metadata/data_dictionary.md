# Data Dictionary — `music_rights_ai`

Tài liệu bắt buộc theo CLAUDE.md §7: mọi thay đổi field cốt lõi của 7 bảng phải
được cập nhật ở đây. Nguồn sự thật của schema là `init_db.py`.

Cập nhật lần cuối: Giai đoạn 2 (thêm bảng jobs/feedback cho API §11; trước đó Giai đoạn 1 bổ sung 4 cột Rule Engine vào bảng `rights`).

---

## 1. `compositions` — Tác phẩm âm nhạc

| Field | Kiểu | Mô tả |
|---|---|---|
| `composition_id` | UUID PK | Định danh tác phẩm |
| `title` | TEXT | Tên tác phẩm |
| `composer` | TEXT | Người sáng tác |
| `year` | INTEGER | Năm sáng tác |
| `public_domain_status` | VARCHAR(50) | `verified` / `possible` / `no` / `unknown` — **PD của TÁC PHẨM** |
| `source` | TEXT | Nguồn metadata |
| `verified_at` | TIMESTAMP | Thời điểm xác minh |

> ⚠️ **§2:** `public_domain_status = verified` **KHÔNG** suy ra bản thu cũng PD.
> PD của bản thu nằm ở `rights.recording_public_domain` và được xét độc lập.

## 2. `recordings` — Bản thu cụ thể

| Field | Kiểu | Mô tả |
|---|---|---|
| `recording_id` | UUID PK | Định danh bản thu |
| `composition_id` | UUID FK | Trỏ tới `compositions` |
| `title`, `artist`, `album` | TEXT | Metadata hiển thị |
| `release_year` | INTEGER | Năm phát hành |
| `duration` | FLOAT | Độ dài (giây) |
| `source_dataset` | TEXT | `MTG-Jamendo` / `FMA` / `YouTube Audio Library` / `Creator Music` / `Public Domain Archive` |
| `source_track_id` | TEXT | ID gốc trong dataset nguồn (dùng để re-key embedding) |
| `audio_path` | TEXT | Đường dẫn audio. **Hiện là `simulated/...`: chưa có audio thật** |
| `metadata_verified` | BOOLEAN | Đã đối chiếu metadata chưa |

## 3. `rights` — Quyền sử dụng

| Field | Kiểu | Mô tả |
|---|---|---|
| `rights_id` | UUID PK | Định danh |
| `recording_id` | UUID FK | Trỏ tới `recordings` |
| `composition_id` | UUID FK | Trỏ tới `compositions` |
| `license_type` | VARCHAR(100) | `AUDIO_LIBRARY`, `CREATOR_MUSIC`, `CONTENT_ID`, `COMMERCIAL`, `CC_BY`, `CC_BY_SA`, `CC_BY_NC`, `CC_BY_NC_ND`, `PUBLIC_DOMAIN`, `UNKNOWN` |
| `copyright_status` | VARCHAR(50) | `PROTECTED` / `LICENSED` / `PUBLIC_DOMAIN` / `UNKNOWN` |
| `attribution_required` | BOOLEAN | Bắt buộc ghi công |
| `commercial_use_allowed` | BOOLEAN | Cho phép dùng thương mại |
| `modification_allowed` | BOOLEAN | Cho phép chỉnh sửa (ND = false) |
| `monetization_allowed` | BOOLEAN | Cho phép bật kiếm tiền |
| `revenue_share_required` | BOOLEAN | Bắt buộc chia doanh thu |
| `territory` | VARCHAR(50) | Phạm vi lãnh thổ |
| `platform` | VARCHAR(50) | Nền tảng áp dụng |
| `valid_from`, `valid_until` | DATE | Hiệu lực giấy phép |
| `source` | TEXT | Nguồn. Tiền tố `SIMULATED (...)` = metadata mô phỏng |
| `source_url` | TEXT | Link nguồn |
| `verified_at` | TIMESTAMP | Thời điểm xác minh |

### Cột thêm ở Giai đoạn 1 (phục vụ Rule Engine §9)

| Field | Kiểu | Vì sao cần |
|---|---|---|
| `policy_action` | VARCHAR(30) | Nhóm 3 Content ID: `MONETIZE_CLAIM` → CONDITIONAL, `BLOCK_OR_TAKEDOWN` → HIGH, `NONE` |
| `license_purchased` | BOOLEAN | Nhóm 2 Creator Music: đã mua giấy phép hợp lệ → LOW, giữ 100% doanh thu |
| `revenue_share_agreed` | BOOLEAN | Nhóm 2 Creator Music: đã đồng ý chia doanh thu → CONDITIONAL |
| `recording_public_domain` | BOOLEAN | Nhóm 5 Public Domain: PD của **bản thu**, xét ĐỘC LẬP với PD của tác phẩm |

## 4. `fingerprints` — Vân âm thanh

| Field | Kiểu | Mô tả |
|---|---|---|
| `fingerprint_id` | UUID PK | Tự sinh |
| `recording_id` | UUID FK | Trỏ tới `recordings` |
| `algorithm` | VARCHAR(50) | Mặc định `chromaprint` |
| `fingerprint` | TEXT | **Chuỗi base64 URL-safe đã nén** (đầu ra `fpcalc` KHÔNG có cờ `-raw`) |
| `duration` | FLOAT | Độ dài audio đã fingerprint |
| `created_at` | TIMESTAMP | Thời điểm tạo |

> ⚠️ Định dạng phải nhất quán: dùng `fpcalc -raw` sẽ cho mảng số nguyên và
> **không so khớp được** với các dòng base64 đang có.

## 5. `embeddings` — Vector MERT

| Field | Kiểu | Mô tả |
|---|---|---|
| `embedding_id` | UUID PK | Tự sinh |
| `recording_id` | UUID FK | **Phải là UUID**, không phải ID nguồn dataset |
| `segment_start`, `segment_end` | FLOAT | Biên đoạn (giây), mỗi đoạn 15s |
| `model` | VARCHAR(50) | `MERT` |
| `model_version` | VARCHAR(50) | `MERT-v1-95M` |
| `dimension` | INTEGER | 768 |
| `vector` | TEXT | 768 số thực phân tách bằng dấu phẩy, đã chuẩn hoá L2 |
| `created_at` | TIMESTAMP | Thời điểm tạo |

Thứ tự dòng của `embeddings_master.csv` **khớp 1-1** với thứ tự vector trong
`mert_faiss.index`; ánh xạ vị trí → recording nằm ở `data/processed/faiss_id_map.json`.

## 6. `test_queries` — Kế hoạch truy vấn kiểm thử

| Field | Kiểu | Mô tả |
|---|---|---|
| `query_id` | UUID PK | Định danh truy vấn |
| `recording_id` | UUID FK | Bản ghi gốc (nguồn để sinh truy vấn) |
| `composition_id` | UUID FK | Tác phẩm tương ứng |
| `transformation` | VARCHAR(100) | `original`, `crop_15s`, `mp3_128k`, `noise_snr10`, `pitch_plus_2`, `tempo_0_90`, `out_of_database`, … |
| `snr` | FLOAT | Tỉ lệ tín hiệu/nhiễu (dB), NULL nếu không thêm nhiễu |
| `pitch_shift` | FLOAT | Số nửa cung dịch cao độ |
| `tempo_factor` | FLOAT | Hệ số thay đổi tốc độ |
| `codec`, `bitrate` | VARCHAR/INT | Định dạng nén dùng khi dựng truy vấn |
| `segment_start`, `duration` | FLOAT | Đoạn được cắt |
| `expected_match_type` | VARCHAR(50) | `EXACT_MATCH` / `NEAR_MATCH` / `UNKNOWN` — nhãn đúng để chấm điểm |

## 7. `analysis_results` — Nhật ký phân tích

| Field | Kiểu | Mô tả |
|---|---|---|
| `job_id` | UUID PK | Định danh job |
| `query_id` | UUID | Trỏ tới `test_queries` khi chạy thí nghiệm |
| `recording_candidate` | UUID | Bản ghi được chọn (NULL nếu UNKNOWN) |
| `composition_candidate` | UUID | Tác phẩm tương ứng |
| `fingerprint_score` | FLOAT | Điểm Chromaprint |
| `embedding_score` | FLOAT | Cosine similarity cao nhất của MERT |
| `cover_score` | FLOAT | Dành cho module cover (chưa dùng) |
| `match_type` | VARCHAR(50) | `EXACT_MATCH` / `NEAR_MATCH` / `UNKNOWN` |
| `risk_level` | VARCHAR(50) | `LOW` / `CONDITIONAL` / `HIGH` / `UNKNOWN` |
| `confidence` | FLOAT | `identity_confidence` của tầng nhận diện |
| `decision_reason` | TEXT | Lý do quyết định dạng văn bản |
| `latency_ms` | FLOAT | Tổng độ trễ |
| `model_version` | VARCHAR(50) | Phiên bản model dùng để suy luận |
| `timestamp` | TIMESTAMP | Thời điểm ghi |

## 8. `jobs` — Vòng đời job phân tích (thêm ở Giai đoạn 2, phục vụ API §11)

| Field | Kiểu | Mô tả |
|---|---|---|
| `job_id` | UUID PK | Định danh job |
| `status` | VARCHAR(20) | `QUEUED` / `PROCESSING` / `DONE` / `FAILED` |
| `filename` | TEXT | Tên file người dùng tải lên |
| `platform` | VARCHAR(30) | `YOUTUBE` / `FACEBOOK` / `TIKTOK` / `OTHER` |
| `commercial_use` | BOOLEAN | Mục đích thương mại do người dùng khai báo |
| `monetization` | BOOLEAN | Có bật kiếm tiền không |
| `error_code` | VARCHAR(40) | Mã lỗi chuẩn hoá khi FAILED (§12) |
| `error_message` | TEXT | Thông điệp lỗi cho người dùng (không chứa traceback) |
| `result` | JSONB | Toàn bộ kết quả phân tích kèm evidence |
| `created_at`, `updated_at` | TIMESTAMP | Mốc thời gian |

## 9. `feedback` — Phản hồi người dùng (API §11)

| Field | Kiểu | Mô tả |
|---|---|---|
| `feedback_id` | UUID PK | Định danh |
| `job_id` | UUID | Job tương ứng |
| `recording_id` | UUID | Bản ghi được đánh giá |
| `verdict` | VARCHAR(20) | `Correct` / `Incorrect` / `Unsure` |
| `note` | TEXT | Ghi chú tự do (tối đa 2000 ký tự) |
| `created_at` | TIMESTAMP | Thời điểm gửi |

## 10. `unknown_tracks` — Sổ bài chưa nhận diện (tab "Thống kê & Tra cứu")

Mỗi dòng là một bài. Không có khoá ngoại tới `recordings` (`init_db.py` chạy
`TRUNCATE recordings CASCADE` khi nạp lại dữ liệu). Không đi vào Rule Engine.

| Field | Kiểu | Mô tả |
|---|---|---|
| `unknown_id` | UUID PK | Định danh mục |
| `kind` | VARCHAR(20) | `NOT_IDENTIFIED` (không khớp bản ghi nào) / `RIGHTS_UNKNOWN` (nhận diện được, rủi ro UNKNOWN) |
| `recording_id` | UUID | Chỉ với `RIGHTS_UNKNOWN`; duy nhất trong nhóm này (`ux_unknown_tracks_recording`) |
| `fingerprint` | TEXT | Chromaprint base64 của lượt gặp đầu — dùng để gộp lượt gặp sau (chỉ `NOT_IDENTIFIED`) |
| `fingerprint_duration` | FLOAT | Độ dài đoạn đã sinh fingerprint (giây) |
| `first_filename` | TEXT | Tên file (đã lọc) của lượt gặp đầu |
| `first_seen_at`, `last_seen_at` | TIMESTAMPTZ | Lần đầu / gần nhất gặp bài này |
| `sighting_count` | INTEGER | Số lượt gặp |
| `last_match_type`, `last_risk_level`, `last_category`, `last_condition` | VARCHAR | Ảnh chụp kết quả của lượt gặp gần nhất |
| `hint_recording_id`, `hint_score` | UUID, FLOAT | Ứng viên MERT đứng đầu nhưng DƯỚI ngưỡng — gợi ý cho người thẩm định, không phải kết luận |
| `review_status` | VARCHAR(20) | `PENDING` / `REVIEWED` |
| `reviewer_title`, `reviewer_artist`, `reviewer_note` | VARCHAR(255), VARCHAR(255), TEXT | Ghi chú của người thẩm định — chỉ để tra cứu, **không phải dữ liệu quyền** |
| `reviewed_at` | TIMESTAMPTZ | Thời điểm chuyển sang `REVIEWED` |

## 11. `unknown_sightings` — Từng lượt gặp

| Field | Kiểu | Mô tả |
|---|---|---|
| `sighting_id` | UUID PK | Định danh |
| `unknown_id` | UUID FK → `unknown_tracks` (ON DELETE CASCADE) | Mục trong sổ |
| `job_id` | UUID | Job sinh ra lượt gặp — chỉ để kiểm toán, API **không** trả trường này |
| `seen_at` | TIMESTAMPTZ | Thời điểm |
| `platform`, `commercial_use`, `monetization` | VARCHAR(30), BOOLEAN, BOOLEAN | Ngữ cảnh người dùng khai báo |
| `match_score` | FLOAT | Điểm của tầng đã nhận ra lượt này (NULL ở lượt đầu / `RIGHTS_UNKNOWN`) |
| `match_method` | VARCHAR(20) | `FINGERPRINT` / `MERT` / `COVER` (NULL = mục mới) |
| `risk_level`, `category` | VARCHAR | Kết quả Rule Engine của lượt này |

## 12. `unknown_track_features` — Đặc trưng sổ đã "học"

| Field | Kiểu | Mô tả |
|---|---|---|
| `feature_id` | UUID PK | Định danh |
| `unknown_id` | UUID FK → `unknown_tracks` (ON DELETE CASCADE) | Mục trong sổ (chỉ `NOT_IDENTIFIED`) |
| `feature_type` | VARCHAR(10) | `MERT` (đoạn 15 s) / `COVER` (30 giây đầu) |
| `model_version` | VARCHAR(80) | Phiên bản đặc trưng; khác bản đang chạy thì không đem so |
| `segment_start`, `segment_end` | FLOAT | Đoạn trong lần gửi đã sinh vector (giây) |
| `dimension` | INTEGER | 768 |
| `vector` | BYTEA | float32 little-endian — không dựng lại được âm thanh |
| `created_at` | TIMESTAMPTZ | Thời điểm học |

---

## Ghi chú về tình trạng dữ liệu hiện tại

Cập nhật 2026-09-19, đếm từ các CSV trong `data/processed/` (158.117 dòng mỗi bảng
`compositions` / `recordings` / `rights`):

- **Dữ liệu thật — 24.375 bản ghi FMA:** `audio_path` là đường dẫn tương đối tới file
  audio thật (vd. `fma/000002.mp3`, gốc là `AUDIO_ROOT`), giấy phép Creative Commons do
  FMA công bố (`rights.source` bắt đầu bằng `FMA`), `metadata_verified = TRUE` — nhóm
  duy nhất có cờ này. Cả 24.375 bản ghi có fingerprint thật của `fpcalc` và 2 vector MERT
  (48.750 vector, file `embeddings_master.csv` không nằm trong repo).
- **Metadata mô phỏng — 130.381 dòng `rights.source` bắt đầu bằng `SIMULATED`:**
  dataset_G nhạc Việt 100.000 và MTG-Jamendo 29.881 (`audio_path` là URL, không có
  audio), YouTube Audio Library 400 và Creator Music 100 (`audio_path` trống). Sinh bằng
  `scripts/enrich_rights_metadata.py`, gắn lại nhãn trung thực bằng
  `scripts/relabel_simulated_sources.py`; quy tắc gắn nhãn có test trong
  `tests/test_honest_labels.py`.
- **Spotify Web API — 3.361 dòng:** `audio_path` là URL, hãng phát hành chưa xác minh.
- Không bản ghi nào ngoài FMA có fingerprint hay embedding; `rights_gate` của Rule Engine
  hạ mọi kết luận dựa trên metadata mô phỏng chưa có `verified_at` về `UNKNOWN`.
