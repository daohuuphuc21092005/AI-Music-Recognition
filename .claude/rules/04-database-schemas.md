---
description: Schema cơ sở dữ liệu quan hệ PostgreSQL, bảng vector embedding và các bảng lịch sử phân tích
globs:
  - "backend/database/**"
  - "backend/schemas/**"
  - "alembic/**"
---

# 04. Đặc tả Cơ sở dữ liệu (Database Schema Specifications)

Hệ thống sử dụng **PostgreSQL** làm cơ sở dữ liệu quan hệ chính. Các bảng được thiết kế để tách biệt rõ ràng giữa tác phẩm âm nhạc (`compositions`) và các bản thu âm cụ thể (`recordings`), cùng với dữ liệu quyền (`rights`), dấu vân tay âm thanh (`fingerprints`), và vector embedding.

---

## 1. Bảng `compositions` (Tác phẩm / Sáng tác)
Lưu thông tin về tác phẩm gốc, giai điệu, lời bài hát do nhạc sĩ/nhà soạn nhạc sáng tác.

| Tên trường | Kiểu dữ liệu | Ràng buộc | Mô tả |
|---|---|---|---|
| `composition_id` | UUID | PRIMARY KEY | Định danh duy nhất của tác phẩm |
| `title` | VARCHAR(255) | NOT NULL | Tên tác phẩm gốc |
| `composer` | VARCHAR(255) | NULL | Nhạc sĩ / Nhà soạn nhạc |
| `year` | INTEGER | NULL | Năm sáng tác |
| `public_domain_status` | VARCHAR(50) | NOT NULL | Trạng thái PD (`verified`, `possible`, `no`, `unknown`) |
| `source` | VARCHAR(100) | NULL | Nguồn dữ liệu (ví dụ: `IMSLP`, `MusicBrainz`) |
| `verified_at` | TIMESTAMP | NULL | Thời điểm xác minh trạng thái |

---

## 2. Bảng `recordings` (Bản ghi âm cụ thể)
Lưu thông tin về một bản thu cụ thể của một tác phẩm. Một `composition` có thể có nhiều `recordings`.

| Tên trường | Kiểu dữ liệu | Ràng buộc | Mô tả |
|---|---|---|---|
| `recording_id` | UUID | PRIMARY KEY | Định danh duy nhất của bản ghi |
| `composition_id` | UUID | FOREIGN KEY → `compositions.composition_id` | Liên kết tới tác phẩm gốc |
| `title` | VARCHAR(255) | NOT NULL | Tiêu đề của bản ghi âm |
| `artist` | VARCHAR(255) | NOT NULL | Nghệ sĩ / Ban nhạc biểu diễn |
| `album` | VARCHAR(255) | NULL | Tên album |
| `release_year` | INTEGER | NULL | Năm phát hành bản ghi |
| `duration` | FLOAT | NOT NULL | Thời lượng bản ghi (tính theo giây) |
| `source_dataset` | VARCHAR(100) | NOT NULL | Nguồn dataset (MTG-Jamendo, FMA, YouTube, ...) |
| `source_track_id` | VARCHAR(100) | NULL | ID gốc trong dataset nguồn |
| `audio_path` | VARCHAR(512) | NOT NULL | Đường dẫn lưu trữ file audio trên ổ đĩa/storage |
| `metadata_verified` | BOOLEAN | DEFAULT FALSE | Cờ xác nhận độ tin cậy của metadata |

---

## 3. Bảng `rights` (Thông tin pháp lý & Giấy phép)
Lưu trữ các điều khoản bản quyền gắn với từng bản ghi âm và tác phẩm.

| Tên trường | Kiểu dữ liệu | Ràng buộc | Mô tả |
|---|---|---|---|
| `rights_id` | UUID | PRIMARY KEY | Định danh bản ghi quyền |
| `recording_id` | UUID | FOREIGN KEY → `recordings.recording_id` | Bản thu được áp dụng |
| `composition_id` | UUID | FOREIGN KEY → `compositions.composition_id` | Tác phẩm được áp dụng |
| `license_type` | VARCHAR(100) | NOT NULL | `CC_BY`, `CC_BY_NC`, `PUBLIC_DOMAIN`, `YOUTUBE_AUDIO_LIBRARY`, `CREATOR_MUSIC`, ... |
| `copyright_status` | VARCHAR(50) | NOT NULL | `PROTECTED`, `PUBLIC_DOMAIN`, `UNKNOWN` |
| `attribution_required` | BOOLEAN | DEFAULT FALSE | Bắt buộc ghi công tác giả hay không |
| `commercial_use_allowed` | BOOLEAN | DEFAULT FALSE | Cho phép sử dụng cho mục đích thương mại |
| `modification_allowed` | BOOLEAN | DEFAULT FALSE | Cho phép chỉnh sửa / phối lại / remix |
| `monetization_allowed` | BOOLEAN | DEFAULT FALSE | Cho phép bật kiếm tiền trên nền tảng |
| `revenue_share_required` | BOOLEAN | DEFAULT FALSE | Yêu cầu chia sẻ doanh thu với chủ bản quyền |
| `territory` | VARCHAR(100) | DEFAULT 'GLOBAL' | Phạm vi lãnh thổ áp dụng |
| `platform` | VARCHAR(100) | DEFAULT 'ALL' | Nền tảng áp dụng (`YouTube`, `TikTok`, `Facebook`, `ALL`) |
| `valid_from` | TIMESTAMP | NULL | Ngày giấy phép bắt đầu có hiệu lực |
| `valid_until` | TIMESTAMP | NULL | Ngày giấy phép hết hiệu lực |
| `source` | VARCHAR(100) | NULL | Đơn vị / nền tảng cấp thông tin bản quyền |
| `source_url` | VARCHAR(512) | NULL | Đường dẫn xác thực thông tin giấy phép |
| `verified_at` | TIMESTAMP | NULL | Thời điểm kiểm tra thông tin |

---

## 4. Bảng `fingerprints` (Dấu vân tay âm thanh Chromaprint)
Lưu trữ fingerprint âm thanh phục vụ so khớp chính xác (Exact Match).

| Tên trường | Kiểu dữ liệu | Ràng buộc | Mô tả |
|---|---|---|---|
| `fingerprint_id` | UUID | PRIMARY KEY | Định danh fingerprint |
| `recording_id` | UUID | FOREIGN KEY → `recordings.recording_id` | Liên kết bản ghi |
| `algorithm` | VARCHAR(50) | DEFAULT 'chromaprint' | Thuật toán sinh fingerprint |
| `fingerprint` | TEXT | NOT NULL | Chuỗi hash fingerprint sinh bởi `fpcalc` |
| `duration` | FLOAT | NOT NULL | Thời lượng đoạn trích xuất fingerprint |
| `created_at` | TIMESTAMP | DEFAULT NOW() | Thời điểm tạo |

---

## 5. Bảng `embeddings` (Vector Embedding MERT)
Lưu trữ vector đại diện âm thanh phục vụ tìm kiếm tương đồng ngữ nghĩa âm nhạc.

| Tên trường | Kiểu dữ liệu | Ràng buộc | Mô tả |
|---|---|---|---|
| `embedding_id` | UUID | PRIMARY KEY | Định danh vector |
| `recording_id` | UUID | FOREIGN KEY → `recordings.recording_id` | Liên kết bản ghi |
| `segment_start` | FLOAT | NOT NULL | Giây bắt đầu của đoạn trích xuất |
| `segment_end` | FLOAT | NOT NULL | Giây kết thúc của đoạn trích xuất |
| `model` | VARCHAR(100) | NOT NULL | Ví dụ: `MERT-v1-95M` |
| `model_version` | VARCHAR(50) | NOT NULL | Phiên bản checkpoint của model |
| `dimension` | INTEGER | NOT NULL | Kích thước vector (ví dụ: 768) |
| `vector` | FLOAT[] / VECTOR | NOT NULL | Vector đặc trưng (hỗ trợ pgvector hoặc FAISS/Qdrant) |
| `created_at` | TIMESTAMP | DEFAULT NOW() | Thời điểm sinh embedding |

---

## 6. Bảng `test_queries` (Tập câu hỏi kiểm thử / Robustness Queries)
Lưu các truy vấn phục vụ đo lường và đánh giá hiệu năng hệ thống.

| Tên trường | Kiểu dữ liệu | Ràng buộc | Mô tả |
|---|---|---|---|
| `query_id` | UUID | PRIMARY KEY | Định danh query kiểm thử |
| `recording_id` | UUID | NULL (FK) | Ground truth recording (nếu có) |
| `composition_id` | UUID | NULL (FK) | Ground truth composition (nếu có) |
| `transformations` | JSONB | NOT NULL | Danh sách biến đổi áp dụng (noise, pitch, tempo, ...) |
| `snr` | FLOAT | NULL | Tỷ lệ tín hiệu trên nhiễu (Signal-to-Noise Ratio) |
| `pitch_shift` | FLOAT | NULL | Mức độ dịch cao độ (semitones) |
| `tempo_factor` | FLOAT | NULL | Hệ số thay đổi tốc độ |
| `codec` | VARCHAR(20) | NULL | Định dạng nén (`mp3`, `aac`, ...) |
| `bitrate` | VARCHAR(20) | NULL | Tốc độ bit (`64k`, `128k`, `320k`) |
| `segment_start` | FLOAT | NOT NULL | Điểm bắt đầu cắt từ file gốc |
| `duration` | FLOAT | NOT NULL | Thời lượng file query (giây) |
| `expected_match_type` | VARCHAR(50) | NOT NULL | Kết quả kỳ vọng (`EXACT`, `EMBEDDING`, `UNKNOWN`) |

---

## 7. Bảng `analysis_results` (Kết quả phân tích tác vụ)
Lưu toàn bộ lịch sử các tác vụ phân tích để theo dõi chất lượng và audit log.

| Tên trường | Kiểu dữ liệu | Ràng buộc | Mô tả |
|---|---|---|---|
| `job_id` | UUID | PRIMARY KEY | Định danh tác vụ phân tích |
| `query_id` | UUID | NULL (FK) | Liên kết tới `test_queries` nếu là bài test |
| `recording_candidate` | UUID | NULL (FK) | Bản ghi được dự đoán khớp nhất |
| `composition_candidate` | UUID | NULL (FK) | Tác phẩm được dự đoán |
| `fingerprint_score` | FLOAT | NULL | Điểm tương đồng Chromaprint |
| `embedding_score` | FLOAT | NULL | Cosine similarity từ MERT |
| `cover_score` | FLOAT | NULL | Điểm cover detection (nếu có) |
| `match_type` | VARCHAR(50) | NOT NULL | `EXACT_MATCH`, `NEAR_EXACT_MATCH`, `EMBEDDING_MATCH`, `NO_MATCH` |
| `risk_level` | VARCHAR(20) | NOT NULL | `LOW`, `CONDITIONAL`, `HIGH`, `UNKNOWN` |
| `confidence` | FLOAT | NOT NULL | Điểm tin cậy tổng thể (0.0 – 1.0) |
| `decision_reason` | TEXT | NOT NULL | Chuỗi quy tắc và giải thích quyết định |
| `latency_ms` | INTEGER | NOT NULL | Tổng thời gian xử lý (milliseconds) |
| `model_version` | VARCHAR(100) | NOT NULL | Phiên bản các module AI sử dụng |
| `timestamp` | TIMESTAMP | DEFAULT NOW() | Thời điểm hoàn thành phân tích |

---

## 8. Bảng `unknown_tracks` (Sổ bài chưa nhận diện)
Mỗi kết quả UNKNOWN của `/analyze` được gộp vào một mục (tab "Thống kê & Tra cứu"). **Không có khoá ngoại tới `recordings`**: `init_db.py` chạy `TRUNCATE recordings CASCADE` khi nạp lại dữ liệu, khoá ngoại sẽ xoá sạch sổ theo. Sổ **không bao giờ** đi vào Rule Engine.

| Tên trường | Kiểu dữ liệu | Ràng buộc | Mô tả |
|---|---|---|---|
| `unknown_id` | UUID | PRIMARY KEY | Định danh mục |
| `kind` | VARCHAR(20) | NOT NULL | `NOT_IDENTIFIED` (match.type = UNKNOWN) / `RIGHTS_UNKNOWN` (nhận diện được, risk = UNKNOWN) |
| `recording_id` | UUID | NULL, UNIQUE khi `kind='RIGHTS_UNKNOWN'` | Bản ghi trong kho (chỉ `RIGHTS_UNKNOWN`) |
| `fingerprint` | TEXT | NULL | Chromaprint base64 của lượt gặp đầu — gộp lượt sau bằng `min_score_for_duration` của tầng 1 |
| `fingerprint_duration` | FLOAT | NULL | Độ dài đoạn sinh fingerprint (giây) |
| `first_filename` | TEXT | NULL | Tên file đã lọc của lượt gặp đầu |
| `first_seen_at` / `last_seen_at` | TIMESTAMPTZ | DEFAULT NOW() | Lần đầu / gần nhất |
| `sighting_count` | INTEGER | NOT NULL DEFAULT 1 | Số lượt gặp |
| `last_match_type`, `last_risk_level`, `last_category`, `last_condition` | VARCHAR | NULL | Ảnh chụp kết quả lượt gần nhất |
| `hint_recording_id`, `hint_score` | UUID, FLOAT | NULL | Ứng viên MERT top-1 dưới ngưỡng — gợi ý, không phải kết luận |
| `review_status` | VARCHAR(20) | NOT NULL DEFAULT 'PENDING' | `PENDING` / `REVIEWED` |
| `reviewer_title`, `reviewer_artist`, `reviewer_note` | VARCHAR(255), VARCHAR(255), TEXT | NULL | Ghi chú thẩm định — chỉ để tra cứu, **không phải dữ liệu quyền** |
| `reviewed_at` | TIMESTAMPTZ | NULL | Thời điểm chuyển sang REVIEWED |

Index: `ix_unknown_tracks_last_seen (last_seen_at DESC)`, `ux_unknown_tracks_recording (recording_id) WHERE kind='RIGHTS_UNKNOWN'`.

---

## 9. Bảng `unknown_sightings` (Từng lượt gặp)

| Tên trường | Kiểu dữ liệu | Ràng buộc | Mô tả |
|---|---|---|---|
| `sighting_id` | UUID | PRIMARY KEY | Định danh |
| `unknown_id` | UUID | FOREIGN KEY → `unknown_tracks` ON DELETE CASCADE | Mục trong sổ |
| `job_id` | UUID | NULL | Job sinh ra lượt gặp — chỉ để kiểm toán, **API không trả** (job_id là khoá truy cập) |
| `seen_at` | TIMESTAMPTZ | DEFAULT NOW() | Thời điểm |
| `platform`, `commercial_use`, `monetization` | VARCHAR(30), BOOLEAN, BOOLEAN | NULL | Ngữ cảnh người dùng khai báo |
| `match_score` | FLOAT | NULL | Điểm của tầng đã nhận ra lượt này (xem `match_method`) |
| `match_method` | VARCHAR(20) | NULL | `FINGERPRINT` / `MERT` / `COVER` — tầng nào của sổ nhận ra; NULL = mục mới |
| `risk_level`, `category` | VARCHAR | NULL | Kết quả Rule Engine của lượt này |

Index: `ix_unknown_sightings_entry (unknown_id, seen_at DESC)`, `ix_unknown_sightings_seen (seen_at)`.

---

## 10. Bảng `unknown_track_features` (Đặc trưng sổ đã "học" — 2026-09-27)
Mỗi mục `NOT_IDENTIFIED` lưu vector MERT từng đoạn 15 s và descriptor Cover (30 giây đầu), để lần sau nhận ra cả bản đã nén, cắt, đổi tông (`backend/services/registry_features.py`). **Không lưu audio.** Chỉ THÊM phần đặc trưng mới, tức chưa nhận ra được bằng τMERT / τCover với những gì mục đã có.

| Tên trường | Kiểu dữ liệu | Ràng buộc | Mô tả |
|---|---|---|---|
| `feature_id` | UUID | PRIMARY KEY | Định danh |
| `unknown_id` | UUID | FOREIGN KEY → `unknown_tracks` ON DELETE CASCADE | Mục trong sổ |
| `feature_type` | VARCHAR(10) | NOT NULL | `MERT` / `COVER` |
| `model_version` | VARCHAR(80) | NOT NULL | vd. `MERT-v1-95M@12af15fef9d0/mean/15s`, `chroma_cqt_64f_meansub_oti_v1`. Khác phiên bản đang chạy thì không đem so (§9 CLAUDE.md) |
| `segment_start`, `segment_end` | FLOAT | NULL | Đoạn trong lần gửi đã sinh vector (giây) |
| `dimension` | INTEGER | NOT NULL | 768 (MERT) / 768 (Cover 12 × 64) |
| `vector` | BYTEA | NOT NULL | float32 little-endian (3 KB; TEXT như bảng `embeddings` là ~16 KB) |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | |

Index: `ix_unknown_features_entry (unknown_id, feature_type)`. Trần mỗi mục: `UNKNOWN_REGISTRY_MAX_FEATURES_PER_ENTRY` vector MERT (48), 8 descriptor Cover. Role `music_ai_app` chỉ có `SELECT, INSERT`.

**Nâng cấp CSDL đang chạy**: `python init_db.py --schema-only` (chỉ tạo phần còn thiếu, không TRUNCATE, không cần pandas), rồi chạy lại `scripts/sql/create_app_role.sql` để cấp `SELECT, INSERT, UPDATE` cho `music_ai_app`.
