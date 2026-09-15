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
