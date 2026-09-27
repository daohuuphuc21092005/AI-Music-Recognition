---
description: Tiêu chuẩn thiết kế Backend FastAPI, kiến trúc Service layer, đặc tả REST API, mã lỗi và chuẩn logging
globs:
  - "backend/**"
---

# 05. Chuẩn Backend & Đặc tả API (Backend & API Standards)

Backend được xây dựng trên nền tảng **FastAPI** (Python 3.10+), tuân thủ kiến trúc phân tầng (Service Layer Pattern) và quản lý tác vụ bất đồng bộ cho quá trình xử lý audio nặng.

---

## 1. Cấu trúc Service Layer bắt buộc

Mọi logic nghiệp vụ phải được tổ chức trong các service độc lập tại `backend/services/`:

1. **`AudioService`**: Chịu trách nhiệm validate file, giải nén/trích xuất audio từ video bằng FFmpeg, chuẩn hóa định dạng/sampling rate, cắt segment và dọn dẹp file tạm.
2. **`FingerprintService`**: Thực thi `fpcalc` (Chromaprint), sinh và so khớp hash fingerprint.
3. **`EmbeddingService`**: Khởi tạo và quản lý model `MERT-v1-95M`, xử lý batch inference trên GPU/CPU, thực thi pooling (Mean, Mean+Std), và chuẩn hóa L2 vector.
4. **`RetrievalService`**: Giao tiếp với FAISS / Qdrant, thực hiện tìm kiếm vector Top-K, áp ngưỡng `τMERT`, và xếp hạng ứng viên (candidate ranking).
5. **`CoverService`** *(Optional)*: Xử lý baseline CQT/Chroma alignment hoặc CoverHunter nếu được kích hoạt.
6. **`RightsService`**: Truy vấn cơ sở dữ liệu `recordings`, `compositions`, và `rights` dựa trên `recording_id`, tổng hợp dữ liệu bản quyền.
7. **`DecisionService`**: Nạp file cấu hình rules, nạp Evidence Object, thực thi cây quyết định tuần tự, xác định `risk_level`, `conditions`, và sinh `recommendations`.

---

## 2. Đặc tả API Endpoints (MVP)

| HTTP Method | Endpoint | Tham số đầu vào | Định dạng kết quả | Ghi chú |
|---|---|---|---|---|
| **POST** | `/api/v1/analyze` | Form-data: `file` (audio/video), `platform` (YouTube/TikTok/ALL), `commercial_use` (bool), `monetization` (bool) | `{"job_id": "UUID", "status": "QUEUED"}` | Khởi tạo tác vụ phân tích bất đồng bộ |
| **GET** | `/api/v1/jobs/{job_id}` | Path: `job_id` | `{"job_id": "UUID", "status": "QUEUED\|PROCESSING\|DONE\|FAILED", "progress": 0.65, "current_step": "CHECKING_RIGHTS"}` | Kiểm tra tiến độ tác vụ |
| **GET** | `/api/v1/results/{job_id}` | Path: `job_id` | Full analysis object (Identity, Rights, Risk, Evidence, Recommendation) | Chỉ khả dụng khi job có status `DONE` |
| **GET** | `/api/v1/tracks/{recording_id}` | Path: `recording_id` | Metadata chi tiết của track, composition và rights | Xem chi tiết bản ghi trong kho dữ liệu |
| **POST** | `/api/v1/feedback` | JSON: `job_id`, `feedback_type` (`CORRECT\|INCORRECT\|UNSURE`), `notes` | `{"status": "recorded"}` | Thu thập phản hồi từ người dùng |
| **GET** | `/api/v1/unknown-tracks/stats` | Query: `utc_offset_min` (−720…840) | `{registry_enabled, totals, by_kind, by_last_risk, by_platform, sightings_by_day[30], top_resighted, jobs}` | Thống kê sổ bài chưa nhận diện; chia ngày theo giờ người xem |
| **GET** | `/api/v1/unknown-tracks` | Query: `q` (≤100), `kind`, `status`, `sort` (`last_seen\|first_seen\|sightings`), `limit` (1–100), `offset` (0–10000) | `{total, limit, offset, items}` | Tra cứu sổ (tìm theo tên bài/nghệ sĩ/ghi chú/tên file hoặc UUID) |
| **GET** | `/api/v1/unknown-tracks/{unknown_id}` | Path: `unknown_id` | Mục + `catalogue`, `hint`, 50 `sightings` gần nhất | Không bao giờ trả `job_id` |
| **PATCH** | `/api/v1/unknown-tracks/{unknown_id}` | JSON: `title`, `artist`, `note`, `review_status` (trường nào gửi mới đổi; chuỗi rỗng = xoá) | Mục sau khi sửa | Ghi chú thẩm định — không phải dữ liệu quyền |

---

## 3. Bảng mã lỗi chuẩn (Mandatory Error Codes)

**Tuyệt đối không trả traceback nội bộ của Python / PyTorch / FFmpeg ra phía client/frontend.** Mọi ngoại lệ phải được bắt và ánh xạ về cấu trúc lỗi chuẩn:

```json
{
  "error_code": "UNSUPPORTED_FORMAT",
  "message": "The uploaded file format is not supported. Please provide MP3, WAV, M4A, MP4, or MOV.",
  "details": null
}
```

**Khuôn lỗi thực tế trên dây**: FastAPI bọc trong `detail` → `{"detail": {"error_code": "...", "message": "..."}}`. Frontend (`apiError` trong `frontend/app.js`) và `tests/test_api.py` đọc đúng khuôn này — đổi khuôn thì phải sửa cả hai.

### Cách backend giữ cam kết "không lộ thông tin nội bộ"
- **Lưới an toàn toàn cục** (`backend/main.py::unhandled_exception`): mọi ngoại lệ không lường trước → HTTP 500 `INTERNAL_ERROR` cùng khuôn trên; chi tiết chỉ nằm trong log máy chủ.
- **Tham số sai cũng theo khuôn** (`backend/main.py::invalid_request`): `RequestValidationError` KHÔNG đi qua lưới toàn cục — mặc định FastAPI trả `{"detail": [...]}` (danh sách, frontend không đọc được mã, trường `input` lặp lại giá trị người dùng gửi). Nay → HTTP 422 `INVALID_REQUEST`, `message` nêu tên trường + giá trị được chấp nhận, `details` = `[{field, message}]`, không lặp lại input.
- **Kiểm UUID trước khi chạm CSDL** (`routes.parse_uuid`): `job_id` / `recording_id` sai dạng → 404 `UNKNOWN_TRACK`, không để PostgreSQL ném lỗi ép kiểu.
- **Lỗi FILE và lỗi MÁY CHỦ phải ra hai mã khác nhau**: `embedding_service` ném `EmbeddingAudioError` (file không giải mã được → `NO_AUDIO`) hoặc `EmbeddingModelError` (không nạp/chạy được MERT → `MODEL_FAILURE`); không còn trả `None`. `fpcalc` không đọc được file (`AudioFingerprintError`) → `NO_AUDIO`.
- **Không đưa nội dung ngoại lệ ra response**: stderr của FFmpeg và lỗi tầng Cover chỉ ghi log; evidence của Cover chỉ mang mã `COVER_STAGE_FAILED`.
- **`/health` là endpoint công khai**: không trả `fpcalc_path` hay nội dung lỗi nạp index/rules — chỉ mã `INDEX_LOAD_FAILED` / `RULES_LOAD_FAILED`.
- **Làm nóng lúc khởi động** (`backend/main.py::warm_up`, `WARMUP_ON_STARTUP`): luồng nền nạp bộ đệm fingerprint tham chiếu, MERT, chỉ mục Cover. `/health.warmup` = `{status: DISABLED|RUNNING|DONE, steps: {fingerprint|mert|cover: {status: READY|FAILED|SKIPPED, seconds}}}` — chỉ trạng thái và số giây, lỗi của từng bước chỉ vào log. Ba bộ đệm có khoá (double-checked) để request đến giữa lúc làm nóng chờ chứ không nạp lần hai (MERT hai bản trên GPU 4 GB). Bộ test tắt làm nóng trong `tests/conftest.py`.
- **Ghi FAILED không được làm hỏng luồng lỗi** (`routes.safe_mark_failed`): CSDL sập thì ghi log, không ném tiếp.
- **Không có mật khẩu CSDL mặc định trong mã nguồn**: thiếu `DATABASE_URL` thì kết nối thất bại, log khởi động báo rõ và `/health` ra `DEGRADED`. Frontend đọc `/health.components.database` (và `rule_engine`), rồi khoá nút phân tích kèm lý do ngay ở màn Tải lên, thay vì để người dùng nhận `DATABASE_FAILURE` sau khi chờ tải file lên.
- **Thiếu FFmpeg** thì video và `.m4a`/`.aac` bị từ chối bằng `MODEL_FAILURE`, câu lỗi gợi ý dùng mp3/wav/flac. Frontend chặn trước theo `/health.components.ffmpeg`.
- **Thiếu chỉ mục FAISS** thì mọi bài không khớp ở tầng 1 ra `MODEL_FAILURE`. Câu lỗi chỉ tới `scripts/rebuild_faiss_index.py --from-db`: dựng từ bảng `embeddings`, không cần `embeddings_master.csv`.
- Endpoint gọi mã chặn (`/analyze`, `/search`) khai báo `def` chứ không `async def`, để FastAPI đẩy sang threadpool thay vì chặn event loop.

### Sổ bài chưa nhận diện (`backend/services/unknown_registry_service.py`)
- **Ghi ở `routes.run_job` → `attach_unknown_registry`**, SAU Rule Engine và TRƯỚC `mark_done`, khi `classify_unknown(result)` khác None: `match.type == UNKNOWN` → `NOT_IDENTIFIED`; hoặc `assessment.risk == UNKNOWN` và có `recording_id` → `RIGHTS_UNKNOWN`. `UNAVAILABLE` không vào sổ. Bước này báo stage **`REGISTRY_LOOKUP`** (có trong `PipelineStage`).
- **Không đụng tới `assessment` / `recommendation`**: kết quả sổ nằm ở `evidence.unknown_registry` (`unknown_id`, `kind`, `is_new`, `sighting_count`, `first_seen_at`, `match_score`, ghi chú thẩm định, `elapsed_ms`, `note`). Test khoá bất biến này: `tests/test_unknown_registry.py`.
- **Sổ lỗi thì job vẫn DONE**: evidence chỉ mang `{"error_code": "REGISTRY_FAILED", "kind": ...}`, nội dung lỗi chỉ vào log.
- `POST /search` **không** ghi sổ (không lưu job; dùng cho demo và script thí nghiệm).
- Gộp lượt gặp: fingerprint của file gốc so với tối đa `UNKNOWN_REGISTRY_SCAN_LIMIT` mục mới nhất bằng `match_decoded`, chấp nhận khi điểm ≥ `min_score_for_duration(đoạn ngắn hơn)`; `pg_advisory_xact_lock` chặn hai job cùng bài tạo hai mục. Tắt toàn bộ bằng `UNKNOWN_REGISTRY_ENABLED=false`.
- **Sổ "học" bài UNKNOWN (2026-09-27, chủ dự án chọn "ghi nhớ đặc trưng vào sổ, tự động mọi bài")**:
  - **Học** = lưu đặc trưng (`registry_features`, bảng `unknown_track_features`), KHÔNG huấn luyện model (§2.9). Bài chưa từng gặp thì vẫn không nhận ra được.
  - **Đặc trưng** tính đúng cách chỉ mục chính, để ngưỡng đã hiệu chỉnh giữ nguyên nghĩa:
    - MERT từng đoạn 15 s, bước 10 s (`scripts/build_embeddings.py`), tối đa `UNKNOWN_REGISTRY_MAX_SEGMENTS` đoạn rải đều.
    - Cover: 30 giây đầu làm reference, truy vấn cắt theo `COVER_TEMPO_FACTORS`.
    - Tính TRƯỚC khoá advisory (`record_unknown`); m4a/video giải mã lại qua FFmpeg.
  - **Nhận lại** theo thứ tự cascade, dừng ở tầng đầu khớp: fingerprint → MERT (≥ τMERT) → Cover (≥ τCover). Không đặt ngưỡng riêng cho sổ. `evidence.unknown_registry` có thêm `match_method`, `threshold`, `oti`/`oti_text`, `features_learned`, `features_total`, `noise_like_segments`, `first_filename` (ẩn khi `EVIDENCE_DETAIL=public`).
  - **Chỉ học phần mới** (`novel_features`): bỏ đoạn đã khớp τMERT (Cover: τCover) với đặc trưng mục đã có. Gửi lại đúng file đó không thêm gì.
  - **Chốt tiếng ồn** (`noise_prototypes`): đo thật, hai đoạn ồn trắng khác nhau có cosine MERT 0,998, cao hơn bài và bản đổi nhịp của chính nó (0,983).
    - Đoạn gần mẫu ồn trắng / hồng / nâu tới mức ≥ τMERT thì không dùng để học hay so bằng MERT.
    - Chốt này không cần ngưỡng mới.
  - **Không đi vào Rule Engine**: nhận ra "bài đã gặp" không đổi `match.type`, `assessment`, `recommendation`. Sổ không có dữ liệu quyền.
  - Học hỏng thì `features_error = FEATURES_FAILED`, vẫn ghi lượt gặp bằng fingerprint. Tắt học: `UNKNOWN_REGISTRY_LEARN=false`. `COVER_ENABLED=false` thì sổ cũng không dùng Cover.
  - **Chưa có thí nghiệm hiệu chỉnh riêng cho sổ.** Tỉ lệ nhận nhầm trên nhạc thật chưa đo vì máy dev không có audio corpus. Mở rộng EXP-06 cho sổ khi có dữ liệu.
- Lỗi SQLAlchemy ở các route của sổ → 503 `DATABASE_FAILURE`; `unknown_id` sai dạng hoặc không có → 404 `UNKNOWN_TRACK`; `EVIDENCE_DETAIL=public` bỏ `first_filename`.

### Vá cứng bảo mật (Security Hardening, `backend/security.py`)
Chi tiết đầy đủ ở [docs/SECURITY.md](file:///d:/PycharmProjects/AMR_advanced/docs/SECURITY.md); tóm tắt phần ảnh hưởng tới API:
- **Xác thực API Key tuỳ chọn**: mọi route dưới `/api/v1/*` (`APIRouter(dependencies=[Depends(verify_api_key)])`) yêu cầu header `X-API-Key` nếu biến môi trường `API_KEY` được đặt; so khớp bằng `secrets.compare_digest`. Không đặt `API_KEY` thì giữ nguyên chế độ mở. Sai/thiếu key → 401 `UNAUTHORIZED`.
- **Rate limit theo IP + giới hạn job đồng thời**: `POST /api/v1/analyze`, `/search` và `PATCH /unknown-tracks/{id}` dùng chung hạn mức `RATE_LIMIT_PER_MIN` request/phút/IP (`security.enforce_rate_limit`); `/analyze` còn giới hạn tối đa `MAX_CONCURRENT_JOBS` job chạy song song. Vượt quá → 429 `RATE_LIMITED` kèm header `Retry-After`.
  - **IP** (`security.get_client_ip`, sửa 2026-09-27): mặc định lấy IP kết nối, bỏ qua `X-Forwarded-For`. Trước đó server tin phần tử đầu của header này, do client tự gửi, nên đổi header là lách được rate limit.
  - Đứng sau N reverse proxy thì đặt `TRUSTED_PROXY_HOPS=N` (lấy phần tử thứ N tính từ phải) và chạy uvicorn với `--no-proxy-headers`.
- **Bản chạy công khai trên Hugging Face Spaces (2026-09-27, chủ dự án chọn công khai)**: GitHub Actions (`.github/workflows/deploy-space.yml`) chạy pytest, đạt thì `deploy/build_space_bundle.py` dựng gói và `deploy/push_space.py` đẩy lên Space. CSDL ở Neon (`deploy/setup_neon_db.py`, role `music_ai_app`); chỉ mục FAISS tải từ dataset repo lúc khởi động (`deploy/fetch_runtime_data.py`). Space đặt `EVIDENCE_DETAIL=public`, `TRUSTED_PROXY_HOPS=1`, `DEVICE=cpu`. `API_KEY` không dùng được cùng giao diện web (frontend không gửi header). Chi tiết: `docs/SECURITY.md` §11.
- **Chặn sớm upload quá khổ/sai định dạng**: kiểm `Content-Length` và đuôi file trước khi ghi đĩa; `save_upload` đếm byte theo khối 64 KB để huỷ giữa chừng nếu vượt `MAX_UPLOAD_MB`, không để lại file rác.
- **`sanitize_filename`**: lọc ký tự điều khiển và cắt độ dài trước khi tên file người dùng đi vào log/CSDL (chống Log/CRLF Injection).
- **`EVIDENCE_DETAIL=public`**: ẩn khối `thresholds` ở `/health` và làm tròn điểm số/ẩn ngưỡng trong evidence trả về từ `/results`, `/search`.

### Danh mục mã lỗi bắt buộc:
- **`FILE_TOO_LARGE`**: Kích thước file vượt quá giới hạn cấu hình (ví dụ: > 100MB).
- **`UNSUPPORTED_FORMAT`**: Định dạng file không nằm trong danh sách hỗ trợ.
- **`UNAUTHORIZED`**: Thiếu hoặc sai `X-API-Key` khi `API_KEY` đã được cấu hình (HTTP 401).
- **`RATE_LIMITED`**: Vượt giới hạn request/phút theo IP hoặc vượt số job phân tích đồng thời tối đa (HTTP 429, kèm header `Retry-After`).
- **`NO_AUDIO`**: File video tải lên không chứa luồng âm thanh (audio stream).
- **`NO_MUSIC`**: Đoạn âm thanh tải lên hoàn toàn là khoảng lặng (silence) hoặc không phát hiện được tín hiệu âm nhạc.
- **`MODEL_FAILURE`**: Lỗi trong quá trình suy luận của Chromaprint hoặc model MERT.
- **`DATABASE_FAILURE`**: Lỗi kết nối hoặc truy vấn cơ sở dữ liệu PostgreSQL / Qdrant.
- **`TIMEOUT`**: Thời gian xử lý tác vụ vượt quá ngưỡng timeout quy định.
- **`UNKNOWN_TRACK`**: Không tìm thấy bản ghi tương đồng nào trong cơ sở dữ liệu.
- **`LOW_CONFIDENCE`**: Độ tương đồng hoặc chất lượng tín hiệu quá thấp để đưa ra quyết định an toàn.
- **`INTERNAL_ERROR`**: Lỗi không lường trước (HTTP 500). Thông báo chung, chi tiết chỉ nằm trong log máy chủ.
- **`INVALID_REQUEST`**: Tham số yêu cầu sai kiểu/giá trị (HTTP 422), vd. `platform=youtube` thay vì `YOUTUBE`.

---

## 4. Quy chuẩn Logging mỗi Request / Job

Mọi tác vụ phân tích phải ghi log có cấu trúc (JSON structured log) để phục vụ kiểm toán và đo lường độ trễ:

- **Các trường log bắt buộc**:
  - `job_id`: UUID của tác vụ.
  - `timestamp`: Thời gian ghi nhận.
  - `file_duration_sec`: Thời lượng của file âm thanh đầu vào.
  - `latency_breakdown_ms`:
    - `audio_extraction`: Thời gian giải nén / tiền xử lý.
    - `fingerprint`: Thời gian tính fpcalc và query hash.
    - `embedding_inference`: Thời gian trích xuất vector MERT.
    - `vector_search`: Thời gian tìm kiếm FAISS / Qdrant.
    - `database_lookup`: Thời gian query metadata và rights.
    - `rule_engine`: Thời gian chạy cây quyết định.
    - `total_latency`: Tổng thời gian thực hiện.
  - `result`: Tóm tắt kết quả phân loại và `risk_level`.
  - `confidence`: Điểm tin cậy.
  - `error`: Chi tiết mã lỗi (nếu có).
  - `model_version`: Phiên bản cụ thể của pipeline.

> [!IMPORTANT]
> **Chính sách lưu trữ file người dùng**: Không lưu trữ vĩnh viễn file tải lên của người dùng nếu không có sự đồng ý. File tạm phải được xóa tự động sau khi tác vụ phân tích hoàn tất hoặc hết hạn phiên làm việc.
