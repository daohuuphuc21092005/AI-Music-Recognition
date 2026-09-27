# Chính sách & Kiến trúc An ninh (Security Architecture & Hardening)

Tài liệu này mô tả các biện pháp bảo vệ, mô hình kiểm soát truy cập và các đánh đổi an ninh trong hệ thống **Music Rights AI**.

---

## 1. Mô hình Kiểm soát Truy cập & UUID Job ID

### 1.1 Cơ chế Capability URL
Trong phiên bản hiện tại, hệ thống hoạt động theo mô hình phi tập trung không trạng thái người dùng (stateless/tokenless):
- Khi gửi yêu cầu phân tích qua `POST /api/v1/analyze`, hệ thống sinh một **UUID v4** ngẫu nhiên làm `job_id`.
- `job_id` đóng vai trò là một **Capability Key** (chìa khoá truy cập): client nắm giữ `job_id` có quyền kiểm tra trạng thái (`GET /api/v1/jobs/{job_id}`) và lấy kết quả phân tích (`GET /api/v1/results/{job_id}`).

### 1.2 Phân tích Rủi ro Chấp nhận (Accepted Risk)
- **Không gian địa chỉ UUID v4**: Với 122 bit entropy thực tế ($5.3 \times 10^{36}$ khả năng), việc dò quét ngẫu nhiên (brute-force) để đoán trúng một `job_id` đang hoạt động là **bất khả thi về mặt tính toán**.
- **Rủi ro chấp nhận**: Bất kỳ ai có được đường dẫn chứa `job_id` (qua nhật ký proxy, chia sẻ màn hình, rò rỉ URL) đều có thể xem kết quả phân tích của bản nhạc đó.
- **Lộ trình nâng cấp**: Khi hệ thống triển khai môi trường doanh nghiệp có nhiều người dùng (multi-tenant), sẽ tích hợp tầng xác thực danh tính (JWT / OAuth2 / API Key per User) và gán quyền sở hữu `owner_user_id` vào từng dòng trong bảng `jobs`.

---

## 2. Xác thực API Key (Tùy chọn)

Để bảo vệ các endpoint backend khỏi việc bị client lạ hoặc bot bên ngoài lạm dụng, hệ thống hỗ trợ cơ chế xác thực tập trung qua biến môi trường `API_KEY`:

### Cách bật:
1. Thêm biến `API_KEY` vào file `.env`:
   ```bash
   API_KEY=your-secure-random-secret-key-here
   ```
2. Khởi động lại backend. Khi `API_KEY` được thiết lập, mọi endpoint thuộc `/api/v1/*` bắt buộc phải có header:
   ```http
   X-API-Key: your-secure-random-secret-key-here
   ```
3. Nếu thiếu header hoặc key sai, API lập tức trả về HTTP 401 `UNAUTHORIZED`.
4. Cơ chế kiểm tra sử dụng so sánh an toàn thời gian cố định (`secrets.compare_digest`) để triệt tiêu nguy cơ tấn công dò thời gian (Timing Attack).
5. Nếu không đặt `API_KEY`, hệ thống giữ nguyên chế độ mở cho môi trường thử nghiệm và ghi cảnh báo rõ trong nhật ký khởi động máy chủ.

---

## 3. Bảo vệ Tầng Tải lên (Upload Protection) & Chống DoS

1. **Từ chối sớm theo Content-Length**: Khi client gửi header `Content-Length > MAX_UPLOAD_MB * 1024 * 1024`, máy chủ trả về ngay HTTP 413 `FILE_TOO_LARGE` trước khi nhận dữ liệu thân request.
2. **Kiểm tra định dạng trước khi ghi đĩa**: Đuôi file được đối chiếu với `config.ALLOWED_EXTENSIONS`. Đuôi không hợp lệ bị từ chối ngay bằng HTTP 415 `UNSUPPORTED_FORMAT`. File lưu tạm trên đĩa chỉ sử dụng đuôi chuẩn hoá từ danh sách cho phép kết hợp UUID v4 làm tên file.
3. **Đếm byte khi ghi stream (Chunk Streaming Counter)**: Hàm `save_upload` đọc và ghi theo từng khối 64 KB, tích luỹ số byte thực tế. Nếu dữ liệu vượt quá `MAX_UPLOAD_MB`, quá trình ghi dừng lại ngay lập tức, xoá file tạm và ném lỗi 413.
4. **Rate Limiting theo IP**: Giới hạn tối đa `RATE_LIMIT_PER_MIN` (mặc định 10 request/phút) cho mỗi địa chỉ IP. Vượt quá sẽ nhận HTTP 429 `RATE_LIMITED` kèm header `Retry-After`. Áp cho `/analyze`, `/search` và `PATCH /unknown-tracks/{id}` (chung một hạn mức).
   - **IP lấy ở đâu** (`security.get_client_ip`, sửa 2026-09-27): mặc định là IP của kết nối TCP; header `X-Forwarded-For` bị bỏ qua.
   - Trước đây server lấy phần tử ĐẦU của `X-Forwarded-For`. Phần tử đó do client tự gửi, nên đổi header ở mỗi request là không bao giờ bị 429.
   - Đứng sau N reverse proxy thì đặt `TRUSTED_PROXY_HOPS=N`. Khi đó IP thật là phần tử thứ N tính từ PHẢI, vì mỗi proxy chỉ nối IP nó thấy vào cuối header.
   - Đặt N lớn hơn số proxy thật thì client lại giả được IP. Đặt nhỏ hơn thì mọi người dùng chung một hạn mức (an toàn nhưng chật).
   - uvicorn phải chạy với `--no-proxy-headers`, để nó không tự thay IP kết nối bằng giá trị trong header (`deploy/space/start.sh`).
5. **Giới hạn tác vụ đồng thời (Concurrency Limiter)**: Sử dụng Semaphore non-blocking giới hạn tối đa `MAX_CONCURRENT_JOBS` (mặc định 2). Nếu server đang chạy hết công suất, request mới nhận HTTP 429 `RATE_LIMITED` kèm header `Retry-After: 30`.
6. **Sanitize Filename**: Mọi tên file đầu vào đều được lọc bỏ ký tự điều khiển (`\r`, `\n`, control characters) và cắt tối đa 255 ký tự trước khi đưa vào logger hoặc CSDL, loại trừ nguy cơ Log Injection và CRLF Injection.

---

## 4. Kiểm soát Thực thi FFmpeg

Khi tách âm thanh từ file video, `audio_service` gọi FFmpeg với các chốt an toàn:
- **Giới hạn thời lượng**: `-t MAX_MEDIA_SECONDS` (mặc định 900 giây / 15 phút) ngăn việc xử lý video dài bất thường làm treo worker.
- **Vô hiệu hoá tương tác stdin**: Cờ `-nostdin` ngăn FFmpeg chờ nhập liệu từ terminal.
- **Timeout cứng**: Giới hạn `FFMPEG_TIMEOUT_S` (mặc định 120 giây). Nếu quá thời gian, tiến trình FFmpeg bị cưỡng chế dừng (`TimeoutExpired`), file tạm được dọn dẹp sạch sẽ và API ném mã lỗi chuẩn hoá `TIMEOUT`.

---

## 5. Toàn vẹn Mô hình Học máy (Model Integrity)

- File trọng số mô hình `models/license/*.joblib` được bảo vệ bằng file sidecar chứa mã băm SHA-256 tương ứng (`.sha256`).
- Trước khi gọi `joblib.load()`, hệ thống bắt buộc tính toán lại SHA-256 và đối chiếu với sidecar.
- Nếu file sidecar bị thiếu, hoặc mã băm không khớp (nghi ngờ file bị sửa đổi / giả mạo), hệ thống từ chối nạp, ghi cảnh báo vào log và tự động chuyển về đường xử lý an toàn "không có model" (`predict() -> None`).

---

## 6. Phân quyền CSDL Tối thiểu (Least Privilege)

Mặc định ở giai đoạn phát triển, ứng dụng kết nối qua tài khoản `postgres`. Để an toàn cho môi trường sản xuất:
1. Áp dụng file script [create_app_role.sql](file:///d:/PycharmProjects/AMR_advanced/scripts/sql/create_app_role.sql) bằng lệnh:
   ```bash
   psql -U postgres -d music_rights_ai -f scripts/sql/create_app_role.sql
   ```
2. Cập nhật `DATABASE_URL` trong file `.env` sử dụng role `music_ai_app`:
   ```bash
   DATABASE_URL=postgresql://music_ai_app:<MAT_KHAU_APP_USER>@localhost:5432/music_rights_ai
   ```
3. Role `music_ai_app` chỉ có quyền:
   - `SELECT, INSERT, UPDATE` trên `jobs`, `analysis_results`, `feedback`, `unknown_tracks`, `unknown_sightings`; `SELECT, INSERT` trên `unknown_track_features`.
   - `SELECT` (Read-only) trên `recordings`, `compositions`, `rights`, `fingerprints`, `embeddings`.
   - Bị thu hồi mọi quyền `DROP`, `TRUNCATE`, `ALTER`.

---

## 7. Ẩn thông tin nhạy cảm (Evidence Detail)

- Cấu hình `EVIDENCE_DETAIL=full|public` (mặc định `full`).
- Ở chế độ `public`:
  - Endpoint `/health` không trả về khối `thresholds` (ngưỡng nhạy cảm của hệ thống).
  - Kết quả trả về qua `/results/{job_id}` và `/search` được ẩn các trường ngưỡng trong `evidence`, và các điểm số tin cậy được làm tròn 2 chữ số thập phân, ngăn chặn đối thủ đảo ngược (reverse-engineer) các ngưỡng phân loại.
  - Sổ bài chưa nhận diện (`/unknown-tracks*`) không trả `first_filename`, không cho tìm theo tên file, và làm tròn điểm về 2 chữ số.

---

## 8. Đánh đổi & Giới hạn của Quét Đa Cửa Sổ (Multi-window Scanning)

- **Tăng tỷ lệ nhận nhầm (False Match Rate - FMR)**: Việc quét $K$ cửa sổ độc lập trên một file làm tăng số cơ hội đưa ra kết luận so khớp sai lên tối đa xấp xỉ $K$ lần so với việc chỉ quét 1 cửa sổ 30 giây đầu. Do đó:
  - Ngưỡng nhận diện của các tầng (`τFP=0.30`, `τMERT=0.98`, `τCover=0.90`) được giữ nghiêm ngặt.
  - Áp dụng cơ chế dừng sớm khi gặp `EXACT_MATCH` để giảm thiểu số lần quét không cần thiết.
  - Ngân sách thời gian `SCAN_TIME_BUDGET_S` khống chế tổng thời gian quét của một file.
- **Khả năng chống né tránh kết hợp (Composite Evasion)**: Các biến thể kết hợp (chèn khoảng lặng đầu file, dịch cao độ kèm co giãn thời gian và nhiễu) được kiểm thử thông qua thí nghiệm mở rộng EXP-10 (`experiments/exp10_multiwindow/run.py`). Kết quả định lượng chi tiết cần được tham khảo từ báo cáo EXP-10 sau khi chạy trên corpus hoàn chỉnh.

---

## 9. Sổ bài chưa nhận diện (`/api/v1/unknown-tracks*`)

Mọi kết quả UNKNOWN của `/analyze` được ghi vào bảng `unknown_tracks` / `unknown_sightings` để thống kê và tra cứu về sau (`backend/services/unknown_registry_service.py`).

- **Lưu gì**: fingerprint Chromaprint (chuỗi hash, không dựng lại được âm thanh), thời lượng, tên file đã qua `sanitize_filename` của lượt gặp ĐẦU TIÊN, nền tảng / mục đích / kiếm tiền người dùng khai báo, thời điểm, và ảnh chụp kết quả Rule Engine. **Không lưu file audio**: file tải lên vẫn bị xoá ngay sau khi phân tích như trước.
- **Không lộ `job_id`**: `unknown_sightings.job_id` chỉ để kiểm toán phía máy chủ; không endpoint nào của sổ trả trường này, nên sổ không làm hỏng mô hình Capability Key ở mục 1.
- **Rủi ro chấp nhận ở chế độ mở**: khi KHÔNG đặt `API_KEY`, ai truy cập được máy chủ cũng đọc được sổ và sửa được ghi chú thẩm định qua `PATCH`. `EVIDENCE_DETAIL=public` ẩn tên file do người khác tải lên.
  - `API_KEY` chỉ bảo vệ được API, **không dùng được cùng giao diện web**: `frontend/app.js` không gửi header `X-API-Key`, nên đặt khoá là mọi thao tác trên trang báo 401.
  - Bản chạy công khai trên Hugging Face Spaces chấp nhận rủi ro này (mục 11).
- **Ghi chú không phải dữ liệu quyền**: Rule Engine không đọc sổ; sửa ghi chú không thể biến một kết quả UNKNOWN thành đã định danh hay đổi mức rủi ro (`tests/test_unknown_registry.py`). Ký tự điều khiển trong ghi chú bị lọc; frontend chèn mọi chuỗi của sổ qua `escapeHtml`.
- **Chi phí mỗi bài lạ**: đối chiếu fingerprint với tối đa `UNKNOWN_REGISTRY_SCAN_LIMIT` (mặc định 500) mục gặp gần nhất, ~1,4 ms/mục — chặn trên thời gian thêm vào mỗi job.
- **Đặc trưng "đã học" (2026-09-27)**: với mục `NOT_IDENTIFIED`, bảng `unknown_track_features` lưu thêm vector MERT 768 chiều (trung bình theo thời gian của từng đoạn 15 s) và descriptor chroma 12 × 64 khung.
  - Cả hai đều mất pha và mất chi tiết thời gian, nên không dựng lại được âm thanh. Cam kết "không lưu file người dùng" vẫn giữ.
  - Mỗi bài lạ tốn thêm vài lượt MERT (tối đa `UNKNOWN_REGISTRY_MAX_SEGMENTS` đoạn, ~0,1–0,3 s/đoạn trên GPU), và phải nạp đặc trưng của tối đa 500 mục (~3 KB/vector).
  - **Rủi ro chấp nhận**: ai gửi được file thì "dạy" được sổ. Một bài bị gửi dưới tên file sai sẽ được nhận ra lại với đúng tên sai đó, cho tới khi người thẩm định sửa. Mức rủi ro vẫn không đổi (sổ không đi vào Rule Engine).
  - Muốn chặn việc này: tắt học bằng `UNKNOWN_REGISTRY_LEARN=false`, hoặc đặt `API_KEY` (chỉ dùng được khi không cần giao diện web, xem trên).

---

## 10. Lịch sử phân tích trong trình duyệt (tab "Lịch sử")

Tab Lịch sử của frontend liệt kê các lần phân tích **do chính trình duyệt đó chạy**. Không có API liệt kê job: liệt kê job của mọi người sẽ phá mô hình Capability Key ở mục 1.

- **Lưu gì, ở đâu**: `localStorage` khoá `mra.history.v1`, tối đa 50 mục. Mỗi mục gồm `job_id`, tên file, thời điểm, mức rủi ro, loại khớp, độ tin cậy, tên bài / nghệ sĩ nếu định danh được. Không gửi đi đâu; máy khác và trình duyệt khác không thấy.
- **Rủi ro chấp nhận**: `job_id` là khoá truy cập kết quả. Ai dùng chung trình duyệt / hồ sơ người dùng (hoặc có mã độc đọc được `localStorage` của trang) sẽ mở lại được các kết quả đó, giống hệt việc họ xem được lịch sử duyệt web. Cùng mức rủi ro với mục 1.2.
- **Xoá**: nút "Xoá lịch sử" (xoá hết) và nút ✕ trên từng mục; xoá dữ liệu trang web của trình duyệt cũng xoá luôn. Chế độ riêng tư / trình duyệt chặn lưu trữ thì tab Lịch sử trống, app vẫn chạy.
- **URL không chứa `job_id`**: các màn Phân tích dùng `#phan-tich/ket-qua`, `#phan-tich/bang-chung`, nên `job_id` không lọt vào lịch sử duyệt web, header Referer hay link người dùng sao chép.
- **Sóng âm của file** được giải mã ngay trong trình duyệt (`OfflineAudioContext`), không có lượt gửi thêm nào ngoài lần tải lên để phân tích.

---

## 11. Bản chạy công khai trên Hugging Face Spaces (2026-09-27)

Chủ dự án chọn đưa hệ thống lên web **công khai cho mọi người**: GitHub Actions chạy test rồi triển khai lên Hugging Face Spaces, CSDL đặt ở Neon (README, mục "Triển khai lên web").

**Cấu hình của Space** (`deploy/space/Dockerfile`):
- `EVIDENCE_DETAIL=public`: ẩn tên file người khác đã gửi trong sổ, ẩn ngưỡng nội bộ, làm tròn điểm.
- `TRUSTED_PROXY_HOPS=1`: Space đứng sau reverse proxy của Hugging Face (mục 3.4). **Chưa kiểm được số proxy thật trước khi triển khai.** Sau lần triển khai đầu, phải thử gửi request kèm `X-Forwarded-For` giả và xem có bị 429 đúng hạn không.
- Container chạy bằng uid 1000, không có `.env`. Mọi bí mật là Secret của Space, do `deploy/push_space.py` đặt từ GitHub Secrets và không bao giờ in ra log.

**CSDL trên Neon** (`deploy/setup_neon_db.py`):
- Server đăng nhập bằng role `music_ai_app`: chỉ `SELECT` trên dữ liệu tham chiếu, `SELECT/INSERT/UPDATE` trên job và sổ, không tạo/xoá/sửa được bảng. Kiểm trên PostgreSQL tạm: `CREATE TABLE` báo `permission denied for schema public`, `DELETE FROM recordings` báo `permission denied for table recordings`.
- Chỉ chép dữ liệu tham chiếu (compositions, recordings, rights, fingerprints). Job, phản hồi và sổ trên máy dev **không** được chép, nên tên file và lịch sử của máy dev không lên web.
- Mật khẩu role do script sinh ngẫu nhiên, chỉ ghi vào `.env.neon` (bị `.gitignore` chặn theo mẫu `.env.*`).

**Rủi ro chấp nhận** (chủ dự án chọn công khai):
- Ai có link cũng tải file lên được, trong hạn mức `RATE_LIMIT_PER_MIN` và `MAX_CONCURRENT_JOBS`.
- Ai cũng đọc được sổ bài chưa nhận diện (không có tên file), sửa được ghi chú, và "dạy" được sổ bằng file bất kỳ (mục 9). Mức rủi ro trả về không đổi vì sổ không đi vào Rule Engine.
- Trang ghi rõ ở màn Tải lên: file bị xoá sau khi phân tích; dấu vân tay và đặc trưng của bài chưa nhận diện được giữ lại; kết quả không phải tư vấn pháp lý.
- Chỉ mục FAISS nằm trong một dataset repo trên Hugging Face, mặc định private. Nó chỉ chứa vector MERT của bài FMA và mã `recording_id`, không chứa audio.
