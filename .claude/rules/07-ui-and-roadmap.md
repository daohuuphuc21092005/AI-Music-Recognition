---
description: Hướng dẫn thiết kế giao diện (4 màn hình MVP), lộ trình 12 tuần phát triển và phân công vai trò
globs:
  - "frontend/**"
  - "docs/**"
---

# 07. Giao diện & Lộ trình Phát triển (UI & Roadmap Standards)

Tài liệu này định hướng xây dựng trải nghiệm người dùng (UI/UX) trực quan, có thể giải thích được, và quy định lộ trình 12 tuần để quản lý phạm vi đồ án.

---

## 1. Thiết kế Giao diện (4 Màn hình MVP)

Hệ thống hướng tới trải nghiệm minh bạch, trực quan và chuyên nghiệp với 4 màn hình cốt lõi:

### Màn hình 1: Upload & Cấu hình Phân tích
- Khu vực kéo thả file (Drag & Drop) hỗ trợ cả audio (`.mp3`, `.wav`, `.m4a`) và video (`.mp4`, `.mov`).
- Các tùy chọn ngữ cảnh sử dụng:
  - **Nền tảng đích**: YouTube / TikTok / Facebook / Toàn bộ (`ALL`).
  - **Mục đích sử dụng**: Cá nhân / Phi thương mại vs Thương mại (`commercial_use`).
  - **Kế hoạch kiếm tiền**: Có bật kiếm tiền không (`monetization`).
- Nút bấm hành động chính: **`ANALYZE AUDIO`**.

### Màn hình 2: Processing (Tiến trình xử lý thời gian thực)
- Hiển thị thanh tiến trình gắn liền với các bước thực tế của pipeline:
  1. `Extracting & Pre-processing Audio`
  2. `Generating Chromaprint Fingerprint`
  3. `Searching Exact Match Database`
  4. `Computing MERT Deep Embedding` *(nếu không khớp fingerprint)*
  5. `Retrieving Similar Tracks (FAISS / Qdrant)`
  6. `Looking up Rights & Metadata`
  7. `Evaluating Decision Rules`
- **Yêu cầu quan trọng**: Tuyệt đối không hiển thị thông báo mơ hồ như *"AI is thinking..."* mà phải hiển thị rõ bước pipeline đang thực thi.

### Màn hình 3: Result (Bảng kết quả phân loại)
Trình bày rõ ràng thành 4 khối thông tin:
1. **Identification**: Tên bài hát, nghệ sĩ, album, thời lượng, đoạn khớp thời gian (timestamp).
2. **Rights Status**: Loại giấy phép (`CC_BY`, `Creator Music`, `Public Domain`, v.v.), nguồn gốc bản quyền, tổ chức sở hữu.
3. **Assessment Badge**:
   - 🟢 **`LOW RISK`**: An toàn sử dụng, được phép kiếm tiền.
   - 🟡 **`CONDITIONAL RISK`**: Được phép sử dụng nhưng phải tuân thủ điều kiện (ghi công tác giả, chia sẻ doanh thu).
   - 🔴 **`HIGH RISK`**: Rủi ro cao, nguy cơ bị chặn video, tắt tiếng hoặc nhận cảnh cáo bản quyền.
   - ⚪ **`UNKNOWN`**: Chưa đủ căn cứ xác minh, khuyến nghị thẩm định thủ công.
4. **Recommendation**: Lời khuyên hành động cụ thể cho nhà sáng tạo nội dung.

### Màn hình 4: Evidence & Explainability (Minh chứng chi tiết)
Dành cho người dùng chuyên sâu hoặc kiểm toán:
- Điểm tương đồng Chromaprint / MERT Cosine similarity score.
- Danh sách Top-K ứng viên tương đồng nhất kèm audio player đối soát.
- Nguồn dữ liệu pháp lý và ngày xác minh metadata (`verified_at`).
- Danh sách các luật quyết định đã được kích hoạt (`rules_triggered`).

### Chuẩn điều hướng & truy cập của 4 màn Phân tích (rà UI/UX 2026-09-26)
- **URL theo màn, không chứa `job_id`**: `#phan-tich` (Tải lên / Xử lý), `#phan-tich/ket-qua`, `#phan-tich/bang-chung` (`goAnalyze` + `route` trong `frontend/app.js`, `history.pushState`). Back/Forward đi đúng giữa Tải lên ↔ Kết quả ↔ Bằng chứng; tải lại trang thì kết quả không còn trong bộ nhớ → về Tải lên. `job_id` là khoá truy cập (docs/SECURITY.md) nên KHÔNG đưa vào URL.
- **Chuyển màn thì chuyển focus** vào `h2` của màn mới (`tabindex="-1"`). Màn Tải lên và Kết quả có `h2.sr-only`; heading của Kết quả mang luôn mức rủi ro. Tải trang / bấm tab từ Thống kê thì KHÔNG kéo focus.
- **Thanh bước là nút thật** (`button.step`), không phải `<li>` chỉ bấm được bằng chuột. Bước chưa tới được thì `disabled`; bước 1 chỉ quay về Tải lên, **không xoá kết quả** (trước đây bấm nhầm là mất). "Phân tích file khác" chỉ bỏ file đang chọn — Back vẫn xem lại được kết quả cũ.
- Màn Xử lý có một vùng `role=status` báo "Bước k/9: …" (chỉ ghi khi bước đổi); `<li>` đang chạy mang `aria-current="step"`; ký hiệu ○◐● có chữ thay thế (`content: "●" / "Xong: "`).
- Vùng thả file không là `role=button` (bên trong đã có nút "chọn file từ máy" — nút lồng trong nút). Lỗi tải lên `role=alert`, thông báo phản hồi `role=status`, nút gửi đổi thành "ĐANG TẢI LÊN…" + `aria-busy`, nút phản hồi khoá trong lúc gửi.
- Trạng thái hệ thống là `<details>` (Esc / bấm ra ngoài để đóng) — không để thông tin chỉ nằm trong `title`. Có link "Bỏ qua tới nội dung chính".
- Kiểm bằng `flow_check.mjs` (CDP, chọn file qua hộp chọn file thật): 37/37; tab Thống kê 25/25.

### Giao diện "Aurora" + Trang chủ + Lịch sử (2026-09-27)
- Chủ dự án yêu cầu phong cách "âm nhạc hiện đại, tươi sáng" và chốt các lựa chọn sau:
  - **Đổi giao diện frontend hiện có**: giữ HTML/JS thuần, nối API thật. KHÔNG làm app React riêng; thay Framer Motion bằng CSS + Web Animations.
  - **Giữ thang 4 mức rủi ro**.
  - Thể loại: **có giao diện, ghi rõ là ví dụ**. Dán link: ban đầu có thẻ "chưa hỗ trợ", **đã bỏ hẳn ngày 2026-09-27** vì chủ dự án chưa định phát triển.
- **4 tab trong thanh trên**: Trang chủ `#trang-chu` (mặc định khi không có hash) · Phân tích `#phan-tich…` · Lịch sử `#lich-su` · Thống kê & Tra cứu `#thong-ke…`.
  - Bấm tab thì focus ở lại tab. Back/Forward, CTA và link trong trang thì focus vào `h2` của màn mới (`route()` + cờ `state.viaTab`).
  - Job xong khi đang ở tab khác thì KHÔNG giật người dùng về (`showAnalyzeScreen`).
- **Trang chủ**: hero aurora, CTA "Kiểm tra bản quyền ngay", 4 thẻ tính năng, 3 bước. Số bản ghi lấy THẬT từ `/health`; không bịa số liệu xã hội.
- **Lịch sử**: `localStorage` (`mra.history.v1`, tối đa 50), chỉ job DO TRÌNH DUYỆT NÀY chạy. KHÔNG có API liệt kê job (job_id là khoá truy cập, docs/SECURITY.md §10).
  - Lọc theo mức rủi ro, mở lại qua `GET /results/{job_id}`. Job không còn thì báo lỗi ngay tại mục, không đổi màn.
- **Màn Tải lên chỉ nhận file** (không còn tablist nguồn). Hộp `#server-note` nói trước khi tải lên nếu máy chủ hạn chế:
  - Mất CSDL / Rule Engine → sọc đỏ, khoá nút PHÂN TÍCH.
  - Thiếu chỉ mục MERT / FFmpeg → sọc vàng, chỉ nhắc.
  - Trang tự hỏi lại `/health` mỗi 10 s và tự mở khoá; hỏi lại không nạp lại file đã chọn (`refreshUploadGate`, tách khỏi `pickFile`).
  - Thiếu FFmpeg thì chặn cả `.m4a`/`.aac`, không chỉ video.
- **Thể loại**: thẻ `.card-sample` có viền nét đứt, huy hiệu "Ví dụ minh hoạ" và thanh gạch chéo. Dữ liệu cố định, giống nhau cho mọi file. Hệ thống CHƯA phân loại thể loại.
- **Hiệu ứng âm nhạc gắn với dữ liệu thật, không bịa tiến trình**:
  - Sóng âm của file = biên độ thật, giải mã trong trình duyệt, chỉ file audio ≤ 15 MB.
  - Thanh tiến trình sóng âm tô theo bước pipeline backend báo (`role=progressbar`).
  - Chỉ số đếm = `match.confidence`. Chưa định danh thì ghi "điểm cao nhất — dưới ngưỡng".
  - Đĩa than và equalizer chỉ để trang trí (`aria-hidden`).
- **Font Be Vietnam Pro tự lưu** trong `frontend/fonts/` (OFL). Righteous/Poppins mà skill gợi ý không có bộ ký tự tiếng Việt. `backend/main.py` đăng ký MIME `font/woff2`.
- Mọi animation dừng khi `prefers-reduced-motion: reduce`.
- Kiểm bằng `v2_check.mjs` 34/34 (gồm máy chủ mất CSDL, thiếu FFmpeg với m4a), `flow_check.mjs` 37/37, `uiux_check.mjs` 25/25 (CDP, Chrome headless). Có ảnh sáng / tối / 390px.

### Tab "Thống kê & Tra cứu" (sổ bài chưa nhận diện — đã triển khai 2026-09-26)
- Là một trong 4 tab của thanh trên (`#screen-registry`); thanh 4 bước chỉ hiện ở luồng Phân tích.
- Nội dung: 6 ô số liệu · thanh ngang theo loại / mức rủi ro gần nhất (chấm màu + mã chữ, không đọc bằng màu) / nền tảng · cột lượt gặp 30 ngày (tooltip hover + Tab, bảng "Xem dạng bảng") · 5 mục gặp lại nhiều nhất · tra cứu (từ khoá, loại, trạng thái, sắp xếp, phân trang 20) · khung chi tiết + form ghi chú thẩm định.
- Màn Kết quả có thẻ `#card-registry` ("lượt gặp thứ N, lần đầu …", nút "Xem trong sổ"); màn Processing có bước `REGISTRY_LOOKUP`.
- **Sổ "học" (2026-09-27)**:
  - Thẻ đổi tiêu đề theo trường hợp:
    - **"Nhận ra bài đã gặp"**: nêu tên do người thẩm định đặt, hoặc tên file lần đầu; tầng nhận ra (vân tay / MERT / Cover); điểm so với ngưỡng; OTI.
    - **"Bài mới — đã ghi vào sổ"**.
  - Dòng `#registry-learned` nói đã ghi nhớ bao nhiêu đặc trưng, hoặc vì sao không học (trùng nội dung, giống tiếng ồn, lỗi trích xuất).
  - Chi tiết mục trong tab Thống kê có dòng "Đặc trưng đã học", và bảng lượt gặp có cột "Nhận ra qua".
  - Câu nhắc cố định: nhận ra bài đã gặp **không phải dữ liệu quyền**, mức rủi ro không đổi.
- Mọi câu chữ nhắc: ghi chú trong sổ **không phải dữ liệu quyền**, không đổi mức rủi ro.
- **URL phản ánh trạng thái**: tab là liên kết `#phan-tich` / `#thong-ke`; mở một mục thì URL thành `#thong-ke/<unknown_id>` (replaceState, không thêm lịch sử). Back/Forward chuyển đúng tab, link tới một mục gửi được cho người khác (`route()` trong `frontend/app.js`).
- **Chuẩn truy cập đã kiểm (rà UI/UX 2026-09-26, Chrome headless, 25/25 kiểm tra)**: ô đầu mỗi hàng là `<button>` thật (không dùng `tr tabindex`); mở chi tiết chuyển focus vào tiêu đề, Đóng trả focus về nút đã mở; mục tiêu bấm ≥ 24 px (WCAG 2.5.8); đang tải thì `aria-busy` + làm mờ nội dung cũ, nút Lưu khoá trong lúc lưu; số kết quả báo qua `role=status`, lỗi qua `role=alert`; mức rủi ro dùng chấm màu theo token + mã chữ (không emoji); trạng thái rỗng có câu hướng dẫn + hành động; cuộn tôn trọng `prefers-reduced-motion`; chữ nhỏ nhất 12 px.

### Design system (token 3 lớp — 2026-09-26)
- Token ở đầu `frontend/styles.css`, gồm 3 lớp: **nguyên thuỷ** (`--gray-*`, `--lilac-*`, `--violet-*`, `--pink-*`, `--orange-*`…, `--space-*`, `--font-size-*`, `--radius-*`) → **ngữ nghĩa** (`--color-*`, `--gradient-*`) → **thành phần** (`--button-*`, `--input-*`, `--card-*`, `--tab-*`, `--chart-*`…). Đặc tả đầy đủ: `docs/DESIGN_SYSTEM.md`.
- Bắt buộc:
  - Rule **không viết mã màu** và **không gọi thẳng dải màu**.
  - Chế độ tối **chỉ ghi đè lớp ngữ nghĩa**.
  - Biến thể (mức rủi ro…) = đặt lại token cục bộ, không viết lại rule.
  - Không đưa `<style>` vào `index.html`.
- Màu rủi ro có 2 bản:
  - `--color-{mức}` để **tô**, cần ≥ 3:1.
  - `--color-{mức}-strong` để làm **chữ** hoặc làm nền dưới chữ trắng, cần ≥ 4.5:1.
  - Chữ màu nhấn dùng `--color-accent-strong`, không dùng `--color-accent`.
  - Thêm màu thì phải đo tương phản cả sáng lẫn tối rồi ghi vào bảng §2 của tài liệu.
- Kích thước bấm và ô nhập:
  - Nút / ô nhập / select cao `--control-height` (44px); nút nhỏ 36px.
  - Một vòng focus chung (2px `--color-focus-ring`, cách phần tử 2px, `:focus-visible`).
  - Ô nhập dùng chữ 16px ở ≤ 720px (chống Safari iOS tự phóng to).

### Trang quản trị `/admin` (Tùy chọn mở rộng)
- Tra cứu cơ sở dữ liệu `recordings`, `compositions`, `rights`.
- Công cụ kích hoạt tính lại fingerprint / re-index vector database.
- Bảng điều khiển xem log lỗi và xuất báo cáo kết quả thí nghiệm (CSV/JSON).

---

## 2. Lộ trình 12 Tuần & Các Milestone Chính

Để tránh tình trạng làm dàn trải hoặc mở rộng phạm vi không kiểm soát, mọi công việc đều phải đối chiếu với mốc Milestone hiện tại:

```
[M1: Data Ready] → [M2: Exact Matching] → [M3: Deep Retrieval] → [M4: Hybrid AI Ready]
    (Tuần 3)            (Tuần 4)                (Tuần 6)              (Tuần 7 - KEY)
       ↓
[M5: Rights Ready] → [M6: Full App] → [M7: Evaluation] → [M8: Deployment]
    (Tuần 9)             (Tuần 10)          (Tuần 11)          (Tuần 12)
```

- **Tuần 1 – 3: `M1 — Data Ready`**
  - Thu thập và chuẩn hóa metadata các tập dữ liệu MVP (Jamendo, FMA, YouTube Audio Library, Public Domain).
  - Xây dựng database schema ban đầu trên PostgreSQL.
- **Tuần 4: `M2 — Exact Matching Ready`**
  - Triển khai Chromaprint (`fpcalc`), xác lập ngưỡng tối ưu `τFP`. Hoàn thành `EXP-01`.
- **Tuần 5 – 6: `M3 — Deep Retrieval Ready`**
  - Tích hợp model `MERT-v1-95M`, thực nghiệm pooling Mean vs Mean+Std (`EXP-03`).
  - Xây dựng index tìm kiếm trên FAISS. Hoàn thành `EXP-02`.
- **Tuần 7: `M4 — Hybrid AI Ready (Milestone quan trọng nhất)`**
  - Ghép tầng Cascade: `Fingerprint → MERT`.
  - Hoàn thành `EXP-04` chứng minh tính ưu việt của mô hình lai.
- **Tuần 8 – 9: `M5 — Rights Intelligence Ready`**
  - Triển khai `RightsService` và `DecisionService` (Rule Engine tuần tự 5 nhóm).
  - Viết bộ 50–100 unit tests cho Rule Engine.
- **Tuần 10: `M6 — Full Application Ready`**
  - Hoàn thiện REST API trên FastAPI, tích hợp giao diện người dùng (Frontend).
  - Kết nối hoàn chỉnh luồng từ Upload → Xử lý → Trả kết quả + Evidence.
- **Tuần 11: `M7 — Final Evaluation`**
  - Chạy toàn bộ bộ test trên Robustness Test Set (`EXP-05`, `EXP-06`, `EXP-08`).
  - Lập báo cáo số liệu so sánh với ngưỡng nghiệm thu nội bộ.
- **Tuần 12: `M8 — Deployment & Handover`**
  - Đóng gói ứng dụng với Docker Compose, hoàn thiện tài liệu kỹ thuật và báo cáo đồ án.

> [!TIP]
> Khi ước lượng công việc hoặc đặt câu hỏi *"còn thiếu gì"*, luôn đối chiếu trực tiếp với Milestone hiện tại thay vì phát triển trước các tính năng của tuần tiếp theo.

---

## 3. Phân công Vai trò Gợi ý (Nhóm 4 thành viên)

1. **Data & Rights Specialist**: Phụ trách thu thập, làm sạch dataset, quản trị database PostgreSQL, thiết kế schema `rights`/`compositions` và bộ unit test cho Rule Engine.
2. **AI / MIR Engineer**: Phụ trách tích hợp Chromaprint, MERT-v1-95M, chiến lược pooling, vector search (FAISS/Qdrant), và các thí nghiệm âm học.
3. **Backend & MLOps Engineer**: Phụ trách kiến trúc FastAPI, pipeline bất đồng bộ, xử lý hàng đợi, quản lý mã lỗi, structured logging, và đóng gói Docker.
4. **Frontend & Evaluation Lead**: Phụ trách giao diện 4 màn hình, tích hợp API, xây dựng Robustness Test Set, thực thi các bài đo lường hiệu năng và trực quan hóa số liệu.
