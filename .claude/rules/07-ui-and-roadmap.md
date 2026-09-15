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
