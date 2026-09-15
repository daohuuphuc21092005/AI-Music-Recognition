---
description: Nguyên tắc bất biến cốt lõi, định nghĩa bài toán và phạm vi đồ án AI Phân loại Nhạc Bản quyền
globs: "*"
---

# 01. Nguyên tắc bất biến & Phạm vi dự án (Core Invariants & Scope)

Tài liệu này là quy tắc tối cao và bắt buộc tuân thủ trong toàn bộ quá trình phát triển dự án **Hệ thống AI Phân loại Nhạc Bản quyền** (AI-based Music Identification and Copyright Usage Risk Assessment System).

## 1. Định nghĩa bài toán

- **Input**: File audio hoặc video chứa âm thanh.
- **Output** cho mỗi đoạn nhạc:
  - `match_type`: Kiểu khớp (`EXACT_MATCH`, `NEAR_EXACT_MATCH`, `EMBEDDING_MATCH`, `COVER_MATCH`, `NO_MATCH`).
  - `confidence`: Độ tin cậy tổng hợp và phân rã.
  - `recording_id` / `composition_id`: Định danh bản ghi và bản quyền tác phẩm.
  - `license` / `rights_status`: Trạng thái pháp lý và loại giấy phép.
  - `risk_level`: Mức độ rủi ro (`LOW`, `CONDITIONAL`, `HIGH`, `UNKNOWN`).
  - `condition`: Điều kiện sử dụng cụ thể.
  - `evidence`: Dữ liệu minh chứng chi tiết.
  - `recommendation`: Khuyến nghị sử dụng thực tế.

### Thứ tự phân loại 5 nhóm bản quyền (Cố định, tuần tự)
Hệ thống phân loại nhạc vào **5 nhóm bản quyền** theo thứ tự ưu tiên **cố định, tuần tự** (không chạy song song, không random):
```
Audio Library → Creator Music → Content ID (Commercial) → Creative Commons → Public Domain
→ (nếu không khớp điều kiện nào) UNCATEGORIZED / USER-GENERATED CONTENT
```

---

## 2. 9 Nguyên tắc bất biến (Tuyệt đối không vi phạm khi code)

1. **Composition ID ≠ Recording ID**:
   - `Composition` = tác phẩm / giai điệu (quyền tác giả).
   - `Recording` = một bản thu cụ thể (quyền bản ghi / quyền liên quan).
   - Một `composition` có thể có nhiều `recording`.
   - **Tuyệt đối không bao giờ suy luận**: `composition public domain ⇒ recording public domain`.

2. **Thứ tự phân loại là cây quyết định tuần tự**:
   - Không phải multi-label song song.
   - Luôn kiểm tra lần lượt: `Audio Library` → `Creator Music` → `Content ID` → `Creative Commons` → `Public Domain` → Fallback (`UNCATEGORIZED` / `USER-GENERATED CONTENT`).

3. **Không kết luận từ kết quả nghi ngờ**:
   - Mọi quyết định phân loại phải đi kèm các chỉ số tin cậy được tách riêng:
     - `identity_confidence`: Độ tin cậy nhận diện âm thanh.
     - `rights_confidence`: Độ tin cậy của metadata bản quyền.
     - `decision_confidence`: Độ tin cậy tổng thể của quyết định.

4. **Không hộp đen (No Black-Box Output)**:
   - Mọi kết quả phân loại đều phải kèm theo:
     - Dữ liệu chứng minh cụ thể (`evidence`).
     - Cơ chế đảo ngược / giải thích được (`decision_rules_triggered`).

5. **`UNKNOWN` không được ép thành `LOW` hoặc `HIGH`**:
   - Bắt buộc trả về `UNKNOWN` khi:
     - Không nhận diện được hoặc similarity dưới ngưỡng tin cậy.
     - License không xác định hoặc metadata mâu thuẫn.
     - Xác định được composition nhưng không xác định được recording.
     - Dữ liệu quyền đã quá hạn hoặc thiếu kiểm chứng.

6. **Rule Engine là logic `if/else` thuần, KHÔNG phải Machine Learning**:
   - Tuyệt đối không dùng model để dự đoán `risk_level` hoặc `license_category`.
   - AI/ML chỉ dùng cho trích xuất đặc trưng nhận diện (`identity`) và độ tương đồng (`similarity`).
   - Quyết định pháp lý và mức độ rủi ro luôn phải qua Rule Engine tường minh, minh bạch.

7. **Không dùng CLAP (hay bất kỳ model nào) làm "Copyright Classifier"**:
   - Trạng thái pháp lý và giấy phép bản quyền không phải là đặc trưng âm học (acoustic class).
   - Tuyệt đối không suy đoán pháp lý từ zero-shot prompt (ví dụ: cấm dùng prompt kiểu *"this is copyrighted music"*).

8. **Data Split nghiêm ngặt theo `recording_id`, không theo segment**:
   - Với Cover Detection, split theo `composition_id`.
   - Các segment của cùng một track hoặc cùng một composition tuyệt đối không được phân tán ở cả tập train và test.

9. **Ưu tiên MVP Core trước (Ràng buộc phạm vi công việc)**:
   - Không phát triển Cover Detection nâng cao, RAG, LLM, cloud phức tạp, hay fine-tuning model trước khi pipeline lõi hoạt động ổn định end-to-end:
     `Fingerprint (Chromaprint) → MERT Embedding → Rights DB → Rule Engine → Result`.

---

## 3. Ngoài phạm vi MVP (Không làm sớm)

Tuyệt đối tránh lãng phí tài nguyên vào các hạng mục sau trong giai đoạn MVP:
- Nhận diện hàng triệu bài hát (chỉ tập trung vào quy mô MVP 2.000–3.000 bài).
- Nhận diện trực tiếp theo thời gian thực từ livestream.
- Sao chép toàn bộ hệ thống Content ID của YouTube.
- Đưa ra kết luận tư vấn pháp lý có giá trị tài phán (chỉ dừng ở đánh giá rủi ro kỹ thuật).
- Auto-crawl toàn bộ Internet.
- Huấn luyện foundation model từ đầu.
- Xử lý các bản mashup / remix phức tạp nhiều tầng.
- Cover Detection phức tạp, RAG, LLM trợ lý trước khi core cascade pipeline hoàn thiện.

---

## 4. Cơ chế Tự động Cập nhật Rules, Skills & Memory (Self-Updating System)

Để đảm bảo tài liệu định hướng và mã nguồn thực tế luôn đồng bộ 100%, hệ sinh thái Claude Code bắt buộc tuân thủ nguyên tắc tự cập nhật:

1. **Khi thay đổi Database Schema / Migrations**:
   - Bắt buộc cập nhật bảng và kiểu dữ liệu mới vào [04-database-schemas.md](file:///d:/PycharmProjects/AMR_advanced/.claude/rules/04-database-schemas.md).
2. **Khi thay đổi hoặc bổ sung API Endpoints / Error Codes**:
   - Bắt buộc cập nhật danh mục endpoint, mã lỗi mới vào [05-backend-api.md](file:///d:/PycharmProjects/AMR_advanced/.claude/rules/05-backend-api.md).
3. **Khi tinh chỉnh Cây Quyết định Bản quyền / File Rules YAML**:
   - Bắt buộc cập nhật logic nhóm bản quyền vào [03-rule-engine-copyright.md](file:///d:/PycharmProjects/AMR_advanced/.claude/rules/03-rule-engine-copyright.md) và playbook [rule-engine-verifier](file:///d:/PycharmProjects/AMR_advanced/.claude/skills/rule-engine-verifier/SKILL.md).
4. **Khi thay đổi tham số Pipeline / Ngưỡng τFP / τMERT / Model Version**:
   - Bắt buộc cập nhật [02-pipeline-architecture.md](file:///d:/PycharmProjects/AMR_advanced/.claude/rules/02-pipeline-architecture.md), [06-experiments-evaluation.md](file:///d:/PycharmProjects/AMR_advanced/.claude/rules/06-experiments-evaluation.md) và playbook [music-rights-pipeline](file:///d:/PycharmProjects/AMR_advanced/.claude/skills/music-rights-pipeline/SKILL.md).
5. **Ghi nhận tiến độ phiên làm việc (Session Sync)**:
   - Mọi mốc hoàn thành tính năng hoặc quyết định kiến trúc mới phải được ghi nhận ngay vào [memory.md](file:///d:/PycharmProjects/AMR_advanced/.claude/memory.md).
