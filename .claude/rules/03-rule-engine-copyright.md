---
description: Cây quyết định phân loại 5 nhóm bản quyền, logic Rule Engine if/else thuần và định dạng cấu hình
globs:
  - "backend/services/decision_service.py"
  - "backend/services/rights_service.py"
  - "backend/schemas/**"
  - "configs/**"
---

# 03. Rule Engine & Phân loại Bản quyền (Copyright Rule Engine)

Rule Engine chịu trách nhiệm xác định trạng thái bản quyền và mức độ rủi ro (`risk_level`). Nó là thành phần cốt lõi đảm bảo tính minh bạch và có thể giải thích được của hệ thống.

## 1. Nguyên tắc thiết kế Rule Engine

1. **Logic thuần if/else**: Tuyệt đối không dùng model học máy / xác suất để gán nhãn rủi ro hoặc loại giấy phép.
2. **Tuần tự & Dừng sớm (First-Match Exit)**: Đánh giá lần lượt theo thứ tự ưu tiên từ Nhóm 1 đến Nhóm 5. Dừng ngay tại nhóm đầu tiên thỏa mãn tất cả điều kiện.
3. **Cấu hình độc lập**: Toàn bộ luật đánh giá phải được định nghĩa trong file cấu hình rời (ví dụ: `configs/rules_v1.yaml` hoặc `.json`), không hardcode trong mã nguồn logic xử lý.
4. **Unit test tối thiểu**: Bắt buộc có suite kiểm thử riêng bao phủ tối thiểu **50 – 100 test case** mô phỏng mọi tình huống pháp lý.

---

## 2. Thứ tự đánh giá 5 nhóm bản quyền + Fallback

### Nhóm 1: AUDIO_LIBRARY (Thư viện âm thanh nền tảng)
- **Điều kiện**: `is_in_audio_library == TRUE`.
- **Đánh giá rủi ro**:
  - Nếu `attribution_required == FALSE`:
    - `risk_level`: **`LOW`**
    - `condition`: Sử dụng tự do không cần ghi công.
    - `monetization`: Cho phép kiếm tiền.
  - Nếu `attribution_required == TRUE`:
    - `risk_level`: **`CONDITIONAL`**
    - `condition`: Bắt buộc đính kèm ghi công tác giả theo đúng định dạng được quy định.

---

### Nhóm 2: CREATOR_MUSIC (Chương trình bản quyền nhà sáng tạo)
- **Điều kiện**: `is_in_creator_music == TRUE`.
- **Đánh giá rủi ro**:
  - Nếu `license_purchased == TRUE`:
    - `risk_level`: **`LOW`**
    - `condition`: Giữ 100% doanh thu kiếm tiền theo điều khoản giấy phép đã mua.
  - Nếu `revenue_share_agreed == TRUE`:
    - `risk_level`: **`CONDITIONAL`**
    - `condition`: Chia sẻ doanh thu theo tỷ lệ đã thỏa thuận với chủ sở hữu bản quyền.

---

### Nhóm 3: COMMERCIAL_CONTENT_ID (Bản quyền thương mại kiểm soát qua Content ID)
- **Điều kiện**: `is_matched_content_id == TRUE`.
- **Đánh giá rủi ro**:
  - Nếu `policy_action == MONETIZE_CLAIM`:
    - `risk_level`: **`CONDITIONAL`**
    - `condition`: Video không bị chặn, nhưng doanh thu sẽ được chuyển cho chủ sở hữu bản quyền.
  - Nếu `policy_action == BLOCK_OR_TAKEDOWN`:
    - `risk_level`: **`HIGH`**
    - `condition`: Video sẽ bị chặn phát trên toàn cầu hoặc tại một số vùng lãnh thổ, nguy cơ nhận cảnh cáo bản quyền (copyright strike).

---

### Nhóm 4: CREATIVE_COMMONS (Giấy phép công cộng CC)
- **Điều kiện**: `license_type` thuộc các định dạng Creative Commons.
- **Đánh giá rủi ro**:
  - Khớp chính xác với giấy phép `CC_BY` và thông tin quyền hợp lệ:
    - `risk_level`: **`CONDITIONAL`**
    - `condition`: Bắt buộc ghi nhận tác giả và cung cấp liên kết tới giấy phép.
  - Nếu mục đích sử dụng là thương mại (`commercial_use == TRUE`) nhưng giấy phép có điều khoản phi thương mại (`CC_BY_NC`, `CC_BY_NC_SA`, `CC_BY_NC_ND`):
    - `risk_level`: **`HIGH`**
    - `condition`: Vi phạm điều khoản phi thương mại của giấy phép.

---

### Nhóm 5: PUBLIC_DOMAIN (Phạm vi công cộng)
- **Nguyên tắc phân biệt bắt buộc**: Composition PD ≠ Recording PD!
- **Đánh giá rủi ro**:
  - Nếu `composition_pd == TRUE` VÀ `recording_pd == TRUE`:
    - `category`: `PUBLIC_DOMAIN_FULL`
    - `risk_level`: **`LOW`**
    - `condition`: Hoàn toàn tự do sử dụng cho mọi mục đích.
  - Nếu `composition_pd == TRUE` nhưng `recording_pd == FALSE` (hoặc bản ghi có bản quyền thương mại):
    - `category`: **`PUBLIC_DOMAIN_COMPOSITION_ONLY`**
    - `risk_level`: **`CONDITIONAL`** hoặc **`HIGH`**
    - `condition`: Tác phẩm (nốt nhạc, lời) thuộc phạm vi công cộng nhưng bản thu âm cụ thể vẫn đang được bảo hộ bản quyền. Bạn chỉ được phép tự cover/thu âm lại, không được sử dụng bản ghi này.

---

### Nhóm 6: Fallback (Xử lý ngoại lệ và nội dung người dùng)
Được kích hoạt khi **không khớp** bất kỳ nhóm nào ở trên:
- Nếu `recording_status == UNKNOWN` HOẶC `confidence < THRESHOLD_MIN`:
  - `category`: **`UNCATEGORIZED`**
  - `risk_level`: **`UNKNOWN`**
  - `condition`: `HUMAN_REVIEW_REQUIRED` (Cần con người thẩm định thủ công, không tự ý cấp phép hoặc cảnh báo sai lệch).
- Nếu dữ liệu không khớp bất kỳ bản ghi nào trong Database (chưa có tuyên bố bản quyền):
  - `category`: **`USER_GENERATED_CONTENT`**
  - `risk_level`: **`LOW`**
  - `condition`: `MONETIZABLE_UNTIL_RETROACTIVE_CLAIM` (Tạm thời kiếm tiền bình thường cho đến khi có khiếu nại bản quyền hồi tố từ chủ sở hữu).

---

### Cổng dữ liệu quyền (`rights_gate`) — chạy SAU khi nhóm 1–5 đã ra kết luận tạm
- **Điều kiện**: `rights_confidence < rights_gate.min_rights_confidence` (hiện `0.50`, khai báo trong `configs/rules_v1.yaml`) và kết luận tạm không phải `UNKNOWN`.
- **Kết quả**: `category: UNCATEGORIZED`, `risk_level: UNKNOWN`, `condition: HUMAN_REVIEW_REQUIRED`.
- Kết luận tạm **không bị bỏ đi**: nằm ở `evidence.provisional_decision` (`category`, `risk_level`, `condition`, `rule_id`, `matched_conditions`, `reason`), kèm `min_rights_confidence` và `rights_shortfall`.
- **Lý do**: giấy phép do license classifier suy đoán từ âm thanh (nguồn `PREDICTED`) vẫn đi qua đúng nhánh và có thể ra `HIGH` — đó là kết luận từ kết quả nghi ngờ mà §2.3/§2.5 cấm.
- `rights_confidence = base − các khoản phạt` trong `rights_confidence.penalties`. Ngưỡng 0.50 tách được: `PREDICTED` (≤ 0.40, bị chặn) / `SIMULATED` đã xác minh (0.70, qua) / `SIMULATED` chưa xác minh (1.00 − 0.30 − 0.20 − 0.10 `simulated_unverified` = **0.40, bị chặn**). **Đổi bảng phạt thì phải xem lại ngưỡng** — `tests/test_license_classifier.py` kiểm tra đúng ba ràng buộc này.
- Vì sao có `simulated_unverified` (2026-09-16): thiếu khoản này thì SIMULATED chưa xác minh dừng đúng 0.50, mà cổng chỉ chặn khi `< 0.50`, nên dữ liệu tổng hợp chưa đối chiếu nguồn nào vẫn ra LOW/HIGH chắc nịch. Nhóm bị ảnh hưởng: 100.000 bản ghi nhạc Việt `dataset_G_vietnam_100k_api` (tên bài dạng "Tác phẩm Nhạc Việt #000001", ID Spotify giả) và nhãn giấy phép Jamendo suy từ cờ tải về. YouTube Audio Library / Creator Music là metadata mô phỏng **có** `verified_at` nên vẫn 0.70 — đúng vai trò ca kiểm thử cho Rule Engine theo đề cương.

### Nhánh ngoại lệ `uncategorized`
- Loại giấy phép không thuộc 5 nhóm (vd. `COVER_MECHANICAL_LICENSE`, 10.032 dòng) → `UNKNOWN` + `HUMAN_REVIEW_REQUIRED`, `rule_id = uncategorized`. Câu lý do **nêu đích danh** loại giấy phép gặp phải, để phân biệt "chưa được hỗ trợ" với "thiếu dữ liệu".

---

## 3. Cấu trúc đối tượng đầu ra (Decision Object Schema)

Mỗi lần chạy qua Rule Engine phải sinh ra đối tượng kết quả với tối thiểu các trường sau:

```json
{
  "category": "CREATIVE_COMMONS",
  "risk_level": "CONDITIONAL",
  "identity_confidence": 0.96,
  "rights_confidence": 0.90,
  "decision_confidence": 0.93,
  "rules_triggered": [
    "RULE_CC_EXACT_MATCH",
    "RULE_CC_BY_ATTRIBUTION_REQUIRED"
  ],
  "conditions": [
    "Must include attribution: 'Track Title by Artist under CC BY 4.0'",
    "Provide a link to the original license"
  ],
  "recommendation": "Safe to use for commercial purposes provided the attribution credit is added to the video description."
}
```
