---
name: rule-engine-verifier
description: Playbook kiểm thử, thẩm định và mở rộng Rule Engine 5 nhóm bản quyền (Audio Library, Creator Music, Content ID, CC, Public Domain + Fallback) và xác thực tính tuân thủ pháp lý.
---

# Skill: Kiểm định & Mở rộng Rule Engine (`rule-engine-verifier`)

Skill này hướng dẫn chi tiết phương pháp vận hành, kiểm định tính đúng đắn, xây dựng bộ test suite và cập nhật các luật phân loại trong **Rule Engine** của hệ thống.

---

## 1. Nguyên lý Thiết kế & Cây Quyết định Tuần tự

Rule Engine hoạt động dựa trên cây quyết định tuần tự, dừng lại ngay tại nhóm đầu tiên thỏa mãn điều kiện (**First-Match Exit**):

```
1. AUDIO_LIBRARY
   ├─ attribution_required == FALSE ──> Risk: LOW (Monetizable)
   └─ attribution_required == TRUE  ──> Risk: CONDITIONAL (Need Attribution)

2. CREATOR_MUSIC
   ├─ license_purchased == TRUE     ──> Risk: LOW (Keep 100% Rev)
   └─ revenue_share_agreed == TRUE  ──> Risk: CONDITIONAL (Rev Share)

3. COMMERCIAL_CONTENT_ID
   ├─ policy == MONETIZE_CLAIM      ──> Risk: CONDITIONAL (Rev Claimed)
   └─ policy == BLOCK_OR_TAKEDOWN   ──> Risk: HIGH (Strike / Block)

4. CREATIVE_COMMONS
   ├─ CC_BY & Valid Rights          ──> Risk: CONDITIONAL (Attribution)
   └─ Commercial Use & CC_BY_NC     ──> Risk: HIGH (License Violation)

5. PUBLIC_DOMAIN
   ├─ Comp PD == TRUE & Rec PD == TRUE ──> Risk: LOW (Full PD)
   └─ Comp PD == TRUE & Rec PD == FALSE ──> Risk: CONDITIONAL / HIGH (Composition Only PD)

6. FALLBACK
   ├─ Unknown status / Low conf     ──> Risk: UNKNOWN (Human Review Required)
   └─ Unmatched in Database         ──> Risk: LOW (User-Generated Content)

CỔNG DỮ LIỆU QUYỀN (chạy SAU khi nhóm 1–5 đã ra kết luận tạm)
   └─ rights_confidence < rights_gate.min_rights_confidence (0.50)
                                    ──> Risk: UNKNOWN (Human Review Required)
                                        kết luận tạm giữ ở evidence.provisional_decision
```

---

## 2. Quy trình Thẩm định Luật (Verification Workflow)

### Bước 1: Kiểm tra Tính bất biến của Luật (Invariants Check)
Khi thêm hoặc chỉnh sửa bất kỳ rule nào trong `configs/rules_v1.yaml`, luôn đối chiếu với 3 tiêu chí:
1. **Quy tắc Composition ≠ Recording**:
   - Luật KHÔNG ĐƯỢC PHÉP cấp quyền tự do (Risk LOW) nếu chỉ có `composition_pd = true` mà `recording_pd = false`.
2. **Quy tắc UNKNOWN**:
   - Trạng thái chưa rõ bản quyền hoặc độ tin cậy thấp (`confidence < THRESHOLD_MIN`) BẮT BUỘC trả về `risk_level: UNKNOWN`. Không được tự tiện fallback về `LOW` hay `HIGH`.
3. **Quy tắc Minh bạch (No Black-Box)**:
   - Output phải chứa danh sách cụ thể các mã luật được kích hoạt trong trường `rules_triggered`.
4. **Quy tắc Tin cậy Dữ liệu quyền**:
   - Kết luận LOW / CONDITIONAL / HIGH chỉ được trả ra khi `rights_confidence` đạt `rights_gate.min_rights_confidence`; dưới ngưỡng phải là `UNKNOWN`, kết luận tạm giữ ở `evidence.provisional_decision`.
   - Mọi con số độ tin cậy phải truy ngược được: `evidence.rights_confidence_breakdown.formula` và `evidence.decision_confidence_formula`.

### Bước 2: Viết & Chạy Unit Tests cho Rule Engine
Mỗi luật mới phải đi kèm tối thiểu 5–10 ca kiểm thử biên (edge cases). Cấu trúc một test case mẫu với `pytest`:

```python
def test_public_domain_composition_only():
    """Kiểm thử trường hợp tác phẩm hết bản quyền nhưng bản thu còn bản quyền."""
    evidence = {
        "composition_pd_status": "verified",
        "recording_copyright_status": "PROTECTED",
        "license_type": "ALL_RIGHTS_RESERVED",
        "confidence": 0.95
    }
    result = decision_service.evaluate(evidence)
    
    assert result.category == "PUBLIC_DOMAIN_COMPOSITION_ONLY"
    assert result.risk_level in ["CONDITIONAL", "HIGH"]
    assert "RULE_PD_COMPOSITION_ONLY" in result.rules_triggered
    assert result.recommendation is not None
```

### Bước 3: Đảm bảo Độ bao phủ Suite Kiểm thử
- Bộ test suite phải duy trì tối thiểu **50 – 100 test cases**.
- Chạy kiểm tra toàn bộ suite:
  ```bash
  pytest tests/test_decision_rules.py -v --cov=backend/services/decision_service
  ```

---

## 3. Cấu trúc Output Chuẩn của Decision Object

Khi tích hợp với API, đảm bảo Decision Object trả về tuân thủ đầy đủ schema:

```json
{
  "category": "CREATIVE_COMMONS",
  "risk_level": "CONDITIONAL",
  "identity_confidence": 0.96,
  "rights_confidence": 0.92,
  "decision_confidence": 0.94,
  "rules_triggered": [
    "RULE_CC_EXACT_MATCH",
    "RULE_CC_BY_ATTRIBUTION_REQUIRED"
  ],
  "conditions": [
    "Credit artist in video description: 'Song Name by Artist Name (CC BY 4.0)'",
    "Include link to Creative Commons license"
  ],
  "recommendation": "Safe for commercial use provided full attribution credit is displayed.",
  "timestamp": "2026-09-15T05:50:00Z"
}
```
