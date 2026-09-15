"""
Test cho ánh xạ giấy phép thật -> taxonomy Rule Engine.

Hai thứ được khẳng định ở đây, và cái thứ hai mới là cái quan trọng:

1. Đọc đúng mã CC từ cả URL lẫn tên dạng chữ (FMA ghi tên, Jamendo trả URL).
2. Mọi `license_type` sinh ra đều nằm trong tập mà `configs/rules_v1.yaml` thật
   sự khai báo. Sinh ra một mã lạ nghĩa là tạo bản ghi không rule nào chạm tới,
   và Rule Engine sẽ lặng lẽ rơi xuống nhánh `uncategorized` — hỏng mà không báo.
"""
import os

import pytest
import yaml

from backend import config
from backend.services.decision_service import evaluate_rights_and_risk
from backend.services.license_mapping import normalize_license

CC_BASE = "http://creativecommons.org/licenses"


# --------------------------------------------------------------------------
# Đọc từ URL
# --------------------------------------------------------------------------
@pytest.mark.parametrize("code,expected", [
    ("by", "CC_BY"),
    ("by-sa", "CC_BY_SA"),
    ("by-nc", "CC_BY_NC"),
    ("by-nc-sa", "CC_BY_NC_SA"),
    ("by-nd", "CC_BY_ND"),
    ("by-nc-nd", "CC_BY_NC_ND"),
])
def test_doc_ma_cc_tu_url(code, expected):
    result = normalize_license(license_url=f"{CC_BASE}/{code}/4.0/")
    assert result["license_type"] == expected
    assert result["matched_on"] == "url"


def test_cc0_va_public_domain_mark():
    zero = normalize_license(license_url="http://creativecommons.org/publicdomain/zero/1.0/")
    assert zero["license_type"] == "CC0"
    assert zero["recording_public_domain"] is True
    assert zero["attribution_required"] is False

    mark = normalize_license(license_url="https://creativecommons.org/publicdomain/mark/1.0/")
    assert mark["license_type"] == "PUBLIC_DOMAIN"
    assert mark["copyright_status"] == "PUBLIC_DOMAIN"


# --------------------------------------------------------------------------
# Đọc từ tên dạng chữ (FMA)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("Attribution 4.0 International", "CC_BY"),
    ("Attribution-NonCommercial-ShareAlike 3.0 International", "CC_BY_NC_SA"),
    ("Creative Commons Attribution-NonCommercial-NoDerivatives 4.0", "CC_BY_NC_ND"),
    ("Attribution-ShareAlike 3.0", "CC_BY_SA"),
    ("Attribution-NoDerivatives 4.0", "CC_BY_ND"),
])
def test_doc_ma_cc_tu_ten_chu(text, expected):
    result = normalize_license(license_text=text)
    assert result["license_type"] == expected
    assert result["matched_on"] == "text"


def test_url_duoc_uu_tien_hon_ten_chu():
    """URL là dạng máy đọc được và không mơ hồ, nên phải thắng khi hai bên lệch."""
    result = normalize_license(license_text="Attribution 4.0",
                               license_url=f"{CC_BASE}/by-nc/4.0/")
    assert result["license_type"] == "CC_BY_NC"
    assert result["matched_on"] == "url"


def test_thu_tu_trong_ma_khong_quan_trong():
    assert normalize_license(license_url=f"{CC_BASE}/nc-by/4.0/")["license_type"] == "CC_BY_NC"


# --------------------------------------------------------------------------
# Không đoán bừa (§2)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {},
    {"license_text": ""},
    {"license_text": "All rights reserved"},
    {"license_text": "Bản quyền thuộc về hãng đĩa"},
    {"license_url": "https://example.com/giay-phep-rieng"},
    # sa và nd loại trừ nhau theo định nghĩa CC -> dữ liệu hỏng, không được đoán
    {"license_url": f"{CC_BASE}/by-sa-nd/4.0/"},
])
def test_khong_nhan_dien_duoc_thi_tra_unknown(kwargs):
    result = normalize_license(**kwargs)
    assert result["license_type"] == "UNKNOWN"
    assert result["attribution_required"] is None
    assert result["commercial_use_allowed"] is None
    assert result["matched_on"] is None


# --------------------------------------------------------------------------
# Cờ quyền suy từ chính điều khoản
# --------------------------------------------------------------------------
def test_nc_cam_ca_thuong_mai_lan_kiem_tien():
    """Bật kiếm tiền là một hình thức sử dụng thương mại, NC chi phối cả hai."""
    result = normalize_license(license_url=f"{CC_BASE}/by-nc/4.0/")
    assert result["commercial_use_allowed"] is False
    assert result["monetization_allowed"] is False


def test_nd_cam_phai_sinh_nhung_khong_cam_thuong_mai():
    result = normalize_license(license_url=f"{CC_BASE}/by-nd/4.0/")
    assert result["modification_allowed"] is False
    assert result["commercial_use_allowed"] is True


def test_moi_bien_the_by_deu_buoc_ghi_cong():
    for code in ("by", "by-sa", "by-nc", "by-nc-sa", "by-nd", "by-nc-nd"):
        result = normalize_license(license_url=f"{CC_BASE}/{code}/4.0/")
        assert result["attribution_required"] is True, code
        # Giấy phép CC không làm tác phẩm hết bản quyền, chỉ cấp phép sử dụng
        assert result["copyright_status"] == "PROTECTED", code
        assert result["recording_public_domain"] is False, code


# --------------------------------------------------------------------------
# Khớp với Rule Engine — phần quan trọng nhất
# --------------------------------------------------------------------------
def _declared_license_types() -> set:
    with open(os.path.join(config.BASE_DIR, "configs", "rules_v1.yaml"),
              encoding="utf-8") as f:
        rules = yaml.safe_load(f)

    declared = set()
    for group in rules["groups"]:
        match = group.get("match", {})
        for source in (match, match.get("any_of", {})):
            value = source.get("license_type")
            if isinstance(value, list):
                declared.update(value)
            elif value:
                declared.add(value)
    return declared


@pytest.mark.parametrize("code", ["by", "by-sa", "by-nc", "by-nc-sa", "by-nd",
                                  "by-nc-nd", "publicdomain/zero", "publicdomain/mark"])
def test_moi_license_type_sinh_ra_deu_co_rule_nhan(code):
    url = (f"https://creativecommons.org/{code}/1.0/" if "publicdomain" in code
           else f"{CC_BASE}/{code}/4.0/")
    assert normalize_license(license_url=url)["license_type"] in _declared_license_types()


def test_cc_by_nc_dung_thuong_mai_bi_Rule_Engine_bat_la_HIGH():
    """Kiểm tra đầu-cuối: giấy phép thật đi qua ánh xạ rồi tới quyết định."""
    rights = normalize_license(license_url=f"{CC_BASE}/by-nc/4.0/")
    rights["source"] = "Jamendo API"

    decision = evaluate_rights_and_risk(
        rights_data=rights,
        match_info={"match_type": "EXACT_MATCH", "identity_confidence": 0.99},
        usage_context={"platform": "youtube", "commercial_use": True,
                       "monetization": True},
    )
    assert decision["risk_level"] == "HIGH"
    assert decision["condition"] == "NON_COMMERCIAL_VIOLATION"


def test_cc_by_phi_thuong_mai_ra_CONDITIONAL_vi_phai_ghi_cong():
    rights = normalize_license(license_url=f"{CC_BASE}/by/4.0/")
    rights["source"] = "FMA"

    decision = evaluate_rights_and_risk(
        rights_data=rights,
        match_info={"match_type": "EXACT_MATCH", "identity_confidence": 0.99},
        usage_context={"platform": "youtube", "commercial_use": False,
                       "monetization": False},
    )
    assert decision["risk_level"] == "CONDITIONAL"
    assert decision["condition"] == "ATTRIBUTION_REQUIRED"


def test_cc0_ra_LOW_free_to_use():
    rights = normalize_license(license_url="https://creativecommons.org/publicdomain/zero/1.0/")
    rights["source"] = "FMA"

    decision = evaluate_rights_and_risk(
        rights_data=rights,
        match_info={"match_type": "EXACT_MATCH", "identity_confidence": 0.99},
        usage_context={"platform": "youtube", "commercial_use": True,
                       "monetization": True},
    )
    assert decision["risk_level"] == "LOW"
    assert decision["condition"] == "FREE_TO_USE"


def test_license_khong_ro_thi_Rule_Engine_tra_UNKNOWN_chu_khong_ep():
    rights = normalize_license(license_text="All rights reserved")
    rights["source"] = "FMA"

    decision = evaluate_rights_and_risk(
        rights_data=rights,
        match_info={"match_type": "EXACT_MATCH", "identity_confidence": 0.99},
        usage_context={"platform": "youtube", "commercial_use": False,
                       "monetization": False},
    )
    assert decision["risk_level"] == "UNKNOWN"
    assert decision["condition"] == "HUMAN_REVIEW_REQUIRED"
