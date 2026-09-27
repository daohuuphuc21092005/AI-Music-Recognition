"""
Bộ test Rule Engine — CLAUDE.md §9 yêu cầu tối thiểu 50–100 case phủ mọi nhánh.

Các test này KHÔNG cần database: Rule Engine là hàm thuần, nhận vào dữ liệu
quyền + kết quả nhận diện + mục đích sử dụng, trả ra quyết định.
"""
from datetime import date, timedelta

import pytest

from backend.services.decision_service import (
    build_facts,
    compute_rights_confidence,
    evaluate_rights_and_risk,
    load_rules,
)

TODAY = date.today()
VALID_FROM = str(TODAY - timedelta(days=365))
VALID_UNTIL = str(TODAY + timedelta(days=365))
EXPIRED_UNTIL = str(TODAY - timedelta(days=30))

# Lấy ngưỡng TỪ CHÍNH rules thay vì viết cứng con số. Bản trước hard-code 0.95
# — đúng bằng τMERT lúc đó — nên khi ngưỡng được hiệu chỉnh lại theo dữ liệu thật
# (0.95 -> 0.97, vì EXP-06 đo được 0.95 cho False Match Rate 14,7%) thì case
# "NEAR_MATCH hợp lệ" lặng lẽ tụt xuống dưới ngưỡng và test fail vì lý do chẳng
# liên quan gì tới điều nó muốn kiểm. Đọc động thì mọi lần hiệu chỉnh sau này
# đều không làm hỏng bộ test.
_GATE = load_rules()["identity_gate"]
_BY_TYPE = _GATE.get("min_identity_confidence_by_match_type") or {}
_MIN_EXACT = float(_BY_TYPE.get("EXACT_MATCH", 0.70))
_MIN_NEAR = float(_BY_TYPE.get("NEAR_MATCH", 0.95))

MATCH_OK = {"match_type": "EXACT_MATCH", "identity_confidence": 0.99}
# Vừa đủ ĐẠT ngưỡng NEAR_MATCH: đây mới là "khớp hợp lệ ở tầng 2"
MATCH_NEAR = {"match_type": "NEAR_MATCH", "identity_confidence": _MIN_NEAR}
# Rõ ràng DƯỚI ngưỡng, để kiểm cổng identity chặn đúng
MATCH_LOW = {"match_type": "NEAR_MATCH", "identity_confidence": round(_MIN_NEAR / 2, 4)}
MATCH_UNKNOWN = {"match_type": "UNKNOWN", "identity_confidence": 0.42}
MATCH_NONE = {"match_type": "NO_MATCH", "identity_confidence": 0.0}

NON_COMMERCIAL = {"platform": "YOUTUBE", "commercial_use": False, "monetization": False}
COMMERCIAL = {"platform": "YOUTUBE", "commercial_use": True, "monetization": True}


def rights(**overrides) -> dict:
    """Bản ghi quyền tối thiểu, hợp lệ, đã xác minh từ nguồn thật."""
    base = {
        "license_type": "CC_BY",
        "copyright_status": "PROTECTED",
        "attribution_required": True,
        "commercial_use_allowed": True,
        "modification_allowed": True,
        "monetization_allowed": True,
        "revenue_share_required": False,
        "policy_action": "NONE",
        "license_purchased": False,
        "revenue_share_agreed": False,
        "recording_public_domain": False,
        "valid_from": VALID_FROM,
        "valid_until": VALID_UNTIL,
        "source": "MTG-Jamendo",
        "verified_at": f"{TODAY}T00:00:00",
        "composition": {"public_domain_status": "no"},
    }
    composition = overrides.pop("composition", None)
    base.update(overrides)
    if composition is not None:
        base["composition"] = composition
    return base


# ===========================================================================
# 1. Cổng định danh — chạy TRƯỚC mọi nhóm (§2: không ép UNKNOWN)
# ===========================================================================
def test_khong_khop_gi_ca_thi_la_ugc():
    d = evaluate_rights_and_risk(None, MATCH_NONE, NON_COMMERCIAL)
    assert d["category"] == "USER_GENERATED_CONTENT"
    assert d["risk_level"] == "LOW"
    assert d["condition"] == "MONETIZABLE_UNTIL_RETROACTIVE_CLAIM"


def test_match_type_unknown_thi_luon_unknown():
    d = evaluate_rights_and_risk(rights(), MATCH_UNKNOWN, NON_COMMERCIAL)
    assert d["risk_level"] == "UNKNOWN"
    assert d["condition"] == "HUMAN_REVIEW_REQUIRED"


def test_do_tin_cay_thap_thi_khong_ap_dung_rights():
    d = evaluate_rights_and_risk(rights(license_type="AUDIO_LIBRARY",
                                        attribution_required=False),
                                 MATCH_LOW, NON_COMMERCIAL)
    # Dù giấy phép là LOW risk, không đủ tin cậy định danh thì vẫn phải UNKNOWN
    assert d["risk_level"] == "UNKNOWN"
    assert d["evidence"]["rule_id"] == "identity_gate.low_confidence"
    assert d["evidence"]["identity_shortfall"] > 0


def test_dinh_danh_duoc_nhung_khong_co_rights():
    d = evaluate_rights_and_risk(None, MATCH_OK, NON_COMMERCIAL)
    assert d["risk_level"] == "UNKNOWN"
    assert d["evidence"]["rule_id"] == "identity_gate.missing_rights"


def test_unknown_thi_decision_confidence_bang_0():
    d = evaluate_rights_and_risk(rights(), MATCH_UNKNOWN, NON_COMMERCIAL)
    assert d["decision_confidence"] == 0.0


# ===========================================================================
# 2. Bảng case chính — mỗi dòng là một nhánh của rules_v1.yaml
# ===========================================================================
CASES = [
    # --- Nhóm 1: Audio Library -------------------------------------------
    ("AL: không cần ghi công",
     rights(license_type="AUDIO_LIBRARY", attribution_required=False),
     MATCH_OK, NON_COMMERCIAL, "AUDIO_LIBRARY", "LOW", "FREE_TO_USE"),
    ("AL: không cần ghi công + thương mại",
     rights(license_type="AUDIO_LIBRARY", attribution_required=False),
     MATCH_OK, COMMERCIAL, "AUDIO_LIBRARY", "LOW", "FREE_TO_USE"),
    ("AL: cần ghi công",
     rights(license_type="AUDIO_LIBRARY", attribution_required=True),
     MATCH_OK, NON_COMMERCIAL, "AUDIO_LIBRARY", "CONDITIONAL", "ATTRIBUTION_REQUIRED"),
    ("AL: cần ghi công + thương mại",
     rights(license_type="AUDIO_LIBRARY", attribution_required=True),
     MATCH_OK, COMMERCIAL, "AUDIO_LIBRARY", "CONDITIONAL", "ATTRIBUTION_REQUIRED"),
    ("AL: thiếu thông tin ghi công",
     rights(license_type="AUDIO_LIBRARY", attribution_required=None),
     MATCH_OK, NON_COMMERCIAL, "AUDIO_LIBRARY", "UNKNOWN", "HUMAN_REVIEW_REQUIRED"),
    ("AL: khớp qua NEAR_MATCH vẫn áp dụng được",
     rights(license_type="AUDIO_LIBRARY", attribution_required=False),
     MATCH_NEAR, NON_COMMERCIAL, "AUDIO_LIBRARY", "LOW", "FREE_TO_USE"),

    # --- Nhóm 2: Creator Music -------------------------------------------
    ("CM: đã mua, còn hạn",
     rights(license_type="CREATOR_MUSIC", license_purchased=True),
     MATCH_OK, COMMERCIAL, "CREATOR_MUSIC", "LOW", "FULL_REVENUE_RETAINED"),
    ("CM: đã mua nhưng hết hạn",
     rights(license_type="CREATOR_MUSIC", license_purchased=True,
            valid_until=EXPIRED_UNTIL),
     MATCH_OK, COMMERCIAL, "CREATOR_MUSIC", "UNKNOWN", "HUMAN_REVIEW_REQUIRED"),
    ("CM: chưa mua nhưng đồng ý chia doanh thu",
     rights(license_type="CREATOR_MUSIC", license_purchased=False,
            revenue_share_agreed=True),
     MATCH_OK, COMMERCIAL, "CREATOR_MUSIC", "CONDITIONAL", "REVENUE_SHARE_APPLIED"),
    ("CM: chưa mua, chưa đồng ý chia doanh thu",
     rights(license_type="CREATOR_MUSIC", license_purchased=False,
            revenue_share_agreed=False),
     MATCH_OK, COMMERCIAL, "CREATOR_MUSIC", "HIGH", "LICENSE_REQUIRED"),
    ("CM: chưa mua, chưa đồng ý, dùng phi thương mại",
     rights(license_type="CREATOR_MUSIC", license_purchased=False,
            revenue_share_agreed=False),
     MATCH_OK, NON_COMMERCIAL, "CREATOR_MUSIC", "HIGH", "LICENSE_REQUIRED"),
    ("CM: thiếu trạng thái giấy phép",
     rights(license_type="CREATOR_MUSIC", license_purchased=None,
            revenue_share_agreed=None),
     MATCH_OK, COMMERCIAL, "CREATOR_MUSIC", "UNKNOWN", "HUMAN_REVIEW_REQUIRED"),
    ("CM: chưa tới ngày hiệu lực",
     rights(license_type="CREATOR_MUSIC", license_purchased=True,
            valid_from=str(TODAY + timedelta(days=10))),
     MATCH_OK, COMMERCIAL, "CREATOR_MUSIC", "UNKNOWN", "HUMAN_REVIEW_REQUIRED"),

    # --- Nhóm 3: Content ID / Thương mại ---------------------------------
    ("CID: chính sách nhận tiền",
     rights(license_type="CONTENT_ID", policy_action="MONETIZE_CLAIM"),
     MATCH_OK, COMMERCIAL, "COMMERCIAL_CONTENT_ID", "CONDITIONAL", "REVENUE_REDIRECTED"),
    ("CID: chính sách chặn",
     rights(license_type="CONTENT_ID", policy_action="BLOCK_OR_TAKEDOWN"),
     MATCH_OK, COMMERCIAL, "COMMERCIAL_CONTENT_ID", "HIGH", "VIDEO_BLOCKED_OR_STRIKE"),
    ("CID: chặn — kể cả dùng phi thương mại",
     rights(license_type="CONTENT_ID", policy_action="BLOCK_OR_TAKEDOWN"),
     MATCH_OK, NON_COMMERCIAL, "COMMERCIAL_CONTENT_ID", "HIGH", "VIDEO_BLOCKED_OR_STRIKE"),
    ("CID: chưa rõ chính sách (NONE)",
     rights(license_type="CONTENT_ID", policy_action="NONE"),
     MATCH_OK, COMMERCIAL, "COMMERCIAL_CONTENT_ID", "UNKNOWN", "HUMAN_REVIEW_REQUIRED"),
    ("CID: chưa rõ chính sách (null)",
     rights(license_type="CONTENT_ID", policy_action=None),
     MATCH_OK, COMMERCIAL, "COMMERCIAL_CONTENT_ID", "UNKNOWN", "HUMAN_REVIEW_REQUIRED"),
    ("CID: license_type COMMERCIAL cũng thuộc nhóm 3",
     rights(license_type="COMMERCIAL", policy_action="MONETIZE_CLAIM"),
     MATCH_OK, COMMERCIAL, "COMMERCIAL_CONTENT_ID", "CONDITIONAL", "REVENUE_REDIRECTED"),
    ("CID: giá trị chính sách lạ -> default",
     rights(license_type="CONTENT_ID", policy_action="SOMETHING_ELSE"),
     MATCH_OK, COMMERCIAL, "COMMERCIAL_CONTENT_ID", "UNKNOWN", "HUMAN_REVIEW_REQUIRED"),

    # --- Nhóm 4: Creative Commons ----------------------------------------
    ("CC_BY: phi thương mại",
     rights(license_type="CC_BY"),
     MATCH_OK, NON_COMMERCIAL, "CREATIVE_COMMONS", "CONDITIONAL", "ATTRIBUTION_REQUIRED"),
    ("CC_BY: thương mại vẫn hợp lệ",
     rights(license_type="CC_BY"),
     MATCH_OK, COMMERCIAL, "CREATIVE_COMMONS", "CONDITIONAL", "ATTRIBUTION_REQUIRED"),
    ("CC_BY_SA: như CC_BY",
     rights(license_type="CC_BY_SA"),
     MATCH_OK, COMMERCIAL, "CREATIVE_COMMONS", "CONDITIONAL", "ATTRIBUTION_REQUIRED"),
    ("CC_BY_NC + thương mại -> VI PHẠM",
     rights(license_type="CC_BY_NC", commercial_use_allowed=False,
            monetization_allowed=False),
     MATCH_OK, COMMERCIAL, "CREATIVE_COMMONS", "HIGH", "NON_COMMERCIAL_VIOLATION"),
    ("CC_BY_NC + phi thương mại -> hợp lệ",
     rights(license_type="CC_BY_NC", commercial_use_allowed=False,
            monetization_allowed=False),
     MATCH_OK, NON_COMMERCIAL, "CREATIVE_COMMONS", "CONDITIONAL", "ATTRIBUTION_REQUIRED"),
    ("CC_BY_NC_ND + thương mại -> VI PHẠM",
     rights(license_type="CC_BY_NC_ND", commercial_use_allowed=False,
            modification_allowed=False, monetization_allowed=False),
     MATCH_OK, COMMERCIAL, "CREATIVE_COMMONS", "HIGH", "NON_COMMERCIAL_VIOLATION"),
    ("CC: cấm kiếm tiền nhưng bật kiếm tiền",
     rights(license_type="CC_BY", commercial_use_allowed=True,
            monetization_allowed=False),
     MATCH_OK, {"platform": "YOUTUBE", "commercial_use": False, "monetization": True},
     "CREATIVE_COMMONS", "HIGH", "MONETIZATION_NOT_PERMITTED"),
    ("CC0: tự do hoàn toàn",
     rights(license_type="CC0", attribution_required=False),
     MATCH_OK, COMMERCIAL, "CREATIVE_COMMONS", "LOW", "FREE_TO_USE"),
    ("CC: không yêu cầu ghi công",
     rights(license_type="CC_BY", attribution_required=False),
     MATCH_OK, NON_COMMERCIAL, "CREATIVE_COMMONS", "LOW", "FREE_TO_USE"),
    ("CC: thiếu thông tin ghi công -> default",
     rights(license_type="CC_BY", attribution_required=None),
     MATCH_OK, NON_COMMERCIAL, "CREATIVE_COMMONS", "UNKNOWN", "HUMAN_REVIEW_REQUIRED"),
    ("CC: license_type CREATIVE_COMMONS chung chung",
     rights(license_type="CREATIVE_COMMONS"),
     MATCH_OK, NON_COMMERCIAL, "CREATIVE_COMMONS", "CONDITIONAL", "ATTRIBUTION_REQUIRED"),

    # --- Nhóm 5: Public Domain -------------------------------------------
    ("PD: tác phẩm PD + bản thu PD",
     rights(license_type="PUBLIC_DOMAIN", copyright_status="PUBLIC_DOMAIN",
            recording_public_domain=True,
            composition={"public_domain_status": "verified"}),
     MATCH_OK, COMMERCIAL, "PUBLIC_DOMAIN", "LOW", "FREE_TO_USE"),
    ("PD: tác phẩm PD, bản thu CÒN bảo hộ + thương mại",
     rights(license_type="PUBLIC_DOMAIN", copyright_status="PROTECTED",
            recording_public_domain=False,
            composition={"public_domain_status": "verified"}),
     MATCH_OK, COMMERCIAL, "PUBLIC_DOMAIN", "HIGH", "RECORDING_PERMISSION_REQUIRED"),
    ("PD: tác phẩm PD, bản thu CÒN bảo hộ + phi thương mại",
     rights(license_type="PUBLIC_DOMAIN", copyright_status="PROTECTED",
            recording_public_domain=False,
            composition={"public_domain_status": "verified"}),
     MATCH_OK, NON_COMMERCIAL, "PUBLIC_DOMAIN", "CONDITIONAL",
     "RECORDING_PERMISSION_REQUIRED"),
    ("PD: tình trạng tác phẩm mới chỉ 'possible'",
     rights(license_type="PUBLIC_DOMAIN", recording_public_domain=True,
            composition={"public_domain_status": "possible"}),
     MATCH_OK, COMMERCIAL, "PUBLIC_DOMAIN", "UNKNOWN", "HUMAN_REVIEW_REQUIRED"),
    ("PD: tình trạng tác phẩm 'unknown'",
     rights(license_type="PUBLIC_DOMAIN", recording_public_domain=True,
            composition={"public_domain_status": "unknown"}),
     MATCH_OK, COMMERCIAL, "PUBLIC_DOMAIN", "UNKNOWN", "HUMAN_REVIEW_REQUIRED"),
    ("PD: thiếu hẳn thông tin tác phẩm",
     rights(license_type="PUBLIC_DOMAIN", recording_public_domain=True,
            composition={"public_domain_status": None}),
     MATCH_OK, COMMERCIAL, "PUBLIC_DOMAIN", "UNKNOWN", "HUMAN_REVIEW_REQUIRED"),
    ("PD: bản thu PD nhưng TÁC PHẨM còn bảo hộ",
     rights(license_type="PUBLIC_DOMAIN", recording_public_domain=True,
            composition={"public_domain_status": "no"}),
     MATCH_OK, COMMERCIAL, "PUBLIC_DOMAIN", "CONDITIONAL",
     "COMPOSITION_PERMISSION_REQUIRED"),
    ("PD: nhận diện qua copyright_status thay vì license_type",
     rights(license_type="ARCHIVE", copyright_status="PUBLIC_DOMAIN",
            recording_public_domain=True,
            composition={"public_domain_status": "verified"}),
     MATCH_OK, COMMERCIAL, "PUBLIC_DOMAIN", "LOW", "FREE_TO_USE"),

    # --- Nhóm ngoại lệ ----------------------------------------------------
    ("Loại giấy phép lạ -> UNCATEGORIZED",
     rights(license_type="MOT_LOAI_LA"),
     MATCH_OK, NON_COMMERCIAL, "UNCATEGORIZED", "UNKNOWN", "HUMAN_REVIEW_REQUIRED"),
    ("license_type = UNKNOWN -> UNCATEGORIZED",
     rights(license_type="UNKNOWN"),
     MATCH_OK, NON_COMMERCIAL, "UNCATEGORIZED", "UNKNOWN", "HUMAN_REVIEW_REQUIRED"),

    # --- Dữ liệu dạng chuỗi (đọc từ CSV) ---------------------------------
    ("CSV: boolean dạng chuỗi 'False'",
     rights(license_type="AUDIO_LIBRARY", attribution_required="False"),
     MATCH_OK, NON_COMMERCIAL, "AUDIO_LIBRARY", "LOW", "FREE_TO_USE"),
    ("CSV: boolean dạng chuỗi 'True'",
     rights(license_type="AUDIO_LIBRARY", attribution_required="True"),
     MATCH_OK, NON_COMMERCIAL, "AUDIO_LIBRARY", "CONDITIONAL", "ATTRIBUTION_REQUIRED"),
    ("CSV: recording_public_domain dạng chuỗi",
     rights(license_type="PUBLIC_DOMAIN", recording_public_domain="True",
            composition={"public_domain_status": "verified"}),
     MATCH_OK, COMMERCIAL, "PUBLIC_DOMAIN", "LOW", "FREE_TO_USE"),
    ("CSV: license_type viết thường",
     rights(license_type="audio_library", attribution_required=False),
     MATCH_OK, NON_COMMERCIAL, "AUDIO_LIBRARY", "LOW", "FREE_TO_USE"),
]


@pytest.mark.parametrize(
    "ten,rights_data,match,context,category,risk,condition",
    CASES, ids=[c[0] for c in CASES],
)
def test_rule_engine_cases(ten, rights_data, match, context, category, risk, condition):
    d = evaluate_rights_and_risk(rights_data, match, context)
    assert d["category"] == category, f"{ten}: sai nhóm"
    assert d["risk_level"] == risk, f"{ten}: sai mức rủi ro"
    assert d["condition"] == condition, f"{ten}: sai điều kiện"
    # §2: mọi quyết định đều phải có lý do và evidence
    assert d["decision_reason"]
    assert d["evidence"]["rule_id"]


# ===========================================================================
# 3. Thứ tự ưu tiên là CỐ ĐỊNH (§2)
# ===========================================================================
@pytest.mark.parametrize("license_type,expected_group", [
    ("AUDIO_LIBRARY", "AUDIO_LIBRARY"),
    ("CREATOR_MUSIC", "CREATOR_MUSIC"),
    ("CONTENT_ID", "COMMERCIAL_CONTENT_ID"),
    ("CC_BY", "CREATIVE_COMMONS"),
])
def test_nhom_uu_tien_cao_hon_thang(license_type, expected_group):
    """
    Bản ghi vừa mang copyright_status = PUBLIC_DOMAIN (nhóm 5) vừa có
    license_type thuộc nhóm ưu tiên cao hơn -> nhóm ưu tiên cao hơn phải thắng.
    """
    d = evaluate_rights_and_risk(
        rights(license_type=license_type, copyright_status="PUBLIC_DOMAIN",
               attribution_required=False, policy_action="MONETIZE_CLAIM",
               license_purchased=True, recording_public_domain=True,
               composition={"public_domain_status": "verified"}),
        MATCH_OK, NON_COMMERCIAL,
    )
    assert d["category"] == expected_group


def test_thu_tu_nhom_trong_file_cau_hinh_dung_nhu_de_cuong():
    rules = load_rules()
    assert [g["id"] for g in rules["groups"]] == [
        "AUDIO_LIBRARY", "CREATOR_MUSIC", "COMMERCIAL_CONTENT_ID",
        "CREATIVE_COMMONS", "PUBLIC_DOMAIN",
    ]
    assert [g["order"] for g in rules["groups"]] == [1, 2, 3, 4, 5]


# ===========================================================================
# 4. Ba loại độ tin cậy tách biệt (§2)
# ===========================================================================
def test_ba_do_tin_cay_deu_co_mat_va_tach_biet():
    d = evaluate_rights_and_risk(rights(), MATCH_NEAR, NON_COMMERCIAL)
    assert set(("identity_confidence", "rights_confidence",
                "decision_confidence")) <= set(d)
    # Kiểm rằng identity_confidence được TRUYỀN NGUYÊN qua, không kiểm một con số
    # cụ thể — con số đó là ngưỡng đã hiệu chỉnh và sẽ còn đổi theo dữ liệu.
    assert d["identity_confidence"] == MATCH_NEAR["identity_confidence"]


def test_decision_confidence_khong_cao_hon_mat_xich_yeu_nhat():
    d = evaluate_rights_and_risk(
        rights(source="SIMULATED (MTG-Jamendo)"), MATCH_NEAR, NON_COMMERCIAL
    )
    assert d["decision_confidence"] <= d["identity_confidence"]
    assert d["decision_confidence"] <= d["rights_confidence"]


def test_metadata_mo_phong_bi_tru_diem_tin_cay():
    score_real, reasons_real = compute_rights_confidence(rights(source="MTG-Jamendo"))
    score_sim, reasons_sim = compute_rights_confidence(
        rights(source="SIMULATED (MTG-Jamendo)")
    )
    assert score_sim < score_real
    assert any("mô phỏng" in r for r in reasons_sim)


def test_thieu_license_type_bi_tru_diem():
    score, reasons = compute_rights_confidence(rights(license_type=None))
    assert score < 1.0
    assert any("license_type" in r for r in reasons)


def test_giay_phep_het_han_bi_tru_diem():
    score, reasons = compute_rights_confidence(rights(valid_until=EXPIRED_UNTIL))
    assert any("hết hạn" in r for r in reasons)


def test_khong_co_rights_thi_tin_cay_bang_0():
    score, reasons = compute_rights_confidence(None)
    assert score == 0.0


# ===========================================================================
# 5. Dữ kiện đầu vào của rule
# ===========================================================================
def test_pd_tac_pham_va_pd_ban_thu_lay_tu_hai_nguon_khac_nhau():
    """§2: không được suy PD của bản thu từ PD của tác phẩm."""
    facts = build_facts(
        rights(recording_public_domain=False),
        composition={"public_domain_status": "verified"},
        context=NON_COMMERCIAL,
    )
    assert facts["composition_public_domain"] == "verified"
    assert facts["recording_public_domain"] is False


def test_context_duoc_dua_vao_facts():
    facts = build_facts(rights(), None, COMMERCIAL)
    assert facts["context.commercial_use"] is True
    assert facts["context.monetization"] is True
    assert facts["context.platform"] == "YOUTUBE"


def test_giay_phep_con_han_thi_license_valid_true():
    assert build_facts(rights())["license_valid"] is True
    assert build_facts(rights(valid_until=EXPIRED_UNTIL))["license_valid"] is False


def test_file_rules_nap_duoc_va_du_khoa():
    rules = load_rules()
    assert rules["version"] == 1
    assert "identity_gate" in rules and "uncategorized" in rules
    for group in rules["groups"]:
        assert "match" in group and "rules" in group and "default" in group
        for rule in group["rules"]:
            assert rule["risk"] in ("LOW", "CONDITIONAL", "HIGH", "UNKNOWN")
            assert rule["condition"] and rule["reason"]
