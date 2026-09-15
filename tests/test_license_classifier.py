"""
Test cho bộ phân loại giấy phép và nhánh quyền SUY ĐOÁN (§9, LICENSE_PREDICTED).

Bộ phân loại này được tích hợp theo yêu cầu của chủ dự án SAU KHI EXP-09 đo được
rằng nó không vượt baseline ở đúng tình huống nó chạy. Chính vì vậy phần lớn test
ở đây không kiểm "model đoán đúng không" — điều đó đã có EXP-09 trả lời — mà kiểm
những thứ phải luôn đúng bất kể model tốt hay tệ:

  * Dự đoán không bao giờ bị trình bày như quyền tra được từ nguồn thật.
  * Nguồn PREDICTED bị trừ điểm nặng hơn metadata mô phỏng.
  * decision_confidence luôn bị kéo xuống theo mắt xích yếu nhất.
  * Mã giấy phép lạ thì trả UNKNOWN chứ không đoán bừa (§2).
"""
import numpy as np
import pytest

from backend.services import license_classifier_service as classifier
from backend.services.decision_service import (
    compute_rights_confidence,
    evaluate_rights_and_risk,
    load_rules,
)
from backend.services.license_mapping import rights_for_license_type

NON_COMMERCIAL = {"platform": "YOUTUBE", "commercial_use": False, "monetization": False}
COMMERCIAL = {"platform": "YOUTUBE", "commercial_use": True, "monetization": True}


def predicted_rights(license_type: str, probability: float = 0.26) -> dict:
    """Dựng đúng khối rights mà analysis_pipeline sinh ra cho bài ngoài CSDL."""
    flags = rights_for_license_type(license_type)
    return {
        **flags,
        "source": f"PREDICTED bởi license_clf_v1 (xác suất {probability})",
        "source_url": None,
        "verified_at": None,
        "revenue_share_required": False,
        "policy_action": "NONE",
        "license_purchased": False,
        "revenue_share_agreed": False,
        "composition": {"public_domain_status": "unknown"},
    }


# ---------------------------------------------------------------------------
# Ánh xạ mã taxonomy -> cờ quyền
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("license_type,commercial,modification,attribution", [
    ("CC_BY", True, True, True),
    ("CC_BY_SA", True, True, True),
    ("CC_BY_NC", False, True, True),
    ("CC_BY_NC_SA", False, True, True),
    ("CC_BY_ND", True, False, True),
    ("CC_BY_NC_ND", False, False, True),
    ("CC0", True, True, False),
])
def test_co_quyen_suy_ra_dung_tu_ma(license_type, commercial, modification, attribution):
    flags = rights_for_license_type(license_type)
    assert flags["license_type"] == license_type
    assert flags["commercial_use_allowed"] is commercial
    assert flags["modification_allowed"] is modification
    assert flags["attribution_required"] is attribution
    # Điều khoản NC chi phối cả quyền kiếm tiền
    assert flags["monetization_allowed"] is commercial


@pytest.mark.parametrize("bad", ["", None, "KHONG_TON_TAI", "MP3", "CC_BY_NC_SA_ND"])
def test_ma_la_tra_unknown_khong_doan_bua(bad):
    """§2: không nhận diện được thì trả UNKNOWN với mọi cờ để None."""
    flags = rights_for_license_type(bad)
    assert flags["license_type"] == "UNKNOWN"
    assert flags["attribution_required"] is None
    assert flags["commercial_use_allowed"] is None


# ---------------------------------------------------------------------------
# Nguồn PREDICTED bị phạt nặng
# ---------------------------------------------------------------------------
def test_nguon_predicted_bi_phat_nang_hon_simulated():
    base = rights_for_license_type("CC_BY")
    verified = "2026-01-01T00:00:00"

    real = {**base, "source": "FMA (giấy phép do nguồn công bố)", "verified_at": verified}
    simulated = {**base, "source": "SIMULATED (MTG-Jamendo)", "verified_at": verified}
    predicted = {**base, "source": "PREDICTED bởi license_clf_v1", "verified_at": verified}

    real_score, _ = compute_rights_confidence(real)
    sim_score, _ = compute_rights_confidence(simulated)
    pred_score, pred_reasons = compute_rights_confidence(predicted)

    assert real_score > sim_score > pred_score
    assert any("suy đoán" in reason for reason in pred_reasons)


def test_predicted_source_khai_bao_trong_rules():
    """Mức phạt phải nằm trong file cấu hình, không phải hằng số rải trong code."""
    penalties = load_rules()["rights_confidence"]["penalties"]
    assert "predicted_source" in penalties
    assert penalties["predicted_source"] > penalties["simulated_source"]


# ---------------------------------------------------------------------------
# Nhánh LICENSE_PREDICTED trong Rule Engine
# ---------------------------------------------------------------------------
def test_cong_identity_co_khai_bao_license_predicted():
    gate = load_rules()["identity_gate"]["min_identity_confidence_by_match_type"]
    assert "LICENSE_PREDICTED" in gate, (
        "Thiếu khai báo thì match_type này rơi về ngưỡng mặc định và nhánh suy "
        "đoán sẽ bị chặn im lặng."
    )


@pytest.mark.parametrize("license_type,context,expected_risk", [
    ("CC0", COMMERCIAL, "LOW"),
    ("CC_BY", COMMERCIAL, "CONDITIONAL"),
    ("CC_BY", NON_COMMERCIAL, "CONDITIONAL"),
    # NC + mục đích thương mại là vi phạm điều khoản, kể cả khi quyền là suy đoán
    ("CC_BY_NC_SA", COMMERCIAL, "HIGH"),
    ("CC_BY_NC_SA", NON_COMMERCIAL, "CONDITIONAL"),
])
def test_quyen_suy_doan_van_chay_qua_dung_nhanh(license_type, context, expected_risk):
    decision = evaluate_rights_and_risk(
        predicted_rights(license_type),
        {"match_type": "LICENSE_PREDICTED", "identity_confidence": 0.72},
        context,
    )
    assert decision["risk_level"] == expected_risk


def test_do_tin_cay_quyet_dinh_luon_thap_voi_quyen_suy_doan():
    """
    Dù mức rủi ro có là gì, decision_confidence phải thấp: nó bằng mắt xích yếu
    nhất, mà rights_confidence của nguồn suy đoán đã bị trừ rất nặng. Đây là thứ
    ngăn một dự đoán không vượt baseline được đọc như kết luận chắc chắn.
    """
    decision = evaluate_rights_and_risk(
        predicted_rights("CC_BY_NC_SA"),
        {"match_type": "LICENSE_PREDICTED", "identity_confidence": 0.99},
        COMMERCIAL,
    )
    assert decision["rights_confidence"] < 0.5
    assert decision["decision_confidence"] <= decision["rights_confidence"]
    assert any("suy đoán" in reason
               for reason in decision["evidence"]["rights_confidence_penalties"])


def test_unknown_that_van_bi_chan_nhu_cu():
    """
    Nhánh suy đoán KHÔNG được nới lỏng cổng cho match_type UNKNOWN thật. Bài
    không nhận ra và cũng không có dự đoán thì vẫn phải là UNKNOWN (§2).
    """
    decision = evaluate_rights_and_risk(
        predicted_rights("CC0"),
        {"match_type": "UNKNOWN", "identity_confidence": 0.72},
        NON_COMMERCIAL,
    )
    assert decision["risk_level"] == "UNKNOWN"
    assert decision["condition"] == "HUMAN_REVIEW_REQUIRED"


# ---------------------------------------------------------------------------
# Service dự đoán
# ---------------------------------------------------------------------------
def test_predict_tu_choi_dau_vao_khong_hop_le():
    assert classifier.predict(None) is None
    assert classifier.predict(np.zeros(7, dtype="float32")) is None


@pytest.mark.skipif(not classifier.is_available(),
                    reason="chưa train model: scripts/train_license_classifier.py")
def test_predict_luon_kem_canh_bao_va_ket_qua_kiem_dinh():
    """
    Con số dự đoán không được phép tách rời khỏi giới hạn của nó. Đây là điều
    kiện để người đọc kết quả không hiểu nhầm nó là một kết luận về bản quyền.
    """
    vector = np.random.RandomState(0).randn(768).astype("float32")
    vector /= np.linalg.norm(vector)
    result = classifier.predict(vector)

    assert result is not None
    assert result["advisory_only"] is True
    assert result["warning"]
    assert result["validation"]["beats_baseline"] is False
    assert result["validation"]["accuracy"] < result["validation"]["baseline_accuracy"]
    assert 0.0 <= result["probability"] <= 1.0
    assert len(result["top_n"]) >= 1
