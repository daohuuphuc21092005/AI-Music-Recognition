"""
Rule Engine — phân loại 5 nhóm bản quyền và đánh giá rủi ro sử dụng.

Đây là LOGIC TƯỜNG MINH đọc từ `configs/rules_v1.yaml`, không phải mô hình ML
(CLAUDE.md §9). Những nguyên tắc bắt buộc được cài đặt ở đây:

  * Thứ tự nhóm CỐ ĐỊNH: Audio Library -> Creator Music -> Content ID ->
    Creative Commons -> Public Domain -> Uncategorized. Không xét song song.
  * Cổng định danh chạy TRƯỚC: chưa nhận diện chắc chắn thì không có quyền nào
    để xét, phải trả UNKNOWN (§2).
  * `composition_public_domain` và `recording_public_domain` xét ĐỘC LẬP, không
    suy cái này ra cái kia (§2).
  * Không bao giờ ép UNKNOWN thành LOW/HIGH khi thiếu thông tin (§2).
  * Ba loại độ tin cậy TÁCH BIỆT: identity / rights / decision (§2).
  * Mọi quyết định đều kèm `evidence` + `decision_reason` (§2: không hộp đen).

Bản cũ của file này chưa từng được gọi ở bất kỳ đâu, và thứ tự xét cũng sai:
nó kiểm tra license_type theo thứ tự tuỳ ý và không hề có khái niệm độ tin cậy.
"""
import os
from datetime import date, datetime
from functools import lru_cache

import yaml

from backend import config

RULES_PATH = os.path.join(config.BASE_DIR, "configs", "rules_v1.yaml")

RISK_LEVELS = ("LOW", "CONDITIONAL", "HIGH", "UNKNOWN")


class RulesError(RuntimeError):
    """File rules không hợp lệ."""


@lru_cache(maxsize=1)
def load_rules(path: str = RULES_PATH) -> dict:
    if not os.path.exists(path):
        raise RulesError(f"Không tìm thấy file rules: {path}")
    with open(path, "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    for key in ("identity_gate", "groups", "uncategorized"):
        if key not in rules:
            raise RulesError(f"File rules thiếu khoá bắt buộc: {key}")
    # Bảo đảm thứ tự ưu tiên luôn cố định, không phụ thuộc thứ tự viết trong YAML
    rules["groups"].sort(key=lambda g: g["order"])
    return rules


# --------------------------------------------------------------------------
# Bộ so khớp điều kiện
# --------------------------------------------------------------------------
def _normalize(value):
    """Đưa về dạng so sánh được: bool/None giữ nguyên, còn lại thành chữ thường."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        low = text.lower()
        # CSV trả boolean dưới dạng chuỗi "True"/"False"; PostgreSQL trả bool thật.
        # Lưu ý: "no" KHÔNG được coi là False — đó là một giá trị hợp lệ của
        # composition.public_domain_status (verified/possible/no/unknown).
        if low == "true":
            return True
        if low == "false":
            return False
        return low
    return value


def _matches_value(actual, expected) -> bool:
    actual = _normalize(actual)
    if isinstance(expected, list):
        return any(_matches_value(actual, item) for item in expected)
    return actual == _normalize(expected)


def _matches_conditions(conditions: dict, facts: dict) -> bool:
    """Tất cả điều kiện phải đúng (AND). Trường không có trong facts coi như None."""
    for field, expected in conditions.items():
        if not _matches_value(facts.get(field), expected):
            return False
    return True


def _group_matches(group: dict, facts: dict) -> bool:
    match = group.get("match", {})
    if "any_of" in match:
        return any(_matches_value(facts.get(field), expected)
                   for field, expected in match["any_of"].items())
    return _matches_conditions(match, facts)


# --------------------------------------------------------------------------
# Chuẩn bị facts
# --------------------------------------------------------------------------
def _license_is_valid(rights: dict, today: date = None) -> bool:
    """Giấy phép còn hiệu lực tại thời điểm xét."""
    today = today or date.today()

    def parse(value):
        if not value:
            return None
        if isinstance(value, date):
            return value
        try:
            return datetime.fromisoformat(str(value)[:10]).date()
        except ValueError:
            return None

    valid_from = parse(rights.get("valid_from"))
    valid_until = parse(rights.get("valid_until"))
    if valid_from and today < valid_from:
        return False
    if valid_until and today > valid_until:
        return False
    return True


def build_facts(rights: dict, composition: dict = None, context: dict = None) -> dict:
    """Gom mọi dữ kiện mà rule có thể tham chiếu vào một từ điển phẳng."""
    rights = rights or {}
    composition = composition or {}
    context = context or {}

    facts = dict(rights)
    facts["license_valid"] = _license_is_valid(rights)
    # PD của TÁC PHẨM lấy từ bảng compositions; PD của BẢN THU lấy từ bảng rights.
    # Hai nguồn khác nhau, xét độc lập (§2).
    facts["composition_public_domain"] = composition.get("public_domain_status")
    facts["recording_public_domain"] = rights.get("recording_public_domain")

    for key in ("platform", "commercial_use", "monetization"):
        facts[f"context.{key}"] = context.get(key)
    return facts


def rights_confidence_breakdown(rights: dict, rules: dict = None) -> dict:
    """
    Độ tin cậy của DỮ LIỆU QUYỀN — tách hẳn khỏi độ tin cậy nhận diện (§2) —
    kèm từng khoản trừ và mức trừ lấy từ rules.

    Trả cả phép tính chứ không chỉ kết quả: mọi bản ghi PREDICTED chưa xác minh
    đều ra đúng 0.20, và nếu chỉ hiện con số đó thì người đọc không phân biệt
    được "phép tính cho ra 0.20" với "một hằng số mặc định".
    """
    rules = rules or load_rules()
    cfg = rules.get("rights_confidence", {})
    penalties = cfg.get("penalties", {})
    base = float(cfg.get("base", 1.0))

    if not rights:
        return {"base": base, "penalties": [], "final": 0.0,
                "formula": "không có dữ liệu quyền -> 0.00"}

    applied = []

    def penalize(code: str, reason: str, default: float = 0.0) -> None:
        applied.append({"code": code,
                        "amount": float(penalties.get(code, default)),
                        "reason": reason})

    if not rights.get("license_type") or rights.get("license_type") == "UNKNOWN":
        penalize("missing_license_type", "thiếu license_type")
    if not rights.get("copyright_status"):
        penalize("missing_copyright_status", "thiếu copyright_status")

    source = str(rights.get("source") or "")
    if source.upper().startswith("PREDICTED"):
        # Quyền do model suy đoán từ âm thanh, không tra từ nguồn nào. Phạt nặng
        # hơn metadata mô phỏng vì EXP-09 đo được bộ phân loại này không vượt
        # baseline ở đúng tình huống nó được dùng (bài của nghệ sĩ chưa từng thấy).
        penalize("predicted_source",
                 "quyền do model suy đoán từ âm thanh, chưa tra từ nguồn nào", 0.60)
    elif source.upper().startswith("SIMULATED"):
        penalize("simulated_source", "metadata mô phỏng, chưa xác minh từ nguồn thật")

    verified_at = rights.get("verified_at")
    if not verified_at:
        penalize("unverified", "thiếu verified_at")
    else:
        try:
            verified = datetime.fromisoformat(str(verified_at)[:19])
            age_days = (datetime.now() - verified).days
            if age_days > penalties.get("stale_verification_days", 365):
                penalize("stale_penalty",
                         f"metadata đã xác minh cách đây {age_days} ngày")
        except ValueError:
            pass

    if not _license_is_valid(rights):
        penalize("license_expired", "giấy phép hết hạn hoặc chưa hiệu lực")

    raw = base - sum(p["amount"] for p in applied)
    final = max(0.0, min(1.0, raw))
    terms = "".join(f" − {p['amount']:.2f} ({p['reason']})" for p in applied)
    clamped = " (kẹp về khoảng [0, 1])" if abs(raw - final) > 1e-9 else ""
    return {
        "base": base,
        "penalties": applied,
        "final": round(final, 4),
        "formula": f"{base:.2f}{terms} = {final:.2f}{clamped}",
    }


def _summarize_rights_confidence(rights: dict, breakdown: dict) -> tuple:
    if not rights:
        return 0.0, ["không có dữ liệu quyền"]
    reasons = [f"{p['reason']} (−{p['amount']:.2f})" for p in breakdown["penalties"]]
    return breakdown["final"], reasons


def compute_rights_confidence(rights: dict, rules: dict = None) -> tuple:
    """Trả (điểm, danh sách lý do trừ điểm kèm mức trừ)."""
    return _summarize_rights_confidence(rights, rights_confidence_breakdown(rights, rules))


# --------------------------------------------------------------------------
# Điểm vào chính
# --------------------------------------------------------------------------
def evaluate_rights_and_risk(rights_data: dict, match_info: dict,
                             usage_context: dict = None) -> dict:
    """
    rights_data   : bản ghi rights (kèm khoá 'composition' nếu có)
    match_info    : kết quả nhận diện — cần 'match_type' và 'identity_confidence'
    usage_context : {'platform', 'commercial_use', 'monetization'}

    Trả về quyết định đầy đủ kèm evidence và ba loại độ tin cậy.
    """
    rules = load_rules()
    gate = rules["identity_gate"]
    context = usage_context or {}
    match_info = match_info or {}

    match_type = match_info.get("match_type")
    identity_confidence = float(match_info.get("identity_confidence") or 0.0)
    # Ngưỡng định danh tối thiểu lấy THEO TẦNG đã sinh ra kết quả này: điểm
    # Chromaprint và cosine similarity của MERT không cùng thang đo, dùng chung
    # một con số sẽ vô hiệu hoá ngưỡng đã hiệu chỉnh của một trong hai tầng.
    by_match_type = gate.get("min_identity_confidence_by_match_type") or {}
    min_identity = float(
        by_match_type.get(match_type, gate.get("min_identity_confidence", 0.0))
    )

    def decision(outcome: dict, category: str, rule_id: str,
                 rights_conf: float, rights_reasons: list,
                 group_order=None, extra_evidence: dict = None) -> dict:
        risk = outcome["risk"]
        if risk not in RISK_LEVELS:
            raise RulesError(f"Mức rủi ro không hợp lệ trong rules: {risk}")
        # Quyết định không thể chắc hơn mắt xích yếu nhất của nó
        decision_confidence = 0.0 if risk == "UNKNOWN" else round(
            min(identity_confidence, rights_conf), 4
        )
        decision_formula = (
            "0.0000 — mức rủi ro UNKNOWN: chưa có kết luận nào để gán độ tin cậy"
            if risk == "UNKNOWN" else
            f"min(nhận diện {identity_confidence:.4f}, dữ liệu quyền {rights_conf:.4f}) "
            f"= {decision_confidence:.4f}"
        )
        return {
            "category": category,
            "risk_level": risk,
            "condition": outcome["condition"],
            "decision_reason": " ".join(str(outcome.get("reason", "")).split()),
            "identity_confidence": round(identity_confidence, 4),
            "rights_confidence": round(rights_conf, 4),
            "decision_confidence": decision_confidence,
            "evidence": {
                "rules_version": rules.get("name", "rules_v1"),
                "rule_id": rule_id,
                "group_order": group_order,
                "match_type": match_type,
                "min_identity_confidence": min_identity,
                "rights_confidence_penalties": rights_reasons,
                "rights_confidence_breakdown": rights_breakdown,
                "decision_confidence_formula": decision_formula,
                "usage_context": {
                    "platform": context.get("platform"),
                    "commercial_use": context.get("commercial_use"),
                    "monetization": context.get("monetization"),
                },
                **(extra_evidence or {}),
            },
        }

    rights_breakdown = rights_confidence_breakdown(rights_data, rules)
    rights_conf, rights_reasons = _summarize_rights_confidence(rights_data, rights_breakdown)

    # --- Cổng định danh (chạy trước mọi nhóm) ------------------------------
    if match_type in (None, "NO_MATCH") and not rights_data:
        outcome = gate["no_match"]
        return decision(outcome, outcome["category"], "identity_gate.no_match",
                        rights_conf, rights_reasons)

    if match_type == "UNKNOWN" or identity_confidence < min_identity:
        outcome = gate["low_confidence"]
        return decision(
            outcome, outcome["category"], "identity_gate.low_confidence",
            rights_conf, rights_reasons,
            extra_evidence={"identity_shortfall": round(
                min_identity - identity_confidence, 4)},
        )

    if not rights_data:
        outcome = gate["missing_rights"]
        return decision(outcome, outcome["category"],
                        "identity_gate.missing_rights", rights_conf, rights_reasons)

    # --- 5 nhóm, đúng thứ tự ưu tiên ---------------------------------------
    facts = build_facts(rights_data, rights_data.get("composition"), context)

    rights_gate = rules.get("rights_gate") or {}
    min_rights = rights_gate.get("min_rights_confidence")

    def gate_on_rights(provisional: dict) -> dict:
        """
        Kết luận của 5 nhóm chỉ đứng được khi dữ liệu quyền đủ tin cậy.

        Giấy phép do model đoán vẫn đi qua đúng nhánh và có thể ra HIGH — nhưng đó
        là HIGH của một phỏng đoán. Hiện nó như kết luận là "kết luận từ kết quả
        nghi ngờ" mà §2 cấm, nên hạ về UNKNOWN và giữ kết luận tạm trong evidence.
        """
        if (min_rights is None or provisional["risk_level"] == "UNKNOWN"
                or rights_conf >= float(min_rights)):
            return provisional
        outcome = rights_gate["low_rights_confidence"]
        provisional_evidence = provisional["evidence"]
        return decision(
            outcome, outcome["category"], "rights_gate.low_rights_confidence",
            rights_conf, rights_reasons,
            group_order=provisional_evidence.get("group_order"),
            extra_evidence={
                "min_rights_confidence": float(min_rights),
                "rights_shortfall": round(float(min_rights) - rights_conf, 4),
                "provisional_decision": {
                    "category": provisional["category"],
                    "risk_level": provisional["risk_level"],
                    "condition": provisional["condition"],
                    "rule_id": provisional_evidence.get("rule_id"),
                    "matched_conditions": provisional_evidence.get("matched_conditions"),
                    "reason": provisional["decision_reason"],
                },
            },
        )

    for group in rules["groups"]:
        if not _group_matches(group, facts):
            continue

        for index, rule in enumerate(group.get("rules", [])):
            if _matches_conditions(rule.get("when", {}), facts):
                return gate_on_rights(decision(
                    rule, group["id"], f"{group['id']}.rule[{index}]",
                    rights_conf, rights_reasons, group_order=group["order"],
                    extra_evidence={"matched_conditions": rule.get("when", {})},
                ))

        return gate_on_rights(decision(
            group["default"], group["id"], f"{group['id']}.default",
            rights_conf, rights_reasons, group_order=group["order"]))

    outcome = rules["uncategorized"]
    return decision(outcome, outcome["category"], "uncategorized",
                    rights_conf, rights_reasons,
                    extra_evidence={"license_type": rights_data.get("license_type")})
