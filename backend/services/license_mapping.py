"""
Ánh xạ chuỗi giấy phép THẬT từ nguồn mở sang taxonomy của Rule Engine (§9).

Đây là chỗ đóng góp C3 trả cổ tức. Cho tới nay 100% bản ghi trong reference
database có `source` bắt đầu bằng `SIMULATED`, nghĩa là toàn bộ nhánh "Rights
Metadata" của framework chạy trên dữ liệu bịa. FMA và Jamendo đều công bố giấy
phép Creative Commons thật cho từng bản thu — module này biến chuỗi đó thành các
trường mà `configs/rules_v1.yaml` biết đọc.

Nguyên tắc:

  * Các cờ quyền được SUY RA TỪ CHÍNH điều khoản CC, không phải đoán: `nc` trong
    mã giấy phép nghĩa là cấm dùng thương mại, `nd` nghĩa là cấm phái sinh, mọi
    biến thể `by` đều bắt buộc ghi công. Đây là suy diễn xác định, kiểm chứng
    được bằng văn bản giấy phép.
  * Không nhận diện được thì trả `UNKNOWN` với mọi cờ để `None` — tuyệt đối
    không suy bừa (§2). Rule Engine gặp trạng thái đó sẽ tự trả UNKNOWN +
    HUMAN_REVIEW_REQUIRED, đúng như thiết kế.
  * `license_type` chỉ nhận các giá trị mà `rules_v1.yaml` thật sự khai báo.
    Sinh ra một mã mới ở đây là tạo ra bản ghi không rule nào chạm tới.

Ba nhóm AUDIO_LIBRARY / CREATOR_MUSIC / CONTENT_ID **không lấy được từ nguồn mở**
và không thuộc phạm vi module này — chúng vẫn là metadata mô phỏng và phải giữ
tiền tố `SIMULATED` ở trường `source` để `compute_rights_confidence` trừ điểm đúng.
"""
import re

# Đúng các giá trị mà configs/rules_v1.yaml khai báo, không hơn.
CC_LICENSE_TYPES = {
    "by": "CC_BY",
    "by-sa": "CC_BY_SA",
    "by-nc": "CC_BY_NC",
    "by-nc-sa": "CC_BY_NC_SA",
    "by-nd": "CC_BY_ND",
    "by-nc-nd": "CC_BY_NC_ND",
}

UNKNOWN_RESULT = {
    "license_type": "UNKNOWN",
    "license_code": None,
    "copyright_status": "UNKNOWN",
    "attribution_required": None,
    "commercial_use_allowed": None,
    "modification_allowed": None,
    "monetization_allowed": None,
    "recording_public_domain": None,
    "matched_on": None,
}

_URL_LICENSE = re.compile(r"creativecommons\.org/licenses/([a-z][a-z-]*)", re.I)
_URL_ZERO = re.compile(r"creativecommons\.org/publicdomain/zero", re.I)
_URL_MARK = re.compile(r"creativecommons\.org/publicdomain/mark", re.I)


def _canonical(code: str) -> str:
    """Sắp mã CC về thứ tự chuẩn by-nc-nd, vì nguồn ghi thứ tự lung tung."""
    parts = {p for p in code.lower().split("-") if p}
    if "by" not in parts:
        return ""
    ordered = ["by"]
    for flag in ("nc", "sa", "nd"):
        if flag in parts:
            ordered.append(flag)
    # sa và nd loại trừ nhau theo định nghĩa của CC; gặp cả hai là dữ liệu hỏng
    if "sa" in parts and "nd" in parts:
        return ""
    return "-".join(ordered)


def _from_url(url: str) -> str:
    if not url:
        return ""
    if _URL_ZERO.search(url):
        return "cc0"
    if _URL_MARK.search(url):
        return "pdm"
    match = _URL_LICENSE.search(url)
    return _canonical(match.group(1)) if match else ""


def _from_text(text: str) -> str:
    """
    Đọc mã từ tên giấy phép dạng chữ, ví dụ FMA ghi
    "Attribution-NonCommercial-ShareAlike 3.0 International".
    """
    if not text:
        return ""
    low = re.sub(r"[^a-z0-9]+", " ", text.lower())

    if "cc0" in low or ("public domain" in low and "dedication" in low) or "zero" in low:
        return "cc0"
    if "public domain" in low:
        return "pdm"
    if "attribution" not in low:
        return ""

    parts = ["by"]
    if "noncommercial" in low or "non commercial" in low:
        parts.append("nc")
    if "sharealike" in low or "share alike" in low:
        parts.append("sa")
    if "noderiv" in low or "no deriv" in low:
        parts.append("nd")
    return _canonical("-".join(parts))


def normalize_license(license_text: str = None, license_url: str = None) -> dict:
    """
    Trả các trường quyền suy ra từ giấy phép thật.

    Ưu tiên URL vì đó là dạng máy đọc được và không mơ hồ; tên dạng chữ chỉ dùng
    khi không có URL. `matched_on` cho biết kết luận đến từ đâu, để lần ngược được.
    """
    code = _from_url(license_url)
    matched_on = "url" if code else None
    if not code:
        code = _from_text(license_text)
        matched_on = "text" if code else None

    if not code:
        return dict(UNKNOWN_RESULT)

    if code == "cc0":
        return {
            "license_type": "CC0",
            "license_code": "cc0",
            "copyright_status": "PUBLIC_DOMAIN",
            # CC0 là từ bỏ quyền, nên không đòi ghi công và không cấm gì cả
            "attribution_required": False,
            "commercial_use_allowed": True,
            "modification_allowed": True,
            "monetization_allowed": True,
            "recording_public_domain": True,
            "matched_on": matched_on,
        }

    if code == "pdm":
        return {
            "license_type": "PUBLIC_DOMAIN",
            "license_code": "pdm",
            "copyright_status": "PUBLIC_DOMAIN",
            "attribution_required": False,
            "commercial_use_allowed": True,
            "modification_allowed": True,
            "monetization_allowed": True,
            "recording_public_domain": True,
            "matched_on": matched_on,
        }

    parts = set(code.split("-"))
    commercial = "nc" not in parts
    return {
        "license_type": CC_LICENSE_TYPES[code],
        "license_code": code,
        # Giấy phép CC KHÔNG làm tác phẩm hết bản quyền — nó chỉ cấp phép sử dụng.
        "copyright_status": "PROTECTED",
        "attribution_required": True,          # mọi biến thể BY đều buộc ghi công
        "commercial_use_allowed": commercial,
        "modification_allowed": "nd" not in parts,
        # Bật kiếm tiền trên nội dung là một hình thức sử dụng thương mại, nên
        # điều khoản NC chi phối luôn cả trường này.
        "monetization_allowed": commercial,
        "recording_public_domain": False,
        "matched_on": matched_on,
    }


# Bảng đảo của CC_LICENSE_TYPES, để đi ngược từ mã taxonomy về mã CC gốc.
_TYPE_TO_CODE = {value: key for key, value in CC_LICENSE_TYPES.items()}


def rights_for_license_type(license_type: str) -> dict:
    """
    Dựng bộ cờ quyền từ MỘT MÃ TAXONOMY (vd. "CC_BY_NC_SA") thay vì từ chuỗi
    giấy phép của nguồn.

    Dùng cho đầu ra của bộ phân loại giấy phép: nó trả về mã taxonomy chứ không
    trả URL hay tên giấy phép. Đi vòng qua `normalize_license` để các cờ quyền
    vẫn được suy ra từ ĐÚNG một chỗ duy nhất — nếu sau này sửa cách diễn giải
    điều khoản NC/ND thì cả hai đường vào cùng đổi theo, không lệch nhau.

    Trả bản sao của UNKNOWN_RESULT nếu mã không nhận diện được (§2: không đoán).
    """
    if not license_type:
        return dict(UNKNOWN_RESULT)

    upper = str(license_type).strip().upper()
    if upper == "CC0":
        return normalize_license(
            license_url="https://creativecommons.org/publicdomain/zero/1.0/")
    if upper == "PUBLIC_DOMAIN":
        return normalize_license(
            license_url="https://creativecommons.org/publicdomain/mark/1.0/")

    code = _TYPE_TO_CODE.get(upper)
    if not code:
        return dict(UNKNOWN_RESULT)
    return normalize_license(
        license_url=f"https://creativecommons.org/licenses/{code}/4.0/")
