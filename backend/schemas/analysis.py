"""
Schema request/response cho API (CLAUDE.md §11).

Tách schema ra khỏi tầng service để hợp đồng API được khai báo tường minh và
Swagger UI sinh tài liệu đúng.
"""
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Platform(str, Enum):
    YOUTUBE = "YOUTUBE"
    FACEBOOK = "FACEBOOK"
    TIKTOK = "TIKTOK"
    OTHER = "OTHER"


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


class MatchType(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    NEAR_MATCH = "NEAR_MATCH"
    UNKNOWN = "UNKNOWN"


class RiskLevel(str, Enum):
    LOW = "LOW"
    CONDITIONAL = "CONDITIONAL"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class FeedbackVerdict(str, Enum):
    CORRECT = "Correct"
    INCORRECT = "Incorrect"
    UNSURE = "Unsure"


class UsageContext(BaseModel):
    """Mục đích sử dụng do người dùng khai báo — đầu vào của Rule Engine."""
    platform: Platform = Field(
        Platform.YOUTUBE,
        description="Nền tảng dự định đăng nội dung. Ảnh hưởng tới điều kiện/khuyến nghị "
                    "trả về (ví dụ chính sách Content ID chỉ áp dụng khi platform=YOUTUBE).",
    )
    commercial_use: bool = Field(
        False, description="Có dùng cho mục đích thương mại hay không. Đầu vào trực tiếp "
                            "cho các rule kiểm tra commercial_use_allowed/CC_BY_NC.",
    )
    monetization: bool = Field(
        False, description="Có dự định bật kiếm tiền trên nội dung hay không. Đầu vào cho "
                            "các rule kiểm tra monetization_allowed/revenue_share.",
    )


class AnalyzeAccepted(BaseModel):
    """Phản hồi ngay khi `POST /analyze` nhận file — tác vụ chạy nền, chưa có kết quả."""
    job_id: str = Field(..., description="UUID của tác vụ phân tích, dùng để poll "
                                          "GET /jobs/{job_id} và GET /results/{job_id}.")
    status: JobStatus = Field(..., description="Trạng thái tác vụ ngay lúc nhận — luôn là QUEUED.")
    message: str = Field(..., description="Thông báo ngắn cho người dùng (không phải lỗi).")


class PipelineStage(str, Enum):
    """Bước đang chạy trong pipeline — màn hình Processing hiển thị đúng bước này."""
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    EXTRACTING_AUDIO = "EXTRACTING_AUDIO"
    FINGERPRINTING = "FINGERPRINTING"
    EMBEDDING = "EMBEDDING"
    VECTOR_SEARCH = "VECTOR_SEARCH"
    RIGHTS_LOOKUP = "RIGHTS_LOOKUP"
    RULE_ENGINE = "RULE_ENGINE"
    DONE = "DONE"


class JobStatusResponse(BaseModel):
    """Trạng thái tiến độ của một tác vụ — `GET /jobs/{job_id}` (§11)."""
    job_id: str = Field(..., description="UUID của tác vụ.")
    status: JobStatus = Field(..., description="QUEUED | PROCESSING | DONE | FAILED.")
    stage: Optional[PipelineStage] = Field(
        None, description="Bước cụ thể đang chạy trong cascade (chỉ có giá trị khi "
                           "status=PROCESSING). Dùng để hiển thị tiến trình thật, không phải "
                           "thông báo mơ hồ kiểu 'AI is thinking'.",
    )
    filename: Optional[str] = Field(None, description="Tên file gốc người dùng đã tải lên.")
    error_code: Optional[str] = Field(
        None, description="Mã lỗi chuẩn hoá nếu status=FAILED (FILE_TOO_LARGE, "
                           "UNSUPPORTED_FORMAT, NO_AUDIO, NO_MUSIC, MODEL_FAILURE, "
                           "DATABASE_FAILURE, TIMEOUT, UNKNOWN_TRACK, LOW_CONFIDENCE). "
                           "Không bao giờ chứa traceback nội bộ.",
    )
    error_message: Optional[str] = Field(None, description="Mô tả lỗi dễ hiểu cho người dùng "
                                                             "(ứng với error_code).")
    created_at: Optional[str] = Field(None, description="Thời điểm tạo tác vụ (ISO 8601).")
    updated_at: Optional[str] = Field(None, description="Thời điểm cập nhật trạng thái gần nhất.")


class Identity(BaseModel):
    """Thông tin nhận diện bài hát — None ở field nào nghĩa là chưa xác định được field đó."""
    track: Optional[str] = Field(None, description="Tên bản ghi (recordings.title).")
    artist: Optional[str] = Field(None, description="Nghệ sĩ/ban nhạc biểu diễn.")
    recording_id: Optional[str] = Field(
        None, description="UUID bản ghi cụ thể đã khớp (quyền bản ghi/phát sóng — khác "
                           "composition_id). None nếu match.type=UNKNOWN.",
    )
    composition_id: Optional[str] = Field(
        None, description="UUID tác phẩm gốc (quyền tác giả/giai điệu) mà recording_id thuộc "
                           "về. Composition ID ≠ Recording ID (§2): một composition có thể có "
                           "nhiều recording, và có thể xác định được composition mà vẫn không "
                           "xác định được recording.",
    )


class Match(BaseModel):
    """Kết quả tầng nhận diện âm thanh (cascade Chromaprint -> MERT -> Cover)."""
    type: MatchType = Field(
        ..., description="EXACT_MATCH (khớp fingerprint Chromaprint) | NEAR_MATCH (khớp "
                          "similarity MERT/Cover đạt ngưỡng) | UNKNOWN (không đủ bằng chứng — "
                          "không bao giờ bị ép thành LOW/HIGH ở tầng risk, §2).",
    )
    confidence: float = Field(
        ..., description="identity_confidence của tầng nhận diện — điểm fingerprint hoặc "
                          "cosine similarity thật, KHÔNG bị pha trộn với độ tin cậy về quyền "
                          "sử dụng (rights_confidence tách biệt, xem Assessment).",
    )
    pipeline_stage: Optional[str] = Field(
        None, description="Tầng nào trong cascade trả về kết quả này: STAGE_1_CHROMAPRINT | "
                           "STAGE_2_MERT_RETRIEVAL | STAGE_3_COVER.",
    )


class Rights(BaseModel):
    """Thông tin pháp lý/giấy phép tra được từ bảng `rights` ứng với recording_id đã khớp."""
    license: Optional[str] = Field(
        None, description="Loại giấy phép (CC_BY, CC_BY_NC, PUBLIC_DOMAIN, "
                           "YOUTUBE_AUDIO_LIBRARY, CREATOR_MUSIC...). None nếu rights_found=False.",
    )
    copyright_status: Optional[str] = Field(None, description="PROTECTED | PUBLIC_DOMAIN | UNKNOWN.")
    attribution_required: Optional[bool] = Field(None, description="Có bắt buộc ghi công tác giả không.")
    commercial_use_allowed: Optional[bool] = Field(None, description="Có cho phép mục đích thương mại không.")
    monetization_allowed: Optional[bool] = Field(None, description="Có cho phép bật kiếm tiền trên nền tảng không.")
    source: Optional[str] = Field(None, description="Nguồn/đơn vị công bố thông tin bản quyền.")
    source_url: Optional[str] = Field(None, description="Đường dẫn xác thực thông tin giấy phép.")
    verified_at: Optional[str] = Field(None, description="Thời điểm thông tin quyền được xác minh lần cuối.")
    rights_found: bool = Field(
        False, description="Có dữ liệu quyền để đưa vào Rule Engine hay không — tra từ CSDL "
                            "hoặc suy đoán (xem `predicted`). False nghĩa là mọi field phía "
                            "trên đều None — không nhầm với việc quyền = 'không có ai sở hữu'.",
    )
    predicted: bool = Field(
        False, description="True nếu dữ liệu quyền do license classifier SUY ĐOÁN từ âm thanh "
                            "(bài không khớp bản ghi nào), không tra từ nguồn nào. Khi đó "
                            "rights_confidence bị trừ nặng, cổng dữ liệu quyền hạ risk về "
                            "UNKNOWN, và kết luận tạm nằm ở "
                            "evidence.rule_engine.provisional_decision.",
    )


class Assessment(BaseModel):
    """Kết quả Rule Engine — cây quyết định tuần tự 5 nhóm bản quyền + fallback (§6)."""
    risk: RiskLevel = Field(
        ..., description="LOW | CONDITIONAL | HIGH | UNKNOWN. UNKNOWN không bao giờ bị ép "
                          "thành LOW hay HIGH khi thiếu bằng chứng (§2) — trường hợp nghi ngờ "
                          "luôn trả UNKNOWN kèm condition=HUMAN_REVIEW_REQUIRED.",
    )
    category: Optional[str] = Field(
        None, description="Một trong 5 nhóm bản quyền (AUDIO_LIBRARY, CREATOR_MUSIC, "
                           "COMMERCIAL_CONTENT_ID, CREATIVE_COMMONS, PUBLIC_DOMAIN) hoặc nhóm "
                           "fallback (UNCATEGORIZED, USER_GENERATED_CONTENT).",
    )
    condition: Optional[str] = Field(
        None, description="Mã điều kiện sử dụng cụ thể (vd ATTRIBUTION_REQUIRED, "
                           "REVENUE_SHARE_APPLIED, HUMAN_REVIEW_REQUIRED) — xem "
                           "configs/rules_v1.yaml để biết đầy đủ danh sách.",
    )
    reason: Optional[str] = Field(None, description="Giải thích bằng văn xuôi vì sao Rule Engine "
                                                      "đi tới quyết định này — không hộp đen (§2).")
    # Ba loại độ tin cậy TÁCH BIỆT (§2) — không gộp thành một "copyright probability"
    identity_confidence: float = Field(
        0.0, description="Độ tin cậy nhận diện ÂM THANH (điểm fingerprint/embedding thật, "
                          "trùng với match.confidence).",
    )
    rights_confidence: float = Field(
        0.0, description="Độ tin cậy của METADATA QUYỀN = base − các khoản phạt khai báo trong "
                          "configs/rules_v1.yaml (nguồn PREDICTED/SIMULATED, thiếu verified_at, "
                          "giấy phép hết hạn...). Phép tính đầy đủ nằm ở "
                          "evidence.rule_engine.rights_confidence_breakdown. Dưới "
                          "rights_gate.min_rights_confidence thì risk bị hạ về UNKNOWN.",
    )
    decision_confidence: float = Field(
        0.0, description="Độ tin cậy của QUYẾT ĐỊNH RỦI RO CUỐI CÙNG = min(identity_confidence, "
                          "rights_confidence), và = 0 khi risk=UNKNOWN. KHÔNG PHẢI trung bình "
                          "cộng, không thay thế hai field kia. Phép tính cụ thể nằm ở "
                          "evidence.rule_engine.decision_confidence_formula.",
    )


class AnalysisResult(BaseModel):
    """Kết quả phân tích đầy đủ — `GET /results/{job_id}` khi status=DONE (§11)."""
    status: JobStatus = Field(..., description="Trạng thái tác vụ tại thời điểm trả về.")
    job_id: str = Field(..., description="UUID của tác vụ.")
    identity: Identity = Field(..., description="Bài hát/nghệ sĩ/ID đã nhận diện được.")
    match: Match = Field(..., description="Kết quả tầng nhận diện âm thanh (cascade).")
    rights: Rights = Field(..., description="Thông tin pháp lý/giấy phép tra được.")
    assessment: Assessment = Field(..., description="Kết quả Rule Engine: mức rủi ro, nhóm, điều kiện.")
    recommendation: str = Field(..., description="Khuyến nghị hành động cụ thể, dễ hiểu cho "
                                                   "người dùng cuối (không phải tư vấn pháp lý "
                                                   "có giá trị tài phán).")
    evidence: dict[str, Any] = Field(
        ...,
        description=(
            "Bằng chứng chi tiết phục vụ giải thích/kiểm toán (§2: không hộp đen) — dict lồng "
            "nhau, KHÔNG ép kiểu cố định vì mỗi rule có thể gắn thêm field riêng qua "
            "extra_evidence. Các khoá thường có: "
            "`identification` — từng tầng cascade: `fingerprint` (fingerprint_score, threshold "
            "hiệu dụng; khi tầng 1 KHÔNG chạy thì fingerprint_score=null kèm `reason_code` "
            "DATABASE_UNAVAILABLE|FPCALC_MISSING và `message`), `embedding` (top_k, threshold, "
            "reference_vectors), `cover`, `thresholds`, `timings_ms`, `decision_reason`; "
            "`candidates` — Top-K ứng viên từ vector search (vẫn có khi match.type=UNKNOWN); "
            "`rule_engine` — rules_version, rule_id, matched_conditions, "
            "`rights_confidence_penalties` (từng khoản trừ kèm mức trừ), "
            "`rights_confidence_breakdown` ({base, penalties[{code, amount, reason}], final, "
            "formula}), `decision_confidence_formula`; khi cổng dữ liệu quyền chặn thì thêm "
            "`min_rights_confidence`, `rights_shortfall` và `provisional_decision` — kết luận "
            "TẠM của 5 nhóm, chỉ để tham khảo, KHÔNG phải kết luận; "
            "`composition`/`rights_record` — bản ghi gốc tra được từ CSDL (quyền suy đoán có "
            "source bắt đầu bằng PREDICTED); `license_prediction` — đầu ra của license "
            "classifier (xác suất, validation, warning); `production_features` — đặc trưng âm "
            "học phụ, không dùng để suy luận quyền sử dụng."
        ),
    )
    latency_ms: Optional[float] = Field(None, description="Tổng thời gian xử lý toàn bộ pipeline (ms).")
    model_version: Optional[str] = Field(None, description="Phiên bản model nhận diện đã dùng "
                                                             "(vd MERT-v1-95M).")


class FeedbackRequest(BaseModel):
    """Phản hồi của người dùng về độ chính xác của một kết quả — `POST /feedback`."""
    job_id: Optional[str] = Field(None, description="Tác vụ mà phản hồi này nói về (nếu có).")
    recording_id: Optional[str] = Field(None, description="Bản ghi mà phản hồi này nói về (nếu có).")
    verdict: FeedbackVerdict = Field(..., description="Correct | Incorrect | Unsure.")
    note: Optional[str] = Field(None, max_length=2000, description="Ghi chú tự do, tối đa 2000 ký tự.")


class FeedbackResponse(BaseModel):
    feedback_id: str = Field(..., description="UUID của bản ghi phản hồi vừa lưu.")
    message: str = Field(..., description="Thông báo xác nhận cho người dùng.")


class ErrorResponse(BaseModel):
    """Cấu trúc lỗi chuẩn hoá — không bao giờ trả traceback nội bộ ra client (§12)."""
    error_code: str = Field(..., description="Mã lỗi chuẩn hoá (FILE_TOO_LARGE, "
                                              "UNSUPPORTED_FORMAT, NO_AUDIO, NO_MUSIC, "
                                              "MODEL_FAILURE, DATABASE_FAILURE, TIMEOUT, "
                                              "UNKNOWN_TRACK, LOW_CONFIDENCE...).")
    message: str = Field(..., description="Thông báo lỗi dễ hiểu cho người dùng.")
