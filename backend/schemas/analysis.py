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
    platform: Platform = Platform.YOUTUBE
    commercial_use: bool = False
    monetization: bool = False


class AnalyzeAccepted(BaseModel):
    job_id: str
    status: JobStatus
    message: str


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
    job_id: str
    status: JobStatus
    stage: Optional[PipelineStage] = None
    filename: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class Identity(BaseModel):
    track: Optional[str] = None
    artist: Optional[str] = None
    recording_id: Optional[str] = None
    composition_id: Optional[str] = None


class Match(BaseModel):
    type: MatchType
    confidence: float = Field(..., description="identity_confidence của tầng nhận diện")
    pipeline_stage: Optional[str] = None


class Rights(BaseModel):
    license: Optional[str] = None
    copyright_status: Optional[str] = None
    attribution_required: Optional[bool] = None
    commercial_use_allowed: Optional[bool] = None
    monetization_allowed: Optional[bool] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    verified_at: Optional[str] = None
    rights_found: bool = False


class Assessment(BaseModel):
    risk: RiskLevel
    category: Optional[str] = None
    condition: Optional[str] = None
    reason: Optional[str] = None
    # Ba loại độ tin cậy TÁCH BIỆT (§2) — không gộp thành một "copyright probability"
    identity_confidence: float = 0.0
    rights_confidence: float = 0.0
    decision_confidence: float = 0.0


class AnalysisResult(BaseModel):
    status: JobStatus
    job_id: str
    identity: Identity
    match: Match
    rights: Rights
    assessment: Assessment
    recommendation: str
    evidence: dict[str, Any]
    latency_ms: Optional[float] = None
    model_version: Optional[str] = None


class FeedbackRequest(BaseModel):
    job_id: Optional[str] = None
    recording_id: Optional[str] = None
    verdict: FeedbackVerdict
    note: Optional[str] = Field(None, max_length=2000)


class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str


class ErrorResponse(BaseModel):
    error_code: str
    message: str
