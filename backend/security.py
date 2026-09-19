"""
Các cơ chế bảo mật cho API và hệ thống (Security Hardening).

1. Sanitize tên file trước khi log hoặc ghi CSDL (tránh Log Injection, CRLF).
2. Rate limiting theo IP (giới hạn số request/phút, trả về 429 RATE_LIMITED).
3. Concurrency limiter (giới hạn số tác vụ phân tích đồng thời bằng Semaphore).
4. Xác thực API Key tuỳ chọn qua header X-API-Key (so sánh constant-time).
5. Lọc thông tin nhạy cảm khỏi evidence và /health khi EVIDENCE_DETAIL=public.
"""
import math
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import HTTPException, Request

from backend import config


def sanitize_filename(filename: Optional[str]) -> str:
    """
    Chuẩn hoá tên file do người dùng cung cấp:
    - Loại bỏ ký tự điều khiển (CR, LF, TAB, null-byte, v.v.).
    - Cắt tối đa 255 ký tự.
    - Loại bỏ khoảng trắng đầu/cuối.
    """
    if not filename:
        return "unnamed"
    # Loại bỏ control characters: \x00-\x1f và \x7f-\x9f
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", str(filename)).strip()
    if not cleaned:
        return "unnamed"
    return cleaned[:255]


class IPRateLimiter:
    """
    Sliding-window in-memory rate limiter theo IP.
    Mỗi IP được phép gửi tối đa `max_requests` trong `window_seconds`.
    """

    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._history = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, ip: str) -> tuple[bool, int]:
        """
        Kiểm tra và ghi nhận request.
        Trả về: (cho_phép: bool, retry_after: int)
        """
        limit = getattr(config, "RATE_LIMIT_PER_MIN", self.max_requests)
        now = time.time()
        with self._lock:
            queue = self._history[ip]
            # Xoá các timestamp quá hạn
            cutoff = now - self.window_seconds
            while queue and queue[0] <= cutoff:
                queue.popleft()

            if len(queue) >= limit:
                oldest = queue[0]
                retry_after = max(1, int(math.ceil(self.window_seconds - (now - oldest))))
                return False, retry_after

            queue.append(now)
            return True, 0

    def reset(self):
        """Dùng cho testing."""
        with self._lock:
            self._history.clear()


class JobConcurrencyLimiter:
    """
    Giới hạn số tác vụ phân tích chạy đồng thời bằng Semaphore non-blocking.
    """

    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self._semaphore = threading.BoundedSemaphore(max(1, max_concurrent))

    def acquire(self) -> bool:
        """Thử lấy 1 slot xử lý không chặn. Trả về True nếu thành công."""
        return self._semaphore.acquire(blocking=False)

    def release(self):
        """Giải phóng slot sau khi xử lý xong."""
        try:
            self._semaphore.release()
        except ValueError:
            pass


# Khởi tạo các singleton limiter
rate_limiter = IPRateLimiter(
    max_requests=config.RATE_LIMIT_PER_MIN,
    window_seconds=60.0,
)

job_concurrency_limiter = JobConcurrencyLimiter(
    max_concurrent=config.MAX_CONCURRENT_JOBS,
)


def get_client_ip(request: Request) -> str:
    """Trích xuất địa chỉ IP của client từ request."""
    # Header X-Forwarded-For nếu nằm sau reverse proxy
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


def verify_api_key(request: Request) -> None:
    """
    Xác thực API Key:
    - Nếu config.API_KEY không được đặt -> cho qua (hành vi cũ).
    - Nếu config.API_KEY được đặt -> kiểm tra header X-API-Key bằng constant-time compare.
    """
    configured_key = getattr(config, "API_KEY", None)
    if not configured_key:
        return

    provided_key = request.headers.get("x-api-key")
    if not provided_key or not secrets.compare_digest(provided_key, configured_key):
        raise HTTPException(
            status_code=401,
            detail={
                "error_code": "UNAUTHORIZED",
                "message": "API Key không hợp lệ hoặc thiếu header X-API-Key.",
            },
        )


def filter_public_evidence(obj):
    """
    Khi EVIDENCE_DETAIL == 'public':
    - Loại bỏ các thông tin về ngưỡng nội bộ: threshold, thresholds, threshold_base, v.v.
    - Làm tròn các điểm số trong evidence về 2 chữ số thập phân.
    """
    if getattr(config, "EVIDENCE_DETAIL", "full") != "public":
        return obj

    if not isinstance(obj, dict):
        return obj

    result = dict(obj)
    evidence = result.get("evidence")
    if isinstance(evidence, dict):
        result["evidence"] = _sanitize_evidence_dict(evidence)
    return result


def _sanitize_evidence_dict(d: dict) -> dict:
    sanitized = {}
    for k, v in d.items():
        if k in (
            "threshold",
            "thresholds",
            "threshold_base",
            "threshold_raised_for_short_query",
            "min_identity_confidence",
            "min_rights_confidence",
        ):
            continue

        if isinstance(v, float):
            sanitized[k] = round(v, 2)
        elif isinstance(v, dict):
            sanitized[k] = _sanitize_evidence_dict(v)
        elif isinstance(v, list):
            sanitized[k] = [
                _sanitize_evidence_dict(item) if isinstance(item, dict)
                else (round(item, 2) if isinstance(item, float) else item)
                for item in v
            ]
        else:
            sanitized[k] = v
    return sanitized
