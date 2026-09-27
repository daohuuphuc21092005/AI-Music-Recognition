"""
Sổ bài chưa nhận diện — nguồn dữ liệu của tab "Thống kê & Tra cứu".

Mỗi kết quả UNKNOWN của /analyze được ghi vào sổ, theo hai loại:
  - NOT_IDENTIFIED: `match.type == UNKNOWN`, không khớp bản ghi nào trong kho
    tham chiếu. Gộp lượt gặp theo thứ tự cascade: fingerprint Chromaprint → MERT →
    Cover, nên cùng một bài gửi lại, kể cả bản đã nén, cắt, đổi tông, chỉ là một mục
    với số lượt gặp tăng dần. Sổ "học" mỗi bài bằng cách lưu vector MERT từng đoạn +
    descriptor Cover (`registry_features`, bảng `unknown_track_features`).
  - RIGHTS_UNKNOWN: đã nhận diện được bản ghi nhưng mức rủi ro vẫn UNKNOWN (thiếu
    quyền, quyền mô phỏng chưa xác minh, loại giấy phép ngoài 5 nhóm). Gộp theo
    `recording_id`.

Ba điều service này KHÔNG làm, và không được làm:
  1. Không đi vào Rule Engine. Tên bài/ghi chú của người thẩm định chỉ để tra
     cứu; một bài gặp lại lần thứ mười vẫn giữ nguyên risk/category/condition.
     Coi ghi chú là dữ liệu quyền chính là kết luận từ nguồn chưa kiểm chứng
     mà CLAUDE.md §2.3/§2.5 cấm.
  2. Không lưu file audio (§8) — chỉ fingerprint, vector MERT mean-pooled và chroma
     64 khung; không thứ nào dựng lại được âm thanh.
  3. Không trả `job_id` ra API: job_id là khoá truy cập kết quả (docs/SECURITY.md).
"""
import logging
import os
import re
import time
import uuid
from datetime import datetime

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend import config
from backend.services import audio_service, registry_features
from backend.services.chromaprint_codec import InvalidFingerprintError
from backend.services.cover_service import describe_oti
from backend.services.fingerprint_service import (
    AudioFingerprintError,
    FingerprintBackendUnavailable,
    _decode_array,
    extract_query_fingerprint,
    match_decoded,
    min_score_for_duration,
)

logger = logging.getLogger("music_rights_ai")

NOT_IDENTIFIED = "NOT_IDENTIFIED"
RIGHTS_UNKNOWN = "RIGHTS_UNKNOWN"

REGISTRY_NOTE = ("Thông tin sổ chỉ để tra cứu; không phải dữ liệu quyền "
                 "và không làm thay đổi mức rủi ro.")

# Khoá advisory của PostgreSQL cho bước "tìm mục cũ rồi thêm/cập nhật": hai job cùng
# một bài chạy song song (MAX_CONCURRENT_JOBS = 2) sẽ tạo HAI mục nếu không khoá.
_ADVISORY_LOCK_KEY = 0x554E4B4E   # "UNKN"

SIGHTINGS_IN_DETAIL = 50
DAYS_IN_STATS = 30

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_CONTROL_CHARS_KEEP_NEWLINE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


# --------------------------------------------------------------------------
# Phân loại & đối chiếu (hàm thuần, không đụng CSDL)
# --------------------------------------------------------------------------
def classify_unknown(result: dict):
    """
    Loại mục sổ cho một kết quả phân tích, hoặc None nếu kết quả đã xác định.

    `UNAVAILABLE` (tầng nhận diện không chạy được) là lỗi hạ tầng, không phải nhạc
    lạ, nên không vào sổ — nó cũng không có recording_id để rơi vào nhánh thứ hai.
    """
    match_type = (result.get("match") or {}).get("type")
    if match_type == "UNKNOWN":
        return NOT_IDENTIFIED
    risk = (result.get("assessment") or {}).get("risk")
    recording_id = (result.get("identity") or {}).get("recording_id")
    if risk == "UNKNOWN" and recording_id:
        return RIGHTS_UNKNOWN
    return None


def find_best_entry(query_vec: np.ndarray, query_duration, entries):
    """
    Mục sổ khớp nhất với fingerprint truy vấn: `(unknown_id, điểm)` hoặc None.

    `entries` = [(unknown_id, vector đã giải mã, độ dài)]. Ngưỡng chấp nhận là ngưỡng
    hiệu dụng của tầng 1 — `min_score_for_duration` theo đoạn NGẮN hơn trong cặp, vì
    điểm Chromaprint chia cho độ dài ngắn hơn và điểm nền tăng khi đoạn ngắn đi. Không
    đặt một ngưỡng riêng cho sổ: τFP đã được hiệu chỉnh bằng EXP-01.
    """
    best_id, best_score = None, 0.0
    for unknown_id, vector, duration in entries:
        if vector is None or vector.size == 0:
            continue
        score = match_decoded(query_vec, vector)
        durations = [float(d) for d in (query_duration, duration) if d]
        threshold = min_score_for_duration(min(durations) if durations else 0)
        if score >= threshold and score > best_score:
            best_id, best_score = unknown_id, score
    return (best_id, best_score) if best_id else None


def escape_like(value: str) -> str:
    """Escape ký tự đại diện của LIKE để ô tìm kiếm khớp đúng chữ người dùng gõ."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def clean_text(value, keep_newlines: bool = False):
    """Bỏ ký tự điều khiển; chuỗi rỗng sau khi lọc thành None (tức là xoá trường)."""
    if value is None:
        return None
    pattern = _CONTROL_CHARS_KEEP_NEWLINE if keep_newlines else _CONTROL_CHARS
    cleaned = pattern.sub("", str(value)).strip()
    return cleaned or None


def _public() -> bool:
    return getattr(config, "EVIDENCE_DETAIL", "full") == "public"


def _round(value, digits: int = 4):
    if value is None:
        return None
    return round(float(value), 2 if _public() else digits)


def _iso(value):
    """ISO 8601. Cột TIMESTAMPTZ -> kèm múi giờ; cắt micro giây (trình duyệt cũ không đọc)."""
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return value.isoformat() if hasattr(value, "isoformat") else (str(value) if value else None)


def _top_candidate(result: dict):
    """Ứng viên MERT đứng đầu (dưới ngưỡng) — gợi ý cho người thẩm định, không phải kết luận."""
    candidates = (result.get("evidence") or {}).get("candidates") or []
    top = candidates[0] if candidates and isinstance(candidates[0], dict) else None
    if not top or not top.get("recording_id"):
        return None, None
    score = top.get("similarity_score")
    return str(top["recording_id"]), (float(score) if score is not None else None)


# --------------------------------------------------------------------------
# Ghi sổ
# --------------------------------------------------------------------------
def fingerprint_for_registry(audio_path: str):
    """(độ dài, fingerprint) của file gửi lên, hoặc None khi fpcalc không chạy được."""
    try:
        return extract_query_fingerprint(audio_path)
    except (AudioFingerprintError, FingerprintBackendUnavailable) as e:
        logger.info("So chua nhan dien: khong sinh duoc fingerprint (%s)", type(e).__name__)
        return None


def features_for_registry(audio_path: str, filename: str = None):
    """
    (đặc trưng, mã lỗi): giải mã lại file (m4a/video qua FFmpeg như pipeline) rồi tính
    vector MERT từng đoạn + descriptor Cover. Hỏng thì (None, "FEATURES_FAILED") — sổ
    vẫn ghi lượt gặp bằng fingerprint, chỉ là lần này không học thêm được.
    """
    temp_path = None
    try:
        decoded_path, temp_path = audio_service.prepare_for_embedding(audio_path, filename)
        return registry_features.extract_features(decoded_path), None
    except Exception as e:
        logger.warning("So chua nhan dien: khong trich duoc dac trung (%s)", type(e).__name__)
        return None, "FEATURES_FAILED"
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _load_learned(db: Session):
    """Vector MERT và descriptor Cover đã học của các mục NOT_IDENTIFIED gặp gần nhất."""
    rows = db.execute(text("""
        SELECT f.unknown_id, f.feature_type, f.dimension, f.vector
        FROM unknown_track_features f
        JOIN (SELECT unknown_id FROM unknown_tracks
              WHERE kind = 'NOT_IDENTIFIED'
              ORDER BY last_seen_at DESC LIMIT :limit) recent
          ON recent.unknown_id = f.unknown_id
        WHERE (f.feature_type = 'MERT' AND f.model_version = :mert_version)
           OR (f.feature_type = 'COVER' AND f.model_version = :cover_version)
    """), {"limit": max(0, int(config.UNKNOWN_REGISTRY_SCAN_LIMIT)),
           "mert_version": registry_features.mert_version(),
           "cover_version": registry_features.COVER_VERSION}).fetchall()
    mert, cover = [], []
    for row in rows:
        try:
            vector = registry_features.from_bytes(row.vector, row.dimension)
        except ValueError:
            continue   # một vector hỏng không được làm hỏng cả lượt ghi
        (mert if row.feature_type == registry_features.MERT else cover).append(
            (str(row.unknown_id), vector))
    return mert, cover


def _learn(db: Session, unknown_id: str, features: dict) -> int:
    """
    Lưu phần đặc trưng MỚI của lần gửi này vào mục (`registry_features.novel_features`),
    không vượt trần: UNKNOWN_REGISTRY_MAX_FEATURES_PER_ENTRY vector MERT và
    MAX_COVER_PER_ENTRY descriptor Cover (mỗi lần gửi một descriptor: 30 giây đầu).
    """
    stored_rows = db.execute(text("""
        SELECT feature_type, model_version, dimension, vector
        FROM unknown_track_features WHERE unknown_id = :unknown_id
    """), {"unknown_id": unknown_id}).fetchall()
    mert_type, cover_type = registry_features.MERT, registry_features.COVER
    current = {mert_type: registry_features.mert_version(),
               cover_type: registry_features.COVER_VERSION}
    stored = {mert_type: [], cover_type: []}
    counts = {mert_type: 0, cover_type: 0}
    for row in stored_rows:
        counts[row.feature_type] = counts.get(row.feature_type, 0) + 1
        if row.model_version == current.get(row.feature_type):
            try:
                stored[row.feature_type].append(
                    registry_features.from_bytes(row.vector, row.dimension))
            except ValueError:
                continue
    mert_new, cover_new = registry_features.novel_features(
        features, stored[mert_type], stored[cover_type],
        mert_room=max(0, int(config.UNKNOWN_REGISTRY_MAX_FEATURES_PER_ENTRY) - counts[mert_type]),
        cover_room=max(0, registry_features.MAX_COVER_PER_ENTRY - counts[cover_type]))
    rows = [(mert_type, current[mert_type], start, end, vector) for start, end, vector in mert_new]
    if cover_new is not None:
        rows.append((cover_type, current[cover_type], 0.0, registry_features.COVER_BASE_S, cover_new))
    for feature_type, version, start, end, vector in rows:
        db.execute(text("""
            INSERT INTO unknown_track_features (
                feature_id, unknown_id, feature_type, model_version,
                segment_start, segment_end, dimension, vector
            ) VALUES (
                :feature_id, :unknown_id, :feature_type, :model_version,
                :segment_start, :segment_end, :dimension, :vector
            )
        """), {
            "feature_id": str(uuid.uuid4()), "unknown_id": unknown_id,
            "feature_type": feature_type, "model_version": version,
            "segment_start": start, "segment_end": end,
            "dimension": int(np.asarray(vector).size),
            "vector": registry_features.to_bytes(vector),
        })
    return len(rows)


def register_sighting(db: Session, *, job_id, result: dict, usage_context: dict,
                      filename: str, kind: str = None, fingerprint=None,
                      features: dict = None) -> dict:
    """
    Ghi một lượt gặp vào sổ: gộp vào mục cũ nếu đã gặp bài này, không thì tạo mục mới.

    `fingerprint` = (độ dài, fingerprint base64) — chỉ dùng cho NOT_IDENTIFIED.
    `features` = kết quả `registry_features.extract_features` — chỉ NOT_IDENTIFIED.
    Đối chiếu theo thứ tự cascade, dừng ở tầng đầu khớp: fingerprint (ngưỡng hiệu dụng
    tầng 1) → MERT (τMERT) → Cover (τCover). Không có cả hai thì lượt gặp thành mục mới.

    Sổ "học": lưu phần đặc trưng MỚI của lần gửi này — chưa nhận ra được bằng những gì
    mục đã có, theo đúng τMERT / τCover. Gửi lại đúng file đó không thêm gì; bản cắt ở
    đoạn khác, bản đổi tông, đổi nhịp thì được học thêm.

    Không đọc hay sửa `result["assessment"]`: sổ không có tiếng nói trong quyết định.
    """
    kind = kind or classify_unknown(result)
    if kind is None:
        return None

    started = time.perf_counter()
    assessment = result.get("assessment") or {}
    match_type = (result.get("match") or {}).get("type")
    recording_id = (result.get("identity") or {}).get("recording_id")
    hint_id, hint_score = _top_candidate(result) if kind == NOT_IDENTIFIED else (None, None)

    fp_text, fp_duration = None, None
    existing_id, match_score, entries_compared = None, None, 0
    match_method, match_threshold, match_oti, features_learned = None, None, None, 0
    try:
        db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _ADVISORY_LOCK_KEY})

        if kind == NOT_IDENTIFIED:
            if fingerprint:
                fp_duration, fp_raw = fingerprint
                fp_text = fp_raw.decode("ascii") if isinstance(fp_raw, bytes) else str(fp_raw)
                query_vec = _decode_array(fp_text)
                rows = db.execute(text("""
                    SELECT unknown_id, fingerprint, fingerprint_duration
                    FROM unknown_tracks
                    WHERE kind = 'NOT_IDENTIFIED' AND fingerprint IS NOT NULL
                    ORDER BY last_seen_at DESC
                    LIMIT :limit
                """), {"limit": max(0, int(config.UNKNOWN_REGISTRY_SCAN_LIMIT))}).fetchall()
                entries = []
                for row in rows:
                    try:
                        entries.append((str(row.unknown_id), _decode_array(row.fingerprint),
                                        row.fingerprint_duration))
                    except (InvalidFingerprintError, ValueError):
                        continue   # một mục hỏng không được làm hỏng cả lượt ghi
                entries_compared = len(entries)
                found = find_best_entry(query_vec, fp_duration, entries)
                if found:
                    existing_id, match_score = found
                    match_method = "FINGERPRINT"
                    entry_duration = next(d for uid, _v, d in entries if uid == existing_id)
                    durations = [float(d) for d in (fp_duration, entry_duration) if d]
                    match_threshold = min_score_for_duration(min(durations) if durations else 0)

            if existing_id is None and features:
                learned_mert, learned_cover = _load_learned(db)
                query_mert = [vector for _s, _e, vector in features.get("mert") or []]
                found_mert = registry_features.best_mert_match(
                    np.vstack(query_mert) if query_mert else None, learned_mert)
                if found_mert:
                    existing_id, match_score = found_mert
                    match_method, match_threshold = "MERT", config.MERT_THRESHOLD
                else:
                    cover_queries = features.get("cover_queries")
                    found_cover = registry_features.best_cover_match(
                        cover_queries[0] if cover_queries else None, learned_cover)
                    if found_cover:
                        existing_id, match_score, match_oti, _row = found_cover
                        match_method, match_threshold = "COVER", config.COVER_THRESHOLD
        else:
            row = db.execute(text("""
                SELECT unknown_id FROM unknown_tracks
                WHERE kind = 'RIGHTS_UNKNOWN' AND recording_id = :rec_id
            """), {"rec_id": recording_id}).fetchone()
            existing_id = str(row.unknown_id) if row else None

        snapshot = {
            "match_type": match_type,
            "risk": assessment.get("risk"),
            "category": assessment.get("category"),
            "condition": assessment.get("condition"),
            "hint_id": hint_id,
            "hint_score": hint_score,
        }
        if existing_id:
            unknown_id, is_new = existing_id, False
            db.execute(text("""
                UPDATE unknown_tracks
                SET last_seen_at = CURRENT_TIMESTAMP,
                    sighting_count = sighting_count + 1,
                    last_match_type = :match_type, last_risk_level = :risk,
                    last_category = :category, last_condition = :condition,
                    hint_recording_id = COALESCE(:hint_id, hint_recording_id),
                    hint_score = CASE WHEN :hint_id IS NULL THEN hint_score ELSE :hint_score END
                WHERE unknown_id = :unknown_id
            """), {**snapshot, "unknown_id": unknown_id})
        else:
            unknown_id, is_new = str(uuid.uuid4()), True
            db.execute(text("""
                INSERT INTO unknown_tracks (
                    unknown_id, kind, recording_id, fingerprint, fingerprint_duration,
                    first_filename, last_match_type, last_risk_level, last_category,
                    last_condition, hint_recording_id, hint_score
                ) VALUES (
                    :unknown_id, :kind, :recording_id, :fingerprint, :fp_duration,
                    :filename, :match_type, :risk, :category,
                    :condition, :hint_id, :hint_score
                )
            """), {
                **snapshot,
                "unknown_id": unknown_id,
                "kind": kind,
                "recording_id": recording_id if kind == RIGHTS_UNKNOWN else None,
                "fingerprint": fp_text,
                "fp_duration": fp_duration,
                "filename": filename,
            })

        if kind == NOT_IDENTIFIED and features:
            features_learned = _learn(db, unknown_id, features)

        db.execute(text("""
            INSERT INTO unknown_sightings (
                sighting_id, unknown_id, job_id, platform, commercial_use, monetization,
                match_score, match_method, risk_level, category
            ) VALUES (
                :sighting_id, :unknown_id, :job_id, :platform, :commercial_use, :monetization,
                :match_score, :match_method, :risk, :category
            )
        """), {
            "sighting_id": str(uuid.uuid4()),
            "unknown_id": unknown_id,
            "job_id": job_id,
            "platform": (usage_context or {}).get("platform"),
            "commercial_use": (usage_context or {}).get("commercial_use"),
            "monetization": (usage_context or {}).get("monetization"),
            "match_score": match_score,
            "match_method": match_method,
            "risk": assessment.get("risk"),
            "category": assessment.get("category"),
        })

        features_total = db.execute(text(
            "SELECT COUNT(*) FROM unknown_track_features WHERE unknown_id = :unknown_id"
        ), {"unknown_id": unknown_id}).scalar() if kind == NOT_IDENTIFIED else 0

        entry = db.execute(text("""
            SELECT sighting_count, first_seen_at, first_filename, review_status,
                   reviewer_title, reviewer_artist, reviewer_note
            FROM unknown_tracks WHERE unknown_id = :unknown_id
        """), {"unknown_id": unknown_id}).fetchone()
        db.commit()
    except Exception:
        db.rollback()
        raise

    info = {
        "unknown_id": unknown_id,
        "kind": kind,
        "is_new": is_new,
        "sighting_count": entry.sighting_count,
        "first_seen_at": _iso(entry.first_seen_at),
        "match_score": round(match_score, 4) if match_score is not None else None,
        # FINGERPRINT | MERT | COVER — tầng nào của sổ nhận ra bài này (None: mục mới)
        "match_method": match_method,
        "threshold": round(float(match_threshold), 4) if match_threshold is not None else None,
        "oti": match_oti,
        "oti_text": describe_oti(match_oti) if match_oti is not None else None,
        "entries_compared": entries_compared,
        "review_status": entry.review_status,
        "reviewer_title": entry.reviewer_title,
        "reviewer_artist": entry.reviewer_artist,
        "reviewer_note": entry.reviewer_note,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "note": REGISTRY_NOTE,
    }
    if kind == NOT_IDENTIFIED:
        info["fingerprint_available"] = fp_text is not None
        if not _public():
            # Bài chưa ai đặt tên thì màn Kết quả gọi nó bằng tên file lần đầu gửi
            info["first_filename"] = entry.first_filename
        info["learning"] = features is not None
        # Đoạn giống tiếng ồn: MERT không phân biệt được với ồn khác nên không học/so
        info["noise_like_segments"] = int((features or {}).get("noise_like_segments") or 0)
        info["features_learned"] = int(features_learned)
        info["features_total"] = int(features_total or 0)
    return info


def record_unknown(db: Session, *, job_id, audio_path: str, result: dict,
                   usage_context: dict, filename: str, kind: str) -> dict:
    """
    Sinh fingerprint + đặc trưng học (chỉ NOT_IDENTIFIED) rồi ghi lượt gặp. Tính đặc
    trưng TRƯỚC khi vào khoá advisory: suy luận MERT mất vài giây, không được giữ khoá
    của cả sổ trong lúc đó. Đo cả thời gian fpcalc + MERT.
    """
    started = time.perf_counter()
    fingerprint, features, features_error = None, None, None
    if kind == NOT_IDENTIFIED:
        fingerprint = fingerprint_for_registry(audio_path)
        if config.UNKNOWN_REGISTRY_LEARN:
            features, features_error = features_for_registry(audio_path, filename)
    info = register_sighting(db, job_id=job_id, result=result, usage_context=usage_context,
                             filename=filename, kind=kind, fingerprint=fingerprint,
                             features=features)
    if features_error:
        info["features_error"] = features_error
    info["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return info


# --------------------------------------------------------------------------
# Tra cứu
# --------------------------------------------------------------------------
_ENTRY_COLUMNS = """
    u.unknown_id, u.kind, u.recording_id, u.fingerprint_duration, u.first_filename,
    u.first_seen_at, u.last_seen_at, u.sighting_count, u.last_match_type,
    u.last_risk_level, u.last_category, u.last_condition, u.hint_recording_id,
    u.hint_score, u.review_status, u.reviewer_title, u.reviewer_artist,
    u.reviewer_note, u.reviewed_at,
    r.title AS catalogue_title, r.artist AS catalogue_artist
"""

_SORTS = {
    "last_seen": "u.last_seen_at DESC",
    "first_seen": "u.first_seen_at DESC",
    "sightings": "u.sighting_count DESC, u.last_seen_at DESC",
}


def _label(row) -> str:
    """Tên hiển thị: tên người thẩm định đặt → tên trong catalogue → tên file."""
    title = row.reviewer_title or row.catalogue_title
    # Tên bài do người thẩm định đặt thì không ghép với nghệ sĩ của catalogue
    artist = row.reviewer_artist or (None if row.reviewer_title else row.catalogue_artist)
    if title or artist:
        return " — ".join(part for part in (title or "Chưa rõ tên bài", artist) if part)
    if not _public() and row.first_filename:
        return row.first_filename
    return "Chưa đặt tên"


def _summary(row) -> dict:
    item = {
        "unknown_id": str(row.unknown_id),
        "kind": row.kind,
        "label": _label(row),
        "recording_id": str(row.recording_id) if row.recording_id else None,
        "first_seen_at": _iso(row.first_seen_at),
        "last_seen_at": _iso(row.last_seen_at),
        "sighting_count": row.sighting_count,
        "last_match_type": row.last_match_type,
        "last_risk_level": row.last_risk_level,
        "last_category": row.last_category,
        "review_status": row.review_status,
        "reviewer_title": row.reviewer_title,
        "reviewer_artist": row.reviewer_artist,
    }
    if not _public():
        item["first_filename"] = row.first_filename
    return item


def list_entries(db: Session, q: str = None, kind: str = None, status: str = None,
                 sort: str = "last_seen", limit: int = 20, offset: int = 0) -> dict:
    where, params = [], {}
    if kind:
        where.append("u.kind = :kind")
        params["kind"] = kind
    if status:
        where.append("u.review_status = :status")
        params["status"] = status

    q = clean_text(q)
    if q:
        try:
            params["uid"] = str(uuid.UUID(q))
            where.append("(u.unknown_id = :uid OR u.recording_id = :uid)")
        except ValueError:
            params["pattern"] = f"%{escape_like(q)}%"
            columns = ["u.reviewer_title", "u.reviewer_artist", "u.reviewer_note",
                       "r.title", "r.artist"]
            if not _public():
                # Ẩn tên file thì cũng không cho dò theo tên file.
                columns.append("u.first_filename")
            where.append("(" + " OR ".join(
                f"{c} ILIKE :pattern ESCAPE '\\'" for c in columns) + ")")

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    base = f"FROM unknown_tracks u LEFT JOIN recordings r ON r.recording_id = u.recording_id {where_sql}"

    total = db.execute(text(f"SELECT COUNT(*) {base}"), params).scalar() or 0
    rows = db.execute(text(f"""
        SELECT {_ENTRY_COLUMNS} {base}
        ORDER BY {_SORTS.get(sort, _SORTS['last_seen'])}
        LIMIT :limit OFFSET :offset
    """), {**params, "limit": limit, "offset": offset}).fetchall()

    return {"total": int(total), "limit": limit, "offset": offset,
            "items": [_summary(row) for row in rows]}


def get_entry(db: Session, unknown_id: str):
    row = db.execute(text(f"""
        SELECT {_ENTRY_COLUMNS}, r.source_dataset AS catalogue_source,
               h.title AS hint_title, h.artist AS hint_artist
        FROM unknown_tracks u
        LEFT JOIN recordings r ON r.recording_id = u.recording_id
        LEFT JOIN recordings h ON h.recording_id = u.hint_recording_id
        WHERE u.unknown_id = :unknown_id
    """), {"unknown_id": unknown_id}).fetchone()
    if row is None:
        return None

    learned = db.execute(text("""
        SELECT COUNT(*) FILTER (WHERE feature_type = 'MERT') AS mert,
               COUNT(*) FILTER (WHERE feature_type = 'COVER') AS cover
        FROM unknown_track_features WHERE unknown_id = :unknown_id
    """), {"unknown_id": unknown_id}).fetchone()

    sightings = db.execute(text("""
        SELECT seen_at, platform, commercial_use, monetization, match_score,
               match_method, risk_level, category
        FROM unknown_sightings
        WHERE unknown_id = :unknown_id
        ORDER BY seen_at DESC
        LIMIT :limit
    """), {"unknown_id": unknown_id, "limit": SIGHTINGS_IN_DETAIL}).fetchall()

    entry = _summary(row)
    entry.update({
        "last_condition": row.last_condition,
        "fingerprint_duration": _round(row.fingerprint_duration, 1),
        "reviewer_note": row.reviewer_note,
        "reviewed_at": _iso(row.reviewed_at),
        "catalogue": ({
            "recording_id": str(row.recording_id),
            "title": row.catalogue_title,
            "artist": row.catalogue_artist,
            "source_dataset": row.catalogue_source,
        } if row.recording_id else None),
        "hint": ({
            "recording_id": str(row.hint_recording_id),
            "title": row.hint_title,
            "artist": row.hint_artist,
            "score": _round(row.hint_score),
            "note": "Ứng viên MERT gần nhất nhưng DƯỚI ngưỡng τMERT — không phải kết luận.",
        } if row.hint_recording_id else None),
        # Sổ đã "học" bài này tới đâu: số đoạn MERT 15 s và số descriptor Cover đã lưu
        "learned": {"mert": int(learned.mert or 0), "cover": int(learned.cover or 0)},
        "sightings": [{
            "seen_at": _iso(s.seen_at),
            "platform": s.platform,
            "commercial_use": s.commercial_use,
            "monetization": s.monetization,
            "match_score": _round(s.match_score),
            "match_method": s.match_method,
            "risk_level": s.risk_level,
            "category": s.category,
        } for s in sightings],
        "note": REGISTRY_NOTE,
    })
    return entry


def update_review(db: Session, unknown_id: str, fields: dict):
    """
    Cập nhật ghi chú thẩm định. Chỉ các trường có mặt trong `fields` bị đổi; chuỗi
    rỗng xoá trường. Trả mục sau khi sửa, hoặc None nếu không có mục đó.
    """
    sets, params = [], {"unknown_id": unknown_id}
    for key, column, keep_newlines in (("title", "reviewer_title", False),
                                       ("artist", "reviewer_artist", False),
                                       ("note", "reviewer_note", True)):
        if key in fields:
            sets.append(f"{column} = :{column}")
            params[column] = clean_text(fields[key], keep_newlines=keep_newlines)
    status = fields.get("review_status")
    if status is not None:
        status = getattr(status, "value", status)
        sets.append("review_status = :review_status")
        sets.append("reviewed_at = " + ("CURRENT_TIMESTAMP" if status == "REVIEWED" else "NULL"))
        params["review_status"] = status

    try:
        if sets:
            updated = db.execute(text(f"""
                UPDATE unknown_tracks SET {', '.join(sets)}
                WHERE unknown_id = :unknown_id
                RETURNING unknown_id
            """), params).fetchone()
            db.commit()
            if updated is None:
                return None
    except Exception:
        db.rollback()
        raise
    return get_entry(db, unknown_id)


# --------------------------------------------------------------------------
# Thống kê
# --------------------------------------------------------------------------
def _counts(db: Session, sql: str) -> dict:
    return {(key if key is not None else "—"): int(n) for key, n in db.execute(text(sql)).fetchall()}


def get_stats(db: Session, utc_offset_min: int = 0) -> dict:
    """
    `utc_offset_min`: độ lệch múi giờ của NGƯỜI XEM so với UTC (UTC+7 -> 420). Lượt
    gặp được chia theo ngày ở giờ địa phương đó; không có thì lượt gặp lúc 1 giờ sáng
    giờ Việt Nam rơi vào ngày hôm trước (múi giờ của container PostgreSQL là UTC).
    """
    totals = db.execute(text("""
        SELECT COUNT(*) AS entries,
               COALESCE(SUM(sighting_count), 0) AS sightings,
               COUNT(*) FILTER (WHERE sighting_count >= 2) AS resighted,
               COUNT(*) FILTER (WHERE review_status = 'PENDING') AS pending,
               COUNT(*) FILTER (WHERE review_status = 'REVIEWED') AS reviewed
        FROM unknown_tracks
    """)).fetchone()

    by_day = db.execute(text(f"""
        WITH local AS (
            SELECT (now() AT TIME ZONE 'UTC' + make_interval(mins => :offset))::date AS today
        ),
        days AS (
            SELECT today - i AS day FROM local, generate_series(0, {DAYS_IN_STATS - 1}) AS i
        )
        SELECT days.day, COUNT(s.sighting_id) AS n
        FROM days
        LEFT JOIN unknown_sightings s
               ON (s.seen_at AT TIME ZONE 'UTC' + make_interval(mins => :offset))::date = days.day
        GROUP BY days.day ORDER BY days.day
    """), {"offset": int(utc_offset_min)}).fetchall()

    top = db.execute(text(f"""
        SELECT {_ENTRY_COLUMNS}
        FROM unknown_tracks u LEFT JOIN recordings r ON r.recording_id = u.recording_id
        WHERE u.sighting_count >= 2
        ORDER BY u.sighting_count DESC, u.last_seen_at DESC
        LIMIT 5
    """)).fetchall()

    # Tỷ lệ job bị ghi sổ, chỉ tính từ job ĐẦU TIÊN được ghi sổ: các job trước khi có
    # sổ không có cơ hội vào sổ, đưa chúng vào mẫu số sẽ kéo tỷ lệ xuống sai.
    jobs = db.execute(text("""
        WITH start AS (
            SELECT MIN(j.created_at) AS t
            FROM jobs j JOIN unknown_sightings s ON s.job_id = j.job_id
        )
        SELECT
            (SELECT COUNT(*) FROM jobs, start
             WHERE jobs.status = 'DONE' AND jobs.created_at >= start.t) AS done,
            (SELECT COUNT(DISTINCT job_id) FROM unknown_sightings
             WHERE job_id IS NOT NULL) AS registered
    """)).fetchone()

    return {
        "registry_enabled": bool(config.UNKNOWN_REGISTRY_ENABLED),
        "totals": {
            "entries": int(totals.entries),
            "sightings": int(totals.sightings),
            "resighted_entries": int(totals.resighted),
            "pending": int(totals.pending),
            "reviewed": int(totals.reviewed),
        },
        "by_kind": _counts(db, "SELECT kind, COUNT(*) FROM unknown_tracks GROUP BY kind"),
        "by_last_risk": _counts(db, "SELECT last_risk_level, COUNT(*) FROM unknown_tracks "
                                    "GROUP BY last_risk_level"),
        "by_platform": _counts(db, "SELECT platform, COUNT(*) FROM unknown_sightings "
                                   "GROUP BY platform"),
        "sightings_by_day": [{"date": _iso(day), "count": int(n)} for day, n in by_day],
        "top_resighted": [_summary(row) for row in top],
        "jobs": {"done_since_start": int(jobs.done or 0),
                 "registered": int(jobs.registered or 0)},
    }
