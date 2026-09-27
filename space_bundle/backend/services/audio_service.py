"""
Tiền xử lý audio.

Hai điều đã kiểm chứng trên môi trường thật và định hình thiết kế ở đây:

1. Chromaprint phải chạy trên FILE GỐC. `fpcalc` 1.5.1 nhúng sẵn FFmpeg nên tự
   giải mã mp3/mp4/flac; đưa file đã qua một vòng chuyển mã vào chỉ làm lệch
   fingerprint so với reference DB.
2. `librosa.load(..., sr=24000)` tự giải mã và resample mp3/wav/flac/m4a qua
   libsndfile, nên với file AUDIO thì bước chuẩn hoá bằng FFmpeg là thừa.
   FFmpeg chỉ thực sự cần khi đầu vào là VIDEO (mp4/mkv/mov/webm/avi).

Nhờ vậy hệ thống chạy được cả khi máy không có FFmpeg — chỉ mất khả năng xử lý
video, và điều đó được báo rõ bằng mã lỗi thay vì hỏng âm thầm.

Ngoại lệ đã gặp thật (2026-09-27): libsndfile KHÔNG giải mã được AAC, nên
`.m4a`/`.aac` cũng phải qua FFmpeg như video. Trước đây fpcalc vẫn đọc được file,
còn MERT thì báo NO_AUDIO, nên người dùng chỉ thấy lỗi "không đọc được file".
FFmpeg lấy từ PATH, nếu không có thì dùng bản đóng gói sẵn của gói pip
`imageio-ffmpeg`. Máy Windows không cần cài FFmpeg riêng.
"""
import logging
import os
import shutil
import subprocess
from functools import lru_cache

from backend import config

logger = logging.getLogger("music_rights_ai")

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv", ".m4v")
# Audio mà libsndfile (soundfile 0.13, libsndfile 1.2) không giải mã được: AAC trong
# MP4/ADTS. mp3/wav/flac/ogg/opus thì librosa đọc thẳng, không cần FFmpeg.
FFMPEG_AUDIO_EXTENSIONS = (".m4a", ".aac")


class AudioProcessingError(RuntimeError):
    """Không xử lý được file đầu vào. `code` map thẳng sang mã lỗi API (§12)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _runs(exe: str) -> bool:
    try:
        proc = subprocess.run(
            [exe, "-version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
        )
        return proc.returncode == 0
    except Exception:
        return False


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str | None:
    """
    Đường dẫn FFmpeg CHẠY ĐƯỢC, không chỉ tồn tại: PATH trước, rồi bản đóng gói
    của `imageio-ffmpeg`. (Trên máy dev, ffmpeg.exe của conda từng tồn tại nhưng lỗi
    STATUS_ENTRYPOINT_NOT_FOUND — `shutil.which` không phát hiện được.)
    """
    candidates = [shutil.which("ffmpeg")]
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    return next((exe for exe in candidates if exe and _runs(exe)), None)


def ffmpeg_available() -> bool:
    return ffmpeg_exe() is not None


def is_video(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in VIDEO_EXTENSIONS


def needs_ffmpeg(filename: str) -> bool:
    """Video, hoặc audio mà libsndfile không giải mã được (AAC)."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in VIDEO_EXTENSIONS or ext in FFMPEG_AUDIO_EXTENSIONS


def validate_upload(file_path: str, filename: str = None) -> None:
    """Kiểm tra đuôi file và dung lượng trước khi xử lý (§12)."""
    name = filename or os.path.basename(file_path)
    ext = os.path.splitext(name)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise AudioProcessingError(
            "UNSUPPORTED_FORMAT",
            f"Đuôi file '{ext or 'không rõ'}' không được hỗ trợ. "
            f"Chấp nhận: {', '.join(config.ALLOWED_EXTENSIONS)}",
        )
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        raise AudioProcessingError(
            "FILE_TOO_LARGE",
            f"File {size_mb:.1f} MB vượt giới hạn {config.MAX_UPLOAD_MB} MB",
        )


def extract_and_normalize_audio(input_file_path: str, output_dir: str = None) -> str:
    """Tách/giải mã audio (video, m4a/aac) sang WAV mono 24kHz. Ném lỗi nếu thất bại."""
    output_dir = output_dir or config.TEMP_UPLOAD_DIR
    os.makedirs(output_dir, exist_ok=True)

    if not ffmpeg_available():
        raise AudioProcessingError(
            "MODEL_FAILURE",
            "Cần FFmpeg để giải mã file video hoặc m4a/aac nhưng FFmpeg không khả dụng "
            "trên máy chủ. Hãy dùng file mp3, wav hoặc flac.",
        )

    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_wav_path = os.path.join(output_dir, f"{base_name}_normalized.wav")

    cmd = [
        ffmpeg_exe() or "ffmpeg", "-nostdin", "-y",
        "-t", str(config.MAX_MEDIA_SECONDS),
        "-i", input_file_path,
        "-vn",                                 # bỏ luồng video
        "-ar", str(config.MERT_SAMPLE_RATE),   # 24kHz cho MERT
        "-ac", "1",                            # mono
        output_wav_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=config.FFMPEG_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        if os.path.exists(output_wav_path):
            try:
                os.remove(output_wav_path)
            except OSError:
                pass
        logger.warning(
            "FFmpeg timeout sau %s giay (%s)", config.FFMPEG_TIMEOUT_S, input_file_path
        )
        raise AudioProcessingError(
            "TIMEOUT",
            f"Thời gian xử lý FFmpeg vượt quá giới hạn ({config.FFMPEG_TIMEOUT_S}s).",
        ) from None

    if proc.returncode != 0 or not os.path.exists(output_wav_path):
        # stderr của FFmpeg chứa đường dẫn tuyệt đối trên máy chủ -> chỉ ghi log
        detail = proc.stderr.decode("utf-8", errors="replace").strip()[-300:]
        logger.warning("FFmpeg that bai (%s): %s", input_file_path, detail)
        raise AudioProcessingError(
            "NO_AUDIO", "FFmpeg không đọc được luồng âm thanh của file (file hỏng hoặc không có tiếng)."
        )
    return output_wav_path


def prepare_for_embedding(input_file_path: str, filename: str = None):
    """
    Trả (đường_dẫn_cho_MERT, file_tạm_cần_xoá).

    - Video, m4a/aac -> giải mã bằng FFmpeg (file tạm cần dọn sau khi dùng).
    - Audio còn lại  -> dùng thẳng file gốc, librosa tự resample về 24kHz.
    """
    name = filename or os.path.basename(input_file_path)
    if needs_ffmpeg(name):
        wav_path = extract_and_normalize_audio(input_file_path)
        return wav_path, wav_path
    return input_file_path, None
