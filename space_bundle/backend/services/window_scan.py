"""
Quét đa cửa sổ (Multi-window Audio Scanning).

Cho phép nhận diện các file âm thanh dài (> 30s) bằng cách quét nhiều cửa sổ
30 giây dọc theo toàn bộ thời lượng bài hát, thay vì chỉ 30–120s đầu tiên.
Không thay đổi bất kỳ ngưỡng nhận diện nào đã hiệu chỉnh.
"""
import math
import os
from typing import List, Optional

import numpy as np

from backend import config


def plan_windows(
    duration_s: float,
    window_s: float = 30.0,
    max_windows: Optional[int] = None,
) -> List[float]:
    """
    Lập kế hoạch các điểm bắt đầu cửa sổ quét:
    - Luôn gồm 0 và cửa sổ đuôi (duration - window_s).
    - Cách đều nhau, tăng dần, không quá max_windows.
    - File <= window_s -> [0].
    """
    if max_windows is None:
        max_windows = config.SCAN_MAX_WINDOWS

    duration = float(duration_s)
    window = float(window_s)

    if duration <= window or max_windows <= 1:
        return [0]

    tail = duration - window
    if tail <= 0:
        return [0]

    # Số cửa sổ tự nhiên cần để phủ bài hát
    needed = max(2, int(math.ceil(duration / window)))
    num_windows = min(max_windows, needed)

    raw = np.linspace(0.0, tail, num=num_windows)
    points = [round(float(x), 2) for x in raw]
    points[0] = 0
    points[-1] = round(tail, 2)
    return points


def get_audio_duration(audio_path: str) -> float:
    """Lấy thời lượng file audio (tính bằng giây) một cách tối ưu."""
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        if info.duration > 0:
            return float(info.duration)
    except Exception:
        pass

    try:
        import librosa

        dur = librosa.get_duration(path=audio_path)
        return float(dur)
    except Exception:
        return 0.0


def slice_audio(
    audio_path: str,
    start_s: float,
    duration_s: float = 30.0,
    output_dir: Optional[str] = None,
) -> str:
    """
    Trích xuất một đoạn audio 24kHz mono dài duration_s bắt đầu từ start_s
    và lưu thành file WAV tạm trên đĩa.
    """
    import uuid

    import librosa
    import soundfile as sf

    out_dir = output_dir or config.TEMP_UPLOAD_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"slice_{uuid.uuid4().hex}.wav")

    y, sr = librosa.load(
        audio_path,
        sr=config.MERT_SAMPLE_RATE,
        mono=True,
        offset=max(0.0, float(start_s)),
        duration=float(duration_s),
    )
    sf.write(out_path, y, sr)
    return out_path
