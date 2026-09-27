"""
Dựng tập truy vấn kiểm thử bằng cách biến đổi audio gốc (§8).

Sinh ra `data/test_queries/` + manifest CSV để các thí nghiệm EXP-01/04/05 dùng
chung một bộ truy vấn, có nhãn đúng rõ ràng.

Các phép biến đổi thực hiện bằng librosa/soundfile nên KHÔNG cần FFmpeg. Riêng
nén mp3/aac chỉ chạy được nếu libsndfile trên máy hỗ trợ ghi MP3 — nếu không,
script bỏ qua và ghi rõ trong manifest thay vì âm thầm tạo file WAV rồi gán
nhãn "mp3".

Dùng:
    python scripts/augment_audio.py --file test.mp3 --recording-id <UUID>
    python scripts/augment_audio.py --from-db          # mọi recording có audio_path thật
"""
import argparse
import csv
import os
import sys
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import numpy as np
import soundfile as sf
from scipy import signal as scipy_signal

from backend import config
from experiments.common import Checkpoint

OUTPUT_DIR = os.path.join(config.BASE_DIR, "data", "test_queries")
MANIFEST = os.path.join(OUTPUT_DIR, "manifest.csv")
SR = 22050
CROP_START = 30.0     # giây, tránh phần intro thường im lặng
CROP_LENGTH = 30.0

# libsndfile nhận "mức nén" 0.0-1.0 chứ không nhận bitrate, nên hai giá trị dưới
# đây là kết quả ĐO trên tín hiệu mono 22050 Hz: 0.20 cho đúng 128 kbps, 0.60 cho
# đúng 64 kbps. Bitrate đạt được của từng file vẫn ghi vào manifest để đối chiếu.
#
# Trước đây cả hai biến thể đều gọi sf.write(..., format="MP3") mà không truyền
# tham số nén nào, nên sinh ra HAI FILE Y HỆT NHAU (trùng cả MD5) — tức nhóm
# "nén codec" của EXP-05 thực chất chỉ thử một mức nén rồi đếm hai lần.
MP3_LEVELS = {"mp3_128k": 0.20, "mp3_64k": 0.60}

# Hệ số co giãn thời gian và lượng dịch cao độ của từng biến thể, ghi vào manifest
# để thí nghiệm ánh xạ truy vấn về trục thời gian bản gốc mà không phải đoán.
TEMPO_FACTORS = {"tempo_0_90": 0.90, "tempo_0_95": 0.95,
                 "tempo_1_05": 1.05, "tempo_1_10": 1.10,
                 "composite_pitch_tempo_noise": 0.95}
PITCH_STEPS = {"pitch_plus_1": 1, "pitch_plus_2": 2,
               "pitch_minus_1": -1, "pitch_minus_2": -2,
               "composite_pitch1_crop10s": 1, "composite_pitch2_crop10s": 2,
               "composite_pitch_tempo_noise": 1}


def mp3_supported() -> bool:
    try:
        return "MP3" in sf.available_formats()
    except Exception:
        return False


def add_noise(audio: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2)) or 1e-6
    noise_rms = rms / (10 ** (snr_db / 20))
    noise = np.random.RandomState(seed).normal(0, noise_rms, audio.shape)
    return np.clip(audio + noise, -1.0, 1.0).astype("float32")


def lowpass(audio: np.ndarray, cutoff_hz: float = 4000.0) -> np.ndarray:
    sos = scipy_signal.butter(8, cutoff_hz / (SR / 2), btype="low", output="sos")
    return scipy_signal.sosfilt(sos, audio).astype("float32")


def build_variants(audio: np.ndarray, overlay: np.ndarray = None, composite: bool = False) -> dict:
    """Trả {tên_biến_đổi: mảng audio}. Mọi biến đổi đều trên cùng đoạn crop 30s."""
    variants = {
        "original_crop30s": audio,
        "crop_15s": audio[: int(15 * SR)],
        "crop_10s": audio[: int(10 * SR)],
        "noise_snr20": add_noise(audio, 20, seed=1),
        "noise_snr10": add_noise(audio, 10, seed=2),
        "noise_snr5": add_noise(audio, 5, seed=3),
        "gain_minus12db": (audio * (10 ** (-12 / 20))).astype("float32"),
        "eq_lowpass_4k": lowpass(audio),
        "pitch_plus_1": librosa.effects.pitch_shift(y=audio, sr=SR, n_steps=1),
        "pitch_minus_1": librosa.effects.pitch_shift(y=audio, sr=SR, n_steps=-1),
        "pitch_plus_2": librosa.effects.pitch_shift(y=audio, sr=SR, n_steps=2),
        "pitch_minus_2": librosa.effects.pitch_shift(y=audio, sr=SR, n_steps=-2),
        "tempo_0_90": librosa.effects.time_stretch(y=audio, rate=0.90),
        "tempo_0_95": librosa.effects.time_stretch(y=audio, rate=0.95),
        "tempo_1_05": librosa.effects.time_stretch(y=audio, rate=1.05),
        "tempo_1_10": librosa.effects.time_stretch(y=audio, rate=1.10),
    }

    if overlay is not None and len(overlay) > 0:
        # Chồng một nguồn âm thanh khác lên (mô phỏng giọng nói/nhạc nền chồng lấn)
        other = np.resize(overlay, audio.shape)
        mixed = audio + 0.35 * other
        peak = np.max(np.abs(mixed)) or 1.0
        variants["audio_overlay"] = (mixed / peak * 0.95).astype("float32")

    if composite:
        # 1. Pitch shift ±1 / ±2 kết hợp crop 10s và offset (cắt từ giây 10 đến 20 của audio 30s rồi shift)
        crop_10_20 = audio[int(10 * SR):int(20 * SR)] if len(audio) >= int(20 * SR) else audio[int(10 * SR):]
        if len(crop_10_20) > 0:
            variants["composite_pitch1_crop10s"] = librosa.effects.pitch_shift(y=crop_10_20, sr=SR, n_steps=1)
            variants["composite_pitch2_crop10s"] = librosa.effects.pitch_shift(y=crop_10_20, sr=SR, n_steps=2)

        # 2. Pitch + tempo 0.95 + noise SNR 10: time_stretch(0.95) -> pitch_shift(+1) -> add_noise(10)
        stretched = librosa.effects.time_stretch(y=audio, rate=0.95)
        shifted = librosa.effects.pitch_shift(y=stretched, sr=SR, n_steps=1)
        variants["composite_pitch_tempo_noise"] = add_noise(shifted, snr_db=10, seed=4)

        # 3. Ghép 150s tĩnh lặng (silence + noise nhẹ) vào trước đoạn audio (để test cửa sổ quét sau 2.5 phút)
        noise_150s = np.random.RandomState(42).normal(0, 1e-4, int(150 * SR)).astype("float32")
        variants["composite_padded_offset150s"] = np.concatenate([noise_150s, audio])

    return variants


def load_crop(path: str, prefer_start: float = CROP_START):
    """
    Đọc đoạn `CROP_LENGTH` giây, bắt đầu từ `prefer_start` nếu file đủ dài.

    Trả (audio, điểm_cắt_thực_tế). Không thể chỉ thử offset rồi xem mảng có rỗng
    không: với file ngắn hơn offset, libsndfile NÉM LỖI ngay ở bước seek
    ("Internal psf_fseek() failed") chứ không trả mảng rỗng. Nên phải đo độ dài
    trước, và vẫn giữ try/except cho các định dạng không đo được metadata.
    """
    try:
        duration = librosa.get_duration(path=path)
    except Exception:
        duration = 0.0

    # Cần chừa ít nhất 1 giây sau điểm cắt thì mới cắt ở đó
    start = prefer_start if duration >= prefer_start + 1.0 else 0.0
    try:
        audio, _ = librosa.load(path, sr=SR, mono=True,
                                offset=start, duration=CROP_LENGTH)
    except Exception:
        if start == 0.0:
            return None, 0.0
        start = 0.0
        try:
            audio, _ = librosa.load(path, sr=SR, mono=True, duration=CROP_LENGTH)
        except Exception:
            return None, 0.0
    return audio, start


def as_stored_path(path: str) -> str:
    """
    Rút gọn đường dẫn theo ĐÚNG quy ước của `recordings.audio_path`: tương đối so
    với AUDIO_ROOT nếu nằm trong corpus, nếu không thì so với thư mục repo.

    Không dùng thẳng os.path.relpath(path, BASE_DIR): trên Windows nó NÉM
    ValueError khi hai bên khác ổ đĩa — mà để corpus sang ổ khác chính là cách
    dùng được khuyến nghị trong .env.example. Giữ nguyên đường dẫn tuyệt đối là
    lựa chọn cuối, vì manifest khi đó chỉ còn đúng trên một máy.
    """
    absolute = os.path.abspath(path)
    for root in (config.AUDIO_ROOT, str(config.BASE_DIR)):
        root_abs = os.path.abspath(root)
        if absolute == root_abs or absolute.startswith(root_abs + os.sep):
            return os.path.relpath(absolute, root_abs).replace(os.sep, "/")
    return absolute.replace(os.sep, "/")


def resolve_targets(args):
    if args.file:
        if not args.recording_id:
            print("❌ Cần --recording-id khi dùng --file")
            return []
        return [(args.file, args.recording_id)]

    from sqlalchemy import text

    from backend.database.session import engine

    # --license: chỉ lấy bản ghi mang giấy phép đó. Cần cho EXP-08 — tập mẫu ngẫu
    # nhiên gần như không có bài CC0 (117/24.375), nên lớp rủi ro LOW có 0 mẫu và
    # macro-F1 bị méo.
    license_filter = getattr(args, "license", None)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT recording_id, audio_path FROM recordings r
            WHERE audio_path IS NOT NULL AND audio_path NOT LIKE 'simulated/%'
              AND (CAST(:license AS TEXT) IS NULL OR EXISTS (
                  SELECT 1 FROM rights ri
                  WHERE ri.recording_id = r.recording_id AND ri.license_type = :license))
            ORDER BY recording_id
        """), {"license": license_filter}).fetchall()

    targets, missing = [], 0
    for rec_id, audio_path in rows:
        # Phải đi qua config.resolve_audio_path: corpus thật nằm ngoài repo
        # (AUDIO_ROOT, thường ở ổ khác), nên ghép với BASE_DIR là không thấy gì.
        path = config.resolve_audio_path(str(audio_path))
        if path:
            targets.append((path, str(rec_id)))
        else:
            missing += 1
            if missing <= 5:
                print(f"⏭️  {audio_path}: không tìm thấy file trên đĩa")
    if missing > 5:
        print(f"⏭️  ...và {missing - 5} file nữa không tìm thấy "
              f"(AUDIO_ROOT = {config.AUDIO_ROOT})")
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--recording-id")
    parser.add_argument("--from-db", action="store_true")
    parser.add_argument("--append", action="store_true",
                        help="Ghi thêm vào manifest thay vì tạo mới")
    parser.add_argument("--sample", type=int, metavar="N",
                        help="Chỉ lấy ngẫu nhiên N bản ghi nguồn (seed cố định). "
                             "Corpus vài nghìn bài x 18 biến đổi là hàng chục nghìn "
                             "file, không cần và không nên.")
    parser.add_argument("--license", metavar="LICENSE_TYPE",
                        help="Chỉ lấy bản ghi mang giấy phép này (vd. CC0)")
    parser.add_argument("--composite", action="store_true", default=False,
                        help="Tạo thêm các biến thể composite cho kiểm thử multi-window")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed cho --sample, để lặp lại được tập truy vấn")
    args = parser.parse_args()

    if not args.file and not args.from_db:
        parser.error("Cần --file hoặc --from-db")

    targets = resolve_targets(args)
    if not targets:
        print("Không có audio nào để biến đổi.")
        return 1

    if args.sample and args.sample < len(targets):
        import random
        targets = sorted(random.Random(args.seed).sample(targets, args.sample))
        print(f"Lấy mẫu {len(targets)} bản ghi nguồn (seed={args.seed})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    can_mp3 = mp3_supported()
    print(f"libsndfile hỗ trợ ghi MP3: {can_mp3}")

    # Nguồn chồng âm: dùng file khác trong danh sách để phép chồng là âm thanh thật
    overlays = {}
    if len(targets) > 1:
        for index, (path, _) in enumerate(targets):
            overlays[path] = targets[(index + 1) % len(targets)]

    rows = []
    if args.append and os.path.exists(MANIFEST):
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    # Checkpoint: mỗi bản ghi nguồn sinh ra 19 file (pitch shift và time stretch
    # là phần đắt nhất), nên với hàng trăm nguồn thì đây là hàng chục phút chạy.
    # Manifest lại chỉ được ghi ở CUỐI, nên bị ngắt giữa chừng là mất toàn bộ
    # bảng kê — dù các file .wav đã nằm sẵn trên đĩa.
    checkpoint = Checkpoint("augment_audio", {
        "sources": len(targets),
        "crop_start": CROP_START,
        "crop_length": CROP_LENGTH,
        "sample_rate": SR,
        "mp3": can_mp3,
        "append": bool(args.append),
        "license": args.license,
        "composite": bool(args.composite),
    })

    for path, rec_id in targets:
        if checkpoint.has(path):
            continue
        source_rows = []
        # Điểm cắt THỰC TẾ phải được ghi lại chứ không được coi là hằng số: với
        # clip dài đúng 30 giây (FMA-small chẳng hạn) thì không thể cắt từ giây
        # 30 và ta buộc phải lùi về giây 0. Thí nghiệm dùng chính con số này để
        # ánh xạ truy vấn về trục thời gian bản gốc — đoán sai là lệch nhãn 30 giây.
        audio, crop_start = load_crop(path)
        if audio is None or len(audio) < SR:
            print(f"⏭️  {os.path.basename(path)}: quá ngắn, bỏ qua")
            continue

        overlay, overlay_rec_id = None, ""
        if path in overlays:
            # Cắt file chồng âm từ CÙNG mốc, nhưng nó có thể ngắn hơn -> tự lùi
            overlay_path, overlay_rec_id = overlays[path]
            overlay, _ = load_crop(overlay_path, crop_start)

        variants = build_variants(audio, overlay, composite=args.composite)
        stem = os.path.splitext(os.path.basename(path))[0]

        for name, data in variants.items():
            out_path = os.path.join(OUTPUT_DIR, f"{stem}__{name}.wav")
            sf.write(out_path, data, SR)
            source_rows.append({
                "query_id": str(uuid.uuid4()),
                "source_recording_id": rec_id,
                "source_file": as_stored_path(path),
                "transformation": name,
                "path": as_stored_path(out_path),
                "duration_s": round(len(data) / SR, 2),
                "crop_start_s": crop_start,
                "tempo_factor": TEMPO_FACTORS.get(name, 1.0),
                "pitch_steps": PITCH_STEPS.get(name, 0),
                "bitrate_kbps": "",
                # Bài bị trộn chồng CŨNG nằm trong CSDL: held-out phải loại cả nó,
                # nếu không thì nhận ra nó (là nhận đúng) bị đếm thành nhận nhầm.
                "overlay_source_recording_id": (
                    overlay_rec_id if name == "audio_overlay" else ""),
                "expected_match_type": "EXACT_MATCH" if name in (
                    "original_crop30s", "crop_15s", "crop_10s", "noise_snr20",
                    "gain_minus12db", "composite_padded_offset150s") else "NEAR_MATCH",
            })

        if can_mp3:
            for bitrate_name, level in MP3_LEVELS.items():
                out_path = os.path.join(OUTPUT_DIR, f"{stem}__{bitrate_name}.mp3")
                sf.write(out_path, audio, SR, format="MP3",
                         compression_level=level, bitrate_mode="CONSTANT")
                seconds = len(audio) / SR
                achieved = round(os.path.getsize(out_path) * 8 / seconds / 1000)
                source_rows.append({
                    "query_id": str(uuid.uuid4()),
                    "source_recording_id": rec_id,
                    "source_file": as_stored_path(path),
                    "transformation": bitrate_name,
                    "path": as_stored_path(out_path),
                    "duration_s": round(seconds, 2),
                    "crop_start_s": crop_start,
                    "tempo_factor": 1.0,
                    "pitch_steps": 0,
                    "bitrate_kbps": achieved,
                    "overlay_source_recording_id": "",
                    "expected_match_type": "EXACT_MATCH",
                })

        checkpoint.add(path, source_rows)
        print(f"✅ {os.path.basename(path)} -> {len(variants) + (2 if can_mp3 else 0)} truy vấn")

    for source_rows in checkpoint.records:
        rows.extend(source_rows)

    fieldnames = ["query_id", "source_recording_id", "source_file", "transformation",
                  "path", "duration_s", "crop_start_s", "tempo_factor", "pitch_steps",
                  "bitrate_kbps", "overlay_source_recording_id", "expected_match_type"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    checkpoint.close(remove=True)
    print(f"\n💾 Manifest: {MANIFEST} ({len(rows)} truy vấn)")
    if not can_mp3:
        print("⚠️  Bỏ qua biến đổi nén mp3/aac: libsndfile trên máy không ghi được MP3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
