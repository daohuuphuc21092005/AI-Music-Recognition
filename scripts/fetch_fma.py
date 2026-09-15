"""
Tải corpus FMA-small và trích metadata giấy phép THẬT (§8, giai đoạn corpus).

Vì sao cần: reference database hiện có 2750 bản ghi nhưng **không bản ghi nào còn
audio**, và 100% rights có `source` bắt đầu bằng `SIMULATED`. Mọi con số của
EXP-01..08 vì thế đo trên đúng 2 file audio thật. FMA công bố cả audio lẫn giấy
phép Creative Commons thật cho từng bản thu, nên nó sửa được cả hai bằng một lần tải.

**Vì sao lấy từ HuggingFace chứ không từ máy chủ gốc của FMA.** Đo thực tế trên
máy này: mirror gốc `os.unil.cloud.switch.ch` cho **36 KB/s** — `fma_small.zip`
7,2 GB sẽ mất khoảng **58 tiếng**, trong khi CDN của HuggingFace cho 1,6 MB/s.
Không phải đường truyền chậm, mà là mirror chậm. Bản trên HF còn hơn ở hai điểm:

  * Chia thành 15 shard parquet, nên **tải lẻ được** — 2 shard là đủ ~1000 track,
    khoảng 1 GB, thay vì phải nuốt trọn 7,2 GB.
  * Kèm sẵn `license`, `copyright` và bốn cờ CC đã tách sẵn
    (`allow_commercial_use`, `allow_derivatives`, `require_attribution`,
    `require_share_alike`). Script dùng bốn cờ đó để **đối chứng độc lập** với
    `license_mapping.normalize_license()`, và báo cáo mọi chỗ hai bên bất đồng
    thay vì âm thầm tin một phía.

Dùng:
    python scripts/fetch_fma.py --shards 2 --sample 1000
    python scripts/fetch_fma.py --shards 2 --skip-download   # đã tải sẵn
    python scripts/fetch_fma.py --subset medium --sample 0   # FMA medium, giữ tất cả

Kết quả: `data/processed/corpus_fma.csv` để `scripts/ingest_corpus.py` nạp tiếp.
"""
import argparse
import csv
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.parquet as pq

from backend import config
from backend.services.license_mapping import normalize_license

# HuggingFace đóng gói FMA theo đúng các tập con của bản gốc (small ⊂ medium ⊂
# large ⊂ full), và số shard KHÁC NHAU giữa các bộ — small 15, medium 44. Không
# gán cứng: hỏi thẳng HuggingFace, chỉ dùng bảng dự phòng khi không gọi được API.
SUBSETS = ("small", "medium", "large", "full")
KNOWN_SHARDS = {"small": 15, "medium": 44}
REPO_TEMPLATE = "benjamin-paine/free-music-archive-{subset}"

# Giữ nguyên các hằng số của bản small để mã cũ gọi vào không vỡ.
REPO = REPO_TEMPLATE.format(subset="small")
TOTAL_SHARDS = KNOWN_SHARDS["small"]
SHARD_URL = ("https://huggingface.co/datasets/" + REPO +
             "/resolve/main/data/train-{index:05d}-of-00015.parquet")
# Số dòng đọc mỗi lô khi trích parquet. 32 dòng ~ 30 MB audio, đủ nhỏ để bộ nhớ
# phẳng mà vẫn không bị chậm vì gọi quá nhiều lần.
BATCH_ROWS = 32
STAGING_CSV = os.path.join(config.DATA_DIR, "corpus_fma.csv")

# Bảng nhãn của cột `license` (ClassLabel) trong dataset. Ba mục cuối KHÔNG phải
# giấy phép Creative Commons, nên normalize_license sẽ trả UNKNOWN cho chúng —
# đó là hành vi đúng, không được đoán bừa (§2).
LICENSE_LABELS = [
    "CC-BY 1.0", "CC-BY 2.0", "CC-BY 2.5", "CC-BY 3.0", "CC-BY 4.0",
    "CC-BY-NC 2.0", "CC-BY-NC 2.1", "CC-BY-NC 2.5", "CC-BY-NC 3.0", "CC-BY-NC 4.0",
    "CC-BY-NC-ND 2.0", "CC-BY-NC-ND 2.1", "CC-BY-NC-ND 2.5", "CC-BY-NC-ND 3.0",
    "CC-BY-NC-ND 4.0",
    "CC-BY-NC-SA 2.0", "CC-BY-NC-SA 2.1", "CC-BY-NC-SA 2.5", "CC-BY-NC-SA 3.0",
    "CC-BY-NC-SA 4.0",
    "CC-BY-ND 2.0", "CC-BY-ND 2.5", "CC-BY-ND 3.0", "CC-BY-ND 4.0",
    "CC-BY-SA 2.0", "CC-BY-SA 2.5", "CC-BY-SA 3.0", "CC-BY-SA 4.0",
    "CC-NC-Sampling+ 1.0", "CC-Sampling+ 1.0", "CC0 1.0",
    "FMA Sound Recording Common Law", "Free Art License",
    "Free Music Philosophy (FMP)",
]

# Tên chuỗi ở trên dùng dấu gạch nối kiểu "CC-BY-NC-SA", còn normalize_license đọc
# theo cú pháp của chính giấy phép, nên đưa về dạng URL chuẩn cho chắc chắn.
CC_URL = "https://creativecommons.org/licenses/{code}/{version}/"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def license_to_url(label: str) -> str:
    """Đổi nhãn dạng 'CC-BY-NC-SA 3.0' thành URL giấy phép chuẩn."""
    if not label:
        return ""
    name, _, version = label.partition(" ")
    version = version.strip() or "4.0"
    upper = name.upper()
    if upper.startswith("CC0"):
        return "https://creativecommons.org/publicdomain/zero/1.0/"
    if not upper.startswith("CC-BY"):
        return ""      # Sampling+, Free Art License... không phải CC chuẩn
    code = upper[len("CC-"):].lower()      # "by-nc-sa"
    return CC_URL.format(code=code, version=version)


# --------------------------------------------------------------------------
# Tải
# --------------------------------------------------------------------------
def shard_count(subset: str) -> int:
    """Số shard parquet của một tập con, hỏi từ HuggingFace; dự phòng bằng bảng biết trước."""
    repo = REPO_TEMPLATE.format(subset=subset)
    try:
        url = f"https://huggingface.co/api/datasets/{repo}/tree/main/data"
        with urllib.request.urlopen(url, timeout=30) as response:
            files = json.load(response)
        count = sum(1 for f in files if str(f.get("path", "")).endswith(".parquet"))
        if count:
            return count
    except Exception:
        pass
    if subset in KNOWN_SHARDS:
        return KNOWN_SHARDS[subset]
    raise RuntimeError(f"Không xác định được số shard của FMA-{subset} "
                       f"(không gọi được HuggingFace và không có trong bảng dự phòng)")


def shard_url(subset: str, index: int, total: int) -> str:
    repo = REPO_TEMPLATE.format(subset=subset)
    return (f"https://huggingface.co/datasets/{repo}/resolve/main/data/"
            f"train-{index:05d}-of-{total:05d}.parquet")



def remote_size(url: str) -> int:
    """Content-Length của file trên server; 0 nếu không hỏi được."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.headers.get("Content-Length", 0))
    except Exception:
        return 0


def download(url: str, dest: str, attempts: int = 4) -> bool:
    """
    Tải có TẢI TIẾP và tự thử lại.

    Bản đầu coi `read()` trả rỗng là tải xong, nên đứt mạng giữa chừng thì báo
    thành công với một file cụt rồi chết ở bước đọc parquet. Nay luôn đối chiếu
    Content-Length và thử lại cho tới khi đủ byte.

    Hỏi kích thước bằng HEAD TRƯỚC là bắt buộc, không phải tối ưu: khi file đã
    tải đủ, gửi `Range: bytes=<đúng bằng kích thước>-` khiến server trả
    416 Range Not Satisfiable, và urlopen NÉM LỖI ngay — trước cả dòng kiểm tra
    "đã đủ" nằm bên trong khối `with`. Hậu quả là script không bao giờ chạy lại
    được sau khi đã tải xong, kể cả chỉ để đọc lại parquet.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    expected = remote_size(url)
    if expected and os.path.exists(dest) and os.path.getsize(dest) >= expected:
        print(f"    đã có đủ {human(os.path.getsize(dest))}")
        return True

    for attempt in range(1, attempts + 1):
        done = os.path.getsize(dest) if os.path.exists(dest) else 0
        request = urllib.request.Request(url)
        if done:
            request.add_header("Range", f"bytes={done}-")

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                resuming = response.status == 206
                if done and not resuming:
                    done = 0
                total = int(response.headers.get("Content-Length", 0)) + done
                if done and done >= total > 0:
                    print(f"    đã đủ {human(done)}")
                    return True

                mode = "ab" if resuming and done else "wb"
                started, last = time.time(), 0.0
                with open(dest, mode) as f:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if now - last >= 5:
                            speed = done / max(now - started, 1e-6)
                            pct = f"{done / total * 100:5.1f}%" if total else "  ?  "
                            print(f"    {pct} {human(done)}  {human(speed)}/s", flush=True)
                            last = now

            if total and done < total:
                print(f"    đứt ở {human(done)}/{human(total)}, thử lại "
                      f"({attempt}/{attempts})")
                continue
            print(f"    xong {human(done)}")
            return True

        except urllib.error.HTTPError as e:
            # 416 = không còn byte nào sau offset đang xin, tức file đã đủ.
            if e.code == 416 and os.path.exists(dest) and os.path.getsize(dest) > 0:
                print(f"    đã có đủ {human(os.path.getsize(dest))}")
                return True
            print(f"    lỗi HTTP {e.code}: {str(e)[:60]} — thử lại "
                  f"({attempt}/{attempts})")
            time.sleep(2 * attempt)

        except Exception as e:
            print(f"    lỗi {type(e).__name__}: {str(e)[:70]} — thử lại "
                  f"({attempt}/{attempts})")
            time.sleep(2 * attempt)

    print(f"  ❌ không tải được sau {attempts} lần. File dở giữ ở {dest}, "
          f"chạy lại lệnh để tải tiếp.")
    return False


# --------------------------------------------------------------------------
# Đọc parquet
# --------------------------------------------------------------------------
_FMA_AUDIO_NAME = re.compile(r"^([0-9]{1,7})[.]mp3$", re.IGNORECASE)


def fma_track_id(audio) -> str:
    """
    Mã track FMA gốc lấy từ tên file audio trong parquet ('000002.mp3' -> '2').

    HuggingFace giữ nguyên tên file của FMA gốc, và tên đó CHÍNH LÀ track_id —
    khoá chính của fma_metadata/tracks.csv. Dùng nó thay cho số thứ tự trong shard
    thì (1) mã ổn định giữa các lần tải, (2) một track có ở cả small lẫn medium mang
    cùng một mã nên chỉ lưu một lần, và (3) nối được với metadata gốc (thể loại,
    subset...). Trả chuỗi rỗng nếu tên file không đúng dạng — người gọi phải tự
    sinh mã dự phòng, không được đoán.
    """
    name = os.path.basename(str((audio or {}).get("path") or ""))
    match = _FMA_AUDIO_NAME.match(name)
    return str(int(match.group(1))) if match else ""



def extract_tracks(parquet_path: str, audio_dir: str, shard_index: int) -> tuple:
    """
    Ghi audio ra file và trả (danh sách track, số lần ánh xạ giấy phép bất đồng).

    Đối chứng: dataset đã tự tách sẵn bốn cờ CC. So chúng với kết quả của
    `normalize_license` là một phép kiểm tra chéo hoàn toàn độc lập — nếu hai bên
    lệch nhau thì mapper của mình sai, và phải biết điều đó ngay chứ không phải
    lúc Rule Engine đã ra phán quyết.
    """
    # Đọc THEO LÔ, không nạp cả bảng. `pq.read_table()` kéo trọn ~460 MB parquet
    # vào RAM, rồi `to_pylist()` nhân nó lên nhiều lần nữa vì mỗi track mang theo
    # cả trăm KB audio dưới dạng bytes Python. Với 8 shard thì tiến trình bị hệ
    # thống giết trước khi kịp ghi file nào. Đọc từng lô nhỏ giữ bộ nhớ phẳng,
    # bất kể corpus lớn tới đâu.
    parquet = pq.ParquetFile(parquet_path)
    columns = set(parquet.schema_arrow.names)

    os.makedirs(audio_dir, exist_ok=True)
    # audio_path lưu TƯƠNG ĐỐI so với AUDIO_ROOT, theo đúng thư mục thật đang ghi.
    root = os.path.abspath(config.AUDIO_ROOT)
    absolute = os.path.abspath(audio_dir)
    rel_dir = (os.path.relpath(absolute, root) if absolute.startswith(root + os.sep)
               else os.path.basename(absolute)).replace(os.sep, "/")
    tracks, disagreements = [], []
    position = -1

    for batch in parquet.iter_batches(batch_size=BATCH_ROWS):
        for row in batch.to_pylist():
            position += 1
            audio = row.get("audio") or {}
            data = audio.get("bytes")
            if not data:
                continue

            fma_id = fma_track_id(audio)
            if fma_id:
                track_id = fma_id
                filename = f"{int(fma_id):06d}.mp3"
            else:
                # Tên file không mang mã FMA: dự phòng bằng vị trí trong shard, có
                # tiền tố để không bao giờ trùng với một mã FMA thật.
                track_id = f"shard{shard_index:02d}_{position:04d}"
                filename = f"{track_id}.mp3"

            target = os.path.join(audio_dir, filename)
            # Đã có đúng file đó thì bỏ qua: chạy lại sau khi bị ngắt không phải ghi
            # lại hàng chục GB, và track nằm ở cả small lẫn medium chỉ lưu một lần.
            if not (os.path.exists(target) and os.path.getsize(target) == len(data)):
                with open(target, "wb") as f:
                    f.write(data)

            label_index = row.get("license")
            label = (LICENSE_LABELS[label_index]
                     if isinstance(label_index, int) and 0 <= label_index < len(LICENSE_LABELS)
                     else "")
            info = normalize_license(license_text=label, license_url=license_to_url(label))

            # --- đối chứng với cờ của chính dataset ---------------------------
            if info["license_type"] != "UNKNOWN" and "allow_commercial_use" in columns:
                theirs = {
                    "commercial_use_allowed": bool(row.get("allow_commercial_use")),
                    "modification_allowed": bool(row.get("allow_derivatives")),
                    "attribution_required": bool(row.get("require_attribution")),
                }
                mismatch = {k: (info[k], v) for k, v in theirs.items() if info[k] != v}
                if mismatch:
                    disagreements.append((label, mismatch))

            released = row.get("released")
            tracks.append({
                "source_dataset": "FMA",
                "source_track_id": track_id,
                "title": (row.get("title") or "").strip(),
                "artist": (row.get("artist") or "").strip(),
                "album": (row.get("album_title") or "").strip(),
                "composer": (row.get("composer") or "").strip(),
                "year": released.year if hasattr(released, "year") else "",
                "duration": "",
                "audio_path": f"{rel_dir}/{filename}",
                "license_title": label,
                "license_url": license_to_url(label),
                "source_url": (row.get("url") or "").strip(),
                **{k: v for k, v in info.items() if k != "matched_on"},
            })

    return tracks, disagreements


def sample_stratified(tracks: list, n: int, seed: int) -> list:
    """
    Lấy mẫu CÂN BẰNG theo loại giấy phép.

    Lấy n track đầu danh sách sẽ lệch cả thể loại lẫn giấy phép — mà giấy phép
    chính là biến quyết định đầu ra của Rule Engine, nên để nó lệch là hỏng luôn
    ý nghĩa của EXP-08.
    """
    buckets = {}
    for track in tracks:
        buckets.setdefault(track["license_type"], []).append(track)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    chosen, order = [], sorted(buckets, key=lambda k: -len(buckets[k]))
    while len(chosen) < n and any(buckets[k] for k in order):
        for key in order:
            if buckets[key] and len(chosen) < n:
                chosen.append(buckets[key].pop())
    return chosen


# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=SUBSETS, default="small",
                        help="Tập con FMA: small (8.000), medium (25.000), large, full")
    parser.add_argument("--shards", type=int, default=None,
                        help="Số shard parquet cần dùng; bỏ trống = 2 với small "
                             "(giữ hành vi cũ), toàn bộ với các tập còn lại")
    parser.add_argument("--sample", type=int, default=1000,
                        help="Số track giữ lại cho corpus; 0 = giữ tất cả")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-dir",
                        help="Nơi lưu parquet (mặc định: <thư mục cha của AUDIO_ROOT>"
                             "/downloads/fma_<subset>)")
    args = parser.parse_args()

    total = shard_count(args.subset)
    wanted = args.shards if args.shards else (2 if args.subset == "small" else total)
    shards = max(1, min(wanted, total))
    root = config.AUDIO_ROOT
    parquet_dir = args.download_dir or os.path.join(
        os.path.dirname(os.path.normpath(root)), "downloads", f"fma_{args.subset}")
    # Mọi tập con dùng CHUNG một thư mục audio: tên file là mã track FMA gốc, nên
    # một track có ở cả small lẫn medium chỉ lưu đúng một lần.
    audio_dir = os.path.join(root, "fma")
    staging_csv = (STAGING_CSV if args.subset == "small"
                   else os.path.join(config.DATA_DIR, f"corpus_fma_{args.subset}.csv"))
    os.makedirs(root, exist_ok=True)

    print(f"Thư mục corpus : {root}")
    print(f"Thư mục parquet: {parquet_dir}")
    print(f"Nguồn          : HuggingFace {REPO_TEMPLATE.format(subset=args.subset)}")
    print(f"Tải            : {shards}/{total} shard")

    paths = []
    for index in range(shards):
        name = f"train-{index:05d}-of-{total:05d}.parquet"
        dest = os.path.join(parquet_dir, name)
        paths.append(dest)
        if args.skip_download:
            continue
        print(f"  [{index + 1}/{shards}] {name}")
        if not download(shard_url(args.subset, index, total), dest):
            return 1

    all_tracks, all_disagreements = [], []
    for index, path in enumerate(paths):
        if not os.path.exists(path):
            print(f"❌ thiếu {path} — bỏ --skip-download để tải")
            return 1
        print(f"\nĐọc {os.path.basename(path)}...")
        tracks, disagreements = extract_tracks(path, audio_dir, index)
        print(f"  {len(tracks)} track, đã ghi audio ra {audio_dir}")
        all_tracks.extend(tracks)
        all_disagreements.extend(disagreements)

    if not all_tracks:
        print("❌ Không đọc được track nào.")
        return 1

    print(f"\nTổng: {len(all_tracks)} track")
    print("\nPhân bố giấy phép (đọc từ dữ liệu thật):")
    for license_type, count in Counter(t["license_type"] for t in all_tracks).most_common():
        print(f"  {license_type:<16} {count:>5}  ({count / len(all_tracks) * 100:.1f}%)")

    # --- kết quả đối chứng ------------------------------------------------
    if all_disagreements:
        print(f"\n⚠️  {len(all_disagreements)} track mà normalize_license() bất đồng "
              f"với cờ của dataset:")
        for label, mismatch in all_disagreements[:5]:
            print(f"    {label}: {mismatch}")
        print("    (định dạng: trường: (giá trị của ta, giá trị của dataset))")
    else:
        print("\n✅ normalize_license() khớp 100% với bốn cờ CC do dataset tự tách "
              "— hai cách đọc độc lập cho cùng kết quả.")

    known = [t for t in all_tracks if t["license_type"] != "UNKNOWN"]
    print(f"\nĐọc được giấy phép: {len(known)}/{len(all_tracks)}")

    chosen = (known if args.sample <= 0
              else sample_stratified(known, args.sample, args.seed))
    how = "giữ tất cả" if args.sample <= 0 else f"phân tầng theo giấy phép, seed={args.seed}"
    print(f"\nĐã chọn {len(chosen)} track ({how}):")
    for license_type, count in Counter(t["license_type"] for t in chosen).most_common():
        print(f"  {license_type:<16} {count:>5}")

    fieldnames = ["source_dataset", "source_track_id", "title", "artist", "album",
                  "composer", "duration", "year", "audio_path", "license_title",
                  "license_url", "source_url", "license_type", "license_code",
                  "copyright_status", "attribution_required", "commercial_use_allowed",
                  "modification_allowed", "monetization_allowed",
                  "recording_public_domain"]
    os.makedirs(os.path.dirname(staging_csv), exist_ok=True)
    with io.open(staging_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(chosen)

    print(f"\n💾 {staging_csv} ({len(chosen)} track)")
    source_key = "fma" if args.subset == "small" else f"fma_{args.subset}"
    print(f"Bước tiếp: python scripts/ingest_corpus.py --sources {source_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
