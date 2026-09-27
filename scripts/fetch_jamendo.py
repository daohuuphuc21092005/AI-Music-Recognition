"""
Tải corpus Jamendo qua API chính thức, chọn CÓ CHỦ ĐÍCH theo từng loại giấy phép.

Vai trò khác hẳn `fetch_fma.py`. FMA cho khối lượng và sự đa dạng, nhưng phân bố
giấy phép của nó là thứ có sẵn — muốn nhiều CC-BY-ND hơn cũng chịu. Jamendo cho
lọc theo giấy phép ngay ở tầng truy vấn, nên đây là cách **chủ động lấp chỗ
trống**: nhóm nào của Rule Engine còn ít dữ liệu thật thì xin thêm đúng nhóm đó.

Điều đó quan trọng vì EXP-08 chấm bằng Macro-F1 — một lớp rủi ro chỉ có vài mẫu
sẽ kéo lệch chỉ số, và tệ hơn là cả một nhánh của Rule Engine không hề được dữ
liệu thật chạm tới.

Cần `JAMENDO_CLIENT_ID` trong `.env` (đăng ký miễn phí tại devportal.jamendo.com).
Thiếu thì script báo rõ rồi dừng, KHÔNG âm thầm bỏ qua để người chạy tưởng đã có
dữ liệu Jamendo.

Dùng:
    python scripts/fetch_jamendo.py --per-license 60
    python scripts/fetch_jamendo.py --per-license 60 --dry-run   # chỉ xem, không tải
"""
import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from backend.services.license_mapping import normalize_license

API = "https://api.jamendo.com/v3.0/tracks/"
PAGE_LIMIT = 200               # trần của Jamendo cho mỗi lần gọi
STAGING_CSV = os.path.join(config.DATA_DIR, "corpus_jamendo.csv")

# Bộ lọc `license_cc` của Jamendo nhận các cờ thành phần, không nhận nguyên mã.
# Ánh xạ từ mã CC chuẩn sang bộ cờ tương ứng.
LICENSE_FILTERS = {
    "by": ["by"],
    "by-sa": ["by", "sa"],
    "by-nd": ["by", "nd"],
    "by-nc": ["by", "nc"],
    "by-nc-sa": ["by", "nc", "sa"],
    "by-nc-nd": ["by", "nc", "nd"],
}


def fetch_page(client_id: str, license_code: str, offset: int) -> list:
    params = {
        "client_id": client_id,
        "format": "json",
        "limit": PAGE_LIMIT,
        "offset": offset,
        "audioformat": "mp32",
        "include": "licenses musicinfo",
        "license_cc": "+".join(LICENSE_FILTERS[license_code]),
        "order": "popularity_total",
    }
    url = API + "?" + urllib.parse.urlencode(params, safe="+")

    with urllib.request.urlopen(url, timeout=45) as response:
        payload = json.load(response)

    headers = payload.get("headers", {})
    if headers.get("status") != "success":
        raise RuntimeError(f"Jamendo trả lỗi: {headers.get('error_message') or headers}")
    return payload.get("results", [])


def download_audio(url: str, dest: str) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        return True
    try:
        with urllib.request.urlopen(url, timeout=90) as response, open(dest, "wb") as f:
            while True:
                chunk = response.read(1 << 18)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        print(f"      lỗi tải: {type(e).__name__}: {str(e)[:60]}")
        if os.path.exists(dest):
            os.remove(dest)
        return False
    return os.path.getsize(dest) > 10_000


def collect(client_id: str, license_code: str, wanted: int) -> list:
    """Gọi API cho tới khi đủ `wanted` track ĐÚNG loại giấy phép đang xin."""
    collected, offset, empty_pages = [], 0, 0

    while len(collected) < wanted and empty_pages < 2:
        try:
            results = fetch_page(client_id, license_code, offset)
        except Exception as e:
            print(f"    dừng ở offset {offset}: {e}")
            break
        if not results:
            empty_pages += 1
            offset += PAGE_LIMIT
            continue

        empty_pages = 0
        for item in results:
            # Bộ lọc của Jamendo có thể nới hơn ta muốn, nên vẫn phải tự xác minh
            # bằng chính URL giấy phép mà nó trả về.
            info = normalize_license(license_url=item.get("license_ccurl", ""))
            if info["license_code"] != license_code:
                continue
            collected.append((item, info))
            if len(collected) >= wanted:
                break

        offset += PAGE_LIMIT
        time.sleep(0.3)        # lịch sự với API công cộng

    return collected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-license", type=int, default=60,
                        help="Số track cho MỖI loại giấy phép (6 loại)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ gọi API và thống kê, không tải audio")
    parser.add_argument("--licenses", default=",".join(LICENSE_FILTERS),
                        help="Danh sách mã CC cần lấy, phân tách bằng dấu phẩy")
    args = parser.parse_args()

    client_id = os.getenv("JAMENDO_CLIENT_ID", "").strip()
    if not client_id:
        print("❌ Thiếu JAMENDO_CLIENT_ID.\n"
              "   Đăng ký miễn phí tại https://devportal.jamendo.com/ rồi thêm vào .env:\n"
              "       JAMENDO_CLIENT_ID=<client_id_cua_ban>\n"
              "   (Bỏ qua nguồn Jamendo thì corpus chỉ có phân bố giấy phép sẵn có\n"
              "    của FMA, không chủ động lấp được nhóm còn thiếu.)")
        return 1

    codes = [c.strip() for c in args.licenses.split(",") if c.strip() in LICENSE_FILTERS]
    if not codes:
        print(f"❌ Không có mã hợp lệ. Hợp lệ: {', '.join(LICENSE_FILTERS)}")
        return 1

    audio_dir = os.path.join(config.AUDIO_ROOT, "jamendo")
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Thư mục corpus : {audio_dir}")
    print(f"Xin {args.per_license} track cho mỗi loại: {', '.join(codes)}")

    rows = []
    for code in codes:
        print(f"\n[{code}] đang gọi API...")
        found = collect(client_id, code, args.per_license)
        print(f"  API trả về {len(found)} track đúng loại {code}")

        for item, info in found:
            track_id = str(item.get("id"))
            filename = f"{track_id}.mp3"
            relative = f"jamendo/{filename}"

            if not args.dry_run:
                url = item.get("audiodownload") or item.get("audio")
                if not url or not download_audio(url, os.path.join(audio_dir, filename)):
                    continue

            rows.append({
                "source_dataset": "Jamendo",
                "source_track_id": track_id,
                "title": (item.get("name") or "").strip(),
                "artist": (item.get("artist_name") or "").strip(),
                "album": (item.get("album_name") or "").strip(),
                "composer": "",
                "duration": item.get("duration") or "",
                "year": (item.get("releasedate") or "")[:4],
                "audio_path": relative,
                "license_title": info["license_code"],
                "license_url": item.get("license_ccurl", ""),
                "source_url": item.get("shorturl") or item.get("prourl") or "",
                **{k: v for k, v in info.items() if k != "matched_on"},
            })
        print(f"  đã lấy {len([r for r in rows if r['license_code'] == code])} track")

    if not rows:
        print("\n❌ Không lấy được track nào.")
        return 1

    print(f"\nTổng: {len(rows)} track")
    for license_type, count in Counter(r["license_type"] for r in rows).most_common():
        print(f"  {license_type:<16} {count:>5}")

    if args.dry_run:
        print("\n--dry-run: không ghi file, không tải audio.")
        return 0

    fieldnames = ["source_dataset", "source_track_id", "title", "artist", "album",
                  "composer", "duration", "year", "audio_path", "license_title",
                  "license_url", "source_url", "license_type", "license_code",
                  "copyright_status", "attribution_required", "commercial_use_allowed",
                  "modification_allowed", "monetization_allowed",
                  "recording_public_domain"]
    os.makedirs(os.path.dirname(STAGING_CSV), exist_ok=True)
    with io.open(STAGING_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n💾 {STAGING_CSV} ({len(rows)} track)")
    print("Bước tiếp: python scripts/ingest_corpus.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
