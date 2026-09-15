"""
Nạp corpus audio THẬT vào các file master, thay cho dữ liệu mô phỏng (§8).

Chạy sau `fetch_fma.py` và/hoặc `fetch_jamendo.py`.

Quyết định thiết kế quan trọng nhất ở đây, cần đọc kỹ trước khi sửa:

**Bản ghi không có audio thì KHÔNG được có fingerprint hay embedding.** Dữ liệu
mô phỏng hiện tại vi phạm đúng điều đó: 2650 fingerprint nhưng chỉ 2016 giá trị
duy nhất, vì fingerprint của `test.mp3` bị gán cho 38 recording_id khác nhau.
Hậu quả là ground truth mơ hồ tới mức EXP-01/04/08 phải chấm điểm theo "lớp tương
đương fingerprint" thay vì theo bản ghi. Đó là một cái nạng chống cho dữ liệu
hỏng, không phải một phương pháp.

Nên sau khi nạp:

  * Bản ghi có audio thật  -> có đủ recording + composition + rights + fingerprint,
    và embedding sinh sau bằng `build_embeddings.py`.
  * Bản ghi chỉ có rights  -> giữ lại ĐÚNG ba nhóm không lấy được từ nguồn mở
    (AUDIO_LIBRARY, CREATOR_MUSIC, CONTENT_ID) để Rule Engine vẫn có dữ liệu cho
    cả 5 nhánh, nhưng **không fingerprint, không embedding**. Chúng giữ tiền tố
    `SIMULATED` ở `source` để `compute_rights_confidence` trừ điểm đúng.
  * Bản ghi mô phỏng thuộc nhóm CC/PUBLIC_DOMAIN bị **loại bỏ**, vì corpus thật
    đã phủ các nhóm đó bằng giấy phép thật.

Về §2 (composition != recording): FMA và Jamendo chỉ công bố thông tin BẢN THU,
không nói gì về tình trạng miền công cộng của TÁC PHẨM. Vì vậy mọi composition
sinh ra ở đây đều mang `public_domain_status = unknown` — kể cả khi bản thu là
CC0. Suy "bản thu CC0 nên tác phẩm cũng PD" chính là điều §2 cấm.

Dùng:
    python scripts/ingest_corpus.py --dry-run
    python scripts/ingest_corpus.py
"""
import argparse
import csv
import io
import os
import sys
import uuid
from collections import Counter
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from experiments.common import Checkpoint
from backend.services.fingerprint_service import (
    AudioFingerprintError,
    FingerprintBackendUnavailable,
    extract_query_fingerprint,
)

csv.field_size_limit(10 ** 9)

STAGING = {
    "fma": os.path.join(config.DATA_DIR, "corpus_fma.csv"),
    "fma_medium": os.path.join(config.DATA_DIR, "corpus_fma_medium.csv"),
    "jamendo": os.path.join(config.DATA_DIR, "corpus_jamendo.csv"),
}
MASTERS = {name: os.path.join(config.DATA_DIR, f"{name}_master.csv")
           for name in ("compositions", "metadata", "rights", "fingerprints",
                        "embeddings")}

# Ba nhóm không có nguồn mở nào cấp được: thư viện âm thanh của nền tảng, nhạc
# mua bản quyền của nhà sáng tạo, và yêu sách Content ID. Giữ lại dạng metadata.
KEEP_SIMULATED_LICENSES = {"AUDIO_LIBRARY", "CREATOR_MUSIC", "CONTENT_ID", "COMMERCIAL"}

FIELDS = {
    "compositions": ["composition_id", "title", "composer", "year",
                     "public_domain_status", "source", "verified_at"],
    "metadata": ["recording_id", "composition_id", "title", "artist", "album",
                 "release_year", "duration", "source_dataset", "source_track_id",
                 "audio_path", "metadata_verified"],
    "rights": ["rights_id", "recording_id", "composition_id", "license_type",
               "copyright_status", "attribution_required", "commercial_use_allowed",
               "modification_allowed", "monetization_allowed", "revenue_share_required",
               "policy_action", "license_purchased", "revenue_share_agreed",
               "recording_public_domain", "territory", "platform", "valid_from",
               "valid_until", "source", "source_url", "verified_at"],
    "fingerprints": ["fingerprint_id", "recording_id", "algorithm", "fingerprint",
                     "duration", "created_at"],
}


def read_csv(path: str) -> list:
    if not os.path.exists(path):
        return []
    with io.open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, fieldnames: list, rows: list) -> None:
    with io.open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_staging(sources: list) -> list:
    tracks = []
    for name in sources:
        rows = read_csv(STAGING[name])
        if not rows:
            print(f"  ⏭️  bỏ qua {name}: chưa có {STAGING[name]}")
            continue
        print(f"  {name}: {len(rows)} track")
        tracks.extend(rows)
    return tracks


def build_rows(tracks: list, now: str, compute_fingerprints: bool) -> tuple:
    """
    Sinh 4 nhóm dòng master từ danh sách track đã tải.

    Có checkpoint vì bước này gọi `fpcalc` một lần cho MỖI bản ghi: với corpus
    4.000 bài là hơn 13 phút chạy liên tục, và trên máy thiếu RAM tiến trình có
    thể bị giết giữa chừng. Không có checkpoint thì mỗi lần bị ngắt là mất sạch
    và corpus lớn không bao giờ nạp xong.

    Khoá là `audio_path` (ổn định giữa các lần chạy), còn UUID được sinh MỘT LẦN
    rồi lưu cùng: sinh lại UUID mới ở lần chạy sau sẽ khiến fingerprint và rights
    trỏ tới những bản ghi khác nhau.
    """
    checkpoint = Checkpoint("ingest_corpus", {
        "tracks": len(tracks),
        "compute_fingerprints": compute_fingerprints,
        "audio_root": str(config.AUDIO_ROOT),
    })

    compositions, recordings, rights, fingerprints = [], [], [], []
    skipped_missing_audio, skipped_fingerprint = 0, 0

    for index, track in enumerate(tracks, start=1):
        if checkpoint.has(track["audio_path"]):
            continue

        absolute = config.resolve_audio_path(track["audio_path"])
        if not absolute:
            skipped_missing_audio += 1
            continue

        recording_id = str(uuid.uuid4())
        composition_id = str(uuid.uuid4())
        source = f"{track['source_dataset']} (giấy phép do nguồn công bố)"

        duration, fingerprint = "", None
        if compute_fingerprints:
            try:
                duration, fingerprint = extract_query_fingerprint(absolute)
            except FingerprintBackendUnavailable:
                raise
            except (AudioFingerprintError, Exception) as e:
                skipped_fingerprint += 1
                print(f"    fingerprint hỏng, bỏ qua {track['audio_path']}: "
                      f"{type(e).__name__}")
                continue

        entry_composition = {
            "composition_id": composition_id,
            "title": track.get("title", ""),
            "composer": track.get("composer") or track.get("artist", ""),
            "year": track.get("year", ""),
            # §2: nguồn không nói gì về PD của TÁC PHẨM -> không được suy từ bản thu
            "public_domain_status": "unknown",
            "source": source,
            "verified_at": now,
        }
        entry_recording = {
            "recording_id": recording_id,
            "composition_id": composition_id,
            "title": track.get("title", ""),
            "artist": track.get("artist", ""),
            "album": track.get("album", ""),
            "release_year": track.get("year", ""),
            "duration": round(float(duration), 2) if duration else track.get("duration", ""),
            "source_dataset": track["source_dataset"],
            "source_track_id": track.get("source_track_id", ""),
            "audio_path": track["audio_path"],
            "metadata_verified": True,
        }
        entry_rights = {
            "rights_id": str(uuid.uuid4()),
            "recording_id": recording_id,
            "composition_id": composition_id,
            "license_type": track["license_type"],
            "copyright_status": track["copyright_status"],
            "attribution_required": track["attribution_required"],
            "commercial_use_allowed": track["commercial_use_allowed"],
            "modification_allowed": track["modification_allowed"],
            "monetization_allowed": track["monetization_allowed"],
            "revenue_share_required": False,
            "policy_action": "NONE",
            "license_purchased": False,
            "revenue_share_agreed": False,
            "recording_public_domain": track["recording_public_domain"],
            "territory": "GLOBAL",
            "platform": "ALL",
            "valid_from": "",
            "valid_until": "",
            # KHÔNG có tiền tố SIMULATED: đây là giấy phép thật do nguồn công bố
            "source": source,
            "source_url": track.get("license_url") or track.get("source_url", ""),
            "verified_at": now,
        }
        entry_fingerprint = None
        if fingerprint is not None:
            entry_fingerprint = {
                "fingerprint_id": str(uuid.uuid4()),
                "recording_id": recording_id,
                "algorithm": "chromaprint",
                "fingerprint": fingerprint.decode() if isinstance(fingerprint, bytes)
                               else fingerprint,
                "duration": round(float(duration), 2) if duration else "",
                "created_at": now,
            }

        # Ghi cả bốn dòng của track này xuống đĩa cùng lúc: chúng dùng chung
        # recording_id / composition_id nên không bao giờ được lưu nửa vời.
        checkpoint.add(track["audio_path"], {
            "composition": entry_composition,
            "recording": entry_recording,
            "rights": entry_rights,
            "fingerprint": entry_fingerprint,
        })

        if index % 100 == 0:
            print(f"    ...{index}/{len(tracks)}", flush=True)

    for entry in checkpoint.records:
        compositions.append(entry["composition"])
        recordings.append(entry["recording"])
        rights.append(entry["rights"])
        if entry.get("fingerprint"):
            fingerprints.append(entry["fingerprint"])

    checkpoint.close()

    if skipped_missing_audio:
        print(f"  ⏭️  {skipped_missing_audio} track bỏ qua vì không thấy file audio")
    if skipped_fingerprint:
        print(f"  ⏭️  {skipped_fingerprint} track bỏ qua vì không sinh được fingerprint")
    return compositions, recordings, rights, fingerprints


def keep_simulated(old_recordings: list, old_rights: list, old_compositions: list):
    """Giữ lại các bản ghi mô phỏng thuộc nhóm không lấy được từ nguồn mở."""
    rights_by_recording = {r["recording_id"]: r for r in old_rights}
    keep_rec, keep_rights, keep_comp_ids = [], [], set()

    for recording in old_recordings:
        right = rights_by_recording.get(recording["recording_id"])
        if not right or right.get("license_type") not in KEEP_SIMULATED_LICENSES:
            continue
        # Bản ghi này không có audio -> xoá audio_path để không ai tưởng là có
        keep_rec.append({**recording, "audio_path": ""})
        keep_rights.append(right)
        keep_comp_ids.add(recording["composition_id"])

    keep_comp = [c for c in old_compositions if c["composition_id"] in keep_comp_ids]
    return keep_comp, keep_rec, keep_rights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="fma,jamendo")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ thống kê, không ghi file master")
    parser.add_argument("--no-fingerprints", action="store_true",
                        help="Bỏ qua bước sinh fingerprint (chỉ để thử nhanh)")
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip() in STAGING]
    print("Đọc corpus đã tải:")
    tracks = load_staging(sources)
    if not tracks:
        print("\n❌ Không có corpus nào. Chạy trước:\n"
              "     python scripts/fetch_fma.py --shards 2\n"
              "     python scripts/fetch_jamendo.py --per-license 60")
        return 1

    print(f"\nTổng {len(tracks)} track. Phân bố giấy phép:")
    for license_type, count in Counter(t["license_type"] for t in tracks).most_common():
        print(f"  {license_type:<16} {count:>5}")

    now = datetime.now().isoformat(timespec="seconds")
    print(f"\nSinh dòng master{' (bỏ fingerprint)' if args.no_fingerprints else ''}...")
    compositions, recordings, rights, fingerprints = build_rows(
        tracks, now, not args.no_fingerprints)

    old_recordings = read_csv(MASTERS["metadata"])
    old_rights = read_csv(MASTERS["rights"])
    old_compositions = read_csv(MASTERS["compositions"])
    sim_comp, sim_rec, sim_rights = keep_simulated(
        old_recordings, old_rights, old_compositions)

    print(f"\nGiữ lại {len(sim_rec)} bản ghi mô phỏng thuộc nhóm không có nguồn mở "
          f"({', '.join(sorted(KEEP_SIMULATED_LICENSES))})")
    print(f"Loại bỏ {len(old_recordings) - len(sim_rec)} bản ghi mô phỏng còn lại")
    print(f"Thêm    {len(recordings)} bản ghi có audio thật")

    total_rec = len(recordings) + len(sim_rec)
    print(f"\nSau khi nạp: {total_rec} bản ghi, trong đó "
          f"{len(recordings)} có audio ({len(recordings) / max(total_rec, 1) * 100:.0f}%) "
          f"và {len(fingerprints)} có fingerprint")
    unique_fp = len({f["fingerprint"] for f in fingerprints})
    print(f"Fingerprint duy nhất: {unique_fp}/{len(fingerprints)}"
          f"{'  ✅ mỗi bản ghi một fingerprint riêng' if unique_fp == len(fingerprints) else '  ⚠️ có trùng lặp'}")

    if args.dry_run:
        print("\n--dry-run: không ghi gì cả.")
        return 0

    # Đã có đủ dữ liệu trong bộ nhớ -> checkpoint hết nhiệm vụ. Xoá TRƯỚC khi
    # ghi master để lần chạy sau không vô tình nối tiếp một corpus đã nạp xong.
    checkpoint_path = os.path.join(config.BASE_DIR, "experiments", "results",
                                   ".checkpoints", "ingest_corpus.jsonl")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Đã xoá checkpoint ingest (nạp xong)")

    write_csv(MASTERS["compositions"], FIELDS["compositions"], sim_comp + compositions)
    write_csv(MASTERS["metadata"], FIELDS["metadata"], sim_rec + recordings)
    write_csv(MASTERS["rights"], FIELDS["rights"], sim_rights + rights)
    # Bản ghi không có audio thì không có fingerprint — đó là điểm mấu chốt
    write_csv(MASTERS["fingerprints"], FIELDS["fingerprints"], fingerprints)

    # Embedding cũ thuộc về các bản ghi vừa bị loại, giữ lại là dữ liệu rác
    if os.path.exists(MASTERS["embeddings"]):
        os.remove(MASTERS["embeddings"])
        print("Đã xoá embeddings_master.csv cũ (thuộc về bản ghi đã loại)")

    print(f"\n💾 Đã ghi 4 file master vào {config.DATA_DIR}")
    # init_db.py phải chạy TRƯỚC build_embeddings.py: script đó lấy danh sách
    # recording từ CSDL, nên chạy ngược thứ tự thì nó chỉ thấy corpus mô phỏng cũ.
    print("\nBước tiếp, theo đúng thứ tự:")
    print("  python init_db.py                          # nạp corpus mới vào PostgreSQL")
    print("  python scripts/build_embeddings.py --all    # MERT + dựng lại FAISS")
    print("  python scripts/augment_audio.py --from-db --sample 60")
    print("  python init_db.py                          # nạp embedding + test_queries")
    print("  python scripts/check_data_integrity.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
