"""
Gắn nhãn trung thực cho dữ liệu mô phỏng ĐANG NẰM trong data/processed/*_master.csv.

Vì sao là script riêng chứ không chạy lại process_all_datasets.py: script gộp đó
ghi đè cả bốn file master bằng pandas, nên sẽ định dạng lại 24.375 dòng FMA mà
ingest_corpus.py đã ghi và dựng lại mọi nguồn khác từ đầu. Ở đây đọc/ghi theo DÒNG
bằng module csv, và dùng CHUNG các quy tắc gắn nhãn với process_all_datasets.py.

Chốt an toàn: trước khi thay file, so từng dòng giữa bản cũ và bản mới. Nếu số
dòng khác nhau không bằng đúng số dòng script chủ ý sửa — tức định dạng bị xáo
trộn ở chỗ khác — thì huỷ, không ghi gì. Bản cũ giữ ở <file>.old (bị gitignore).

Chạy lại nhiều lần cho cùng một kết quả.

    python scripts/relabel_simulated_sources.py            # chỉ xem trước
    python scripts/relabel_simulated_sources.py --write
"""
import argparse
import csv
import os
import shutil
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from scripts.process_all_datasets import (
    VIETNAM_SIMULATED_SOURCE,
    creator_music_flags,
    metadata_is_verified,
    recording_pd_plausible,
    spotify_rights_source,
)

csv.field_size_limit(10 ** 9)
VIETNAM = "dataset_G_vietnam_100k_api"
JAMENDO = "MTG-Jamendo"


def is_true(value) -> bool:
    return str(value).strip().lower() in ("true", "1")


def read(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def line_terminator(path: str) -> str:
    with open(path, "rb") as f:
        return "\r\n" if f.readline().endswith(b"\r\n") else "\n"


def relabel_recording(row: dict) -> bool:
    want = str(metadata_is_verified(row["source_dataset"]))
    if row["metadata_verified"] == want:
        return False
    row["metadata_verified"] = want
    return True


def relabel_rights(row: dict, recording: dict) -> bool:
    before = dict(row)
    if recording.get("source_dataset") == VIETNAM:
        row["source"] = VIETNAM_SIMULATED_SOURCE
        row["verified_at"] = ""
        purchased, share = creator_music_flags(row["license_type"],
                                               row["revenue_share_required"])
        row["license_purchased"], row["revenue_share_agreed"] = str(purchased), str(share)
        if (is_true(row["recording_public_domain"])
                and not recording_pd_plausible(recording.get("release_year"))):
            # Bản thu phát hành quá gần hiện tại thì không thể đã hết bảo hộ
            row["recording_public_domain"] = "False"
            row["copyright_status"] = "PROTECTED"
            row["commercial_use_allowed"] = "False"
            row["monetization_allowed"] = "False"
    elif row["source"].strip() == "Spotify API / nan":
        row["source"] = spotify_rights_source(None)
    return row != before


def relabel_composition(row: dict, dataset: str) -> bool:
    before = dict(row)
    if dataset == VIETNAM:
        row["source"] = VIETNAM_SIMULATED_SOURCE
        row["verified_at"] = ""
        if row["public_domain_status"] == "verified":
            row["public_domain_status"] = "possible"
    elif dataset == JAMENDO:
        row["verified_at"] = ""
    return row != before


def replace_checked(path: str, fieldnames: list, rows: list, changed: int) -> None:
    name = os.path.basename(path)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator=line_terminator(path))
        writer.writeheader()
        writer.writerows(rows)

    with open(path, encoding="utf-8", newline="") as a, open(tmp, encoding="utf-8", newline="") as b:
        old_lines, new_lines = a.read().splitlines(), b.read().splitlines()
    if len(old_lines) != len(new_lines):
        os.remove(tmp)
        raise SystemExit(f"❌ {name}: số dòng {len(old_lines)} -> {len(new_lines)}, KHÔNG ghi")
    differing = sum(1 for x, y in zip(old_lines, new_lines) if x != y)
    if differing != changed:
        os.remove(tmp)
        raise SystemExit(f"❌ {name}: {differing} dòng khác nhau nhưng chỉ chủ ý sửa "
                         f"{changed} -> định dạng bị xáo trộn ở chỗ khác, KHÔNG ghi")

    shutil.copy2(path, path + ".old")
    os.replace(tmp, path)
    print(f"💾 {name}: đã thay {changed} dòng (bản cũ: {name}.old)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Ghi thật (mặc định chỉ xem trước)")
    args = parser.parse_args()

    paths = {
        "recordings": os.path.join(config.DATA_DIR, "metadata_master.csv"),
        "rights": os.path.join(config.DATA_DIR, "rights_master.csv"),
        "compositions": os.path.join(config.DATA_DIR, "compositions_master.csv"),
    }
    rec_fields, recordings = read(paths["recordings"])
    rights_fields, rights = read(paths["rights"])
    comp_fields, compositions = read(paths["compositions"])

    # Tra theo bản ghi GỐC: metadata_verified sắp đổi không ảnh hưởng quy tắc bên dưới
    rec_by_id = {r["recording_id"]: dict(r) for r in recordings}
    dataset_by_comp = {r["composition_id"]: r["source_dataset"] for r in recordings}

    changes = {name: Counter() for name in paths}
    for row in recordings:
        if relabel_recording(row):
            changes["recordings"][row["source_dataset"]] += 1
    for row in rights:
        recording = rec_by_id.get(row["recording_id"], {})
        if relabel_rights(row, recording):
            changes["rights"][recording.get("source_dataset", "?")] += 1
    for row in compositions:
        dataset = dataset_by_comp.get(row["composition_id"], "")
        if relabel_composition(row, dataset):
            changes["compositions"][dataset] += 1

    print("Số dòng sẽ đổi, theo nguồn:")
    for name, counter in changes.items():
        print(f"  {name:<13} {sum(counter.values()):>7}  {dict(counter)}")

    if not args.write:
        print("\nChỉ xem trước. Thêm --write để ghi.")
        return 0

    for name, fields, rows in (("recordings", rec_fields, recordings),
                               ("rights", rights_fields, rights),
                               ("compositions", comp_fields, compositions)):
        total = sum(changes[name].values())
        if total:
            replace_checked(paths[name], fields, rows, total)
        else:
            print(f"   {os.path.basename(paths[name])}: không có gì cần đổi")
    print("\nBước tiếp: python init_db.py && python scripts/check_data_integrity.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
