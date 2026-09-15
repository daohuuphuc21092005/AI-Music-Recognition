"""
Sinh lại rights_master.csv sao cho phủ đủ 5 NHÓM BẢN QUYỀN của Rule Engine.

Vì sao cần: dữ liệu cũ có `copyright_status` = PROTECTED cho 100% bản ghi và
`license_type` chỉ gồm COMMERCIAL / CREATIVE_COMMONS. Với dữ liệu đó, 4/6 nhánh
của Rule Engine (CLAUDE.md §9) KHÔNG BAO GIỜ chạy, nên không thể viết test cho
chúng, cũng không thể demo được đầy đủ.

⚠️ ĐÂY LÀ METADATA MÔ PHỎNG. Cột `source` của mọi dòng đều ghi rõ "SIMULATED".
Không được trích dẫn như giấy phép thật của bất kỳ bản ghi nào. CLAUDE.md §8 cho
phép điều này với Creator Music ("chỉ metadata mô phỏng cho Rule Engine"); ở đây
mở rộng ra các nhóm còn lại với cùng nguyên tắc: gán theo dataset nguồn, xác
định (deterministic) bằng hash của recording_id nên chạy lại luôn ra kết quả cũ.

Quy tắc gán theo dataset nguồn:

  YouTube Audio Library -> AUDIO_LIBRARY     60% không cần ghi công / 40% cần
  Creator Music         -> CREATOR_MUSIC     50% đã mua license / 50% chia doanh thu
  MTG-Jamendo           -> CREATIVE_COMMONS  CC_BY 40 / CC_BY_SA 20 / CC_BY_NC 25 / CC_BY_NC_ND 15
  FMA                   -> 50% CREATIVE_COMMONS, 50% CONTENT_ID
                           (CONTENT_ID: 70% MONETIZE_CLAIM / 30% BLOCK_OR_TAKEDOWN)
  Public Domain Archive -> PUBLIC_DOMAIN     tác phẩm PD 100%;
                           BẢN THU chỉ 60% PD, 40% còn bản quyền
                           (đúng nguyên tắc §2: composition PD KHÔNG suy ra recording PD)

Dùng:  python scripts/enrich_rights_metadata.py [--dry-run]
Khôi phục: git checkout HEAD -- data/processed/rights_master.csv
"""
import argparse
import csv
import hashlib
import os
import sys
import uuid
from collections import Counter
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config

csv.field_size_limit(10 ** 9)

FIELDNAMES = [
    "rights_id", "recording_id", "composition_id",
    "license_type", "copyright_status",
    "attribution_required", "commercial_use_allowed", "modification_allowed",
    "monetization_allowed", "revenue_share_required",
    # --- Cột bổ sung phục vụ Rule Engine (§9). Xem data/metadata/data_dictionary.md
    "policy_action",             # Content ID: MONETIZE_CLAIM | BLOCK_OR_TAKEDOWN | NONE
    "license_purchased",         # Creator Music: đã mua giấy phép hợp lệ chưa
    "revenue_share_agreed",      # Creator Music: đã đồng ý chia doanh thu chưa
    "recording_public_domain",   # PD của BẢN THU — xét độc lập với PD của tác phẩm
    # ---
    "territory", "platform", "valid_from", "valid_until",
    "source", "source_url", "verified_at",
]

SOURCE_URLS = {
    "YouTube Audio Library": "https://www.youtube.com/audiolibrary",
    "Creator Music": "https://www.youtube.com/creatormusic",
    "MTG-Jamendo": "https://mtg.github.io/mtg-jamendo-dataset/",
    "FMA": "https://freemusicarchive.org/",
    "Public Domain Archive": "https://archive.org/details/audio",
}


def bucket(recording_id: str, salt: str, modulo: int = 100) -> int:
    """Số ngẫu nhiên nhưng TIỀN ĐỊNH theo recording_id -> chạy lại ra kết quả cũ."""
    digest = hashlib.sha256(f"{salt}:{recording_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def base_row(rec, now):
    return {
        "rights_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"rights:{rec['recording_id']}")),
        "recording_id": rec["recording_id"],
        "composition_id": rec["composition_id"],
        "policy_action": "NONE",
        "license_purchased": False,
        "revenue_share_agreed": False,
        "recording_public_domain": False,
        "territory": "GLOBAL",
        "platform": "ALL",
        "valid_from": "2020-01-01",
        "valid_until": "2030-12-31",
        "source": f"SIMULATED ({rec['source_dataset']})",
        "source_url": SOURCE_URLS.get(rec["source_dataset"], ""),
        "verified_at": now,
    }


def build_row(rec, now):
    row = base_row(rec, now)
    dataset = rec["source_dataset"]
    rec_id = rec["recording_id"]

    if dataset == "YouTube Audio Library":
        needs_attribution = bucket(rec_id, "ytal") < 40
        row.update({
            "license_type": "AUDIO_LIBRARY",
            "copyright_status": "LICENSED",
            "attribution_required": needs_attribution,
            "commercial_use_allowed": True,
            "modification_allowed": True,
            "monetization_allowed": True,
            "revenue_share_required": False,
        })

    elif dataset == "Creator Music":
        purchased = bucket(rec_id, "cm") < 50
        row.update({
            "license_type": "CREATOR_MUSIC",
            "copyright_status": "PROTECTED",
            "attribution_required": False,
            "commercial_use_allowed": True,
            "modification_allowed": False,
            "monetization_allowed": True,
            "revenue_share_required": not purchased,
            "license_purchased": purchased,
            "revenue_share_agreed": not purchased,
        })

    elif dataset == "MTG-Jamendo":
        b = bucket(rec_id, "jamendo")
        if b < 40:
            variant = "CC_BY"
        elif b < 60:
            variant = "CC_BY_SA"
        elif b < 85:
            variant = "CC_BY_NC"
        else:
            variant = "CC_BY_NC_ND"
        row.update({
            "license_type": variant,
            "copyright_status": "PROTECTED",
            "attribution_required": True,
            "commercial_use_allowed": "NC" not in variant,
            "modification_allowed": "ND" not in variant,
            "monetization_allowed": "NC" not in variant,
            "revenue_share_required": False,
        })

    elif dataset == "FMA":
        if bucket(rec_id, "fma") < 50:
            variant = "CC_BY" if bucket(rec_id, "fma_cc") < 60 else "CC_BY_NC"
            row.update({
                "license_type": variant,
                "copyright_status": "PROTECTED",
                "attribution_required": True,
                "commercial_use_allowed": "NC" not in variant,
                "modification_allowed": True,
                "monetization_allowed": "NC" not in variant,
                "revenue_share_required": False,
            })
        else:
            blocked = bucket(rec_id, "fma_policy") < 30
            row.update({
                "license_type": "CONTENT_ID",
                "copyright_status": "PROTECTED",
                "attribution_required": False,
                "commercial_use_allowed": False,
                "modification_allowed": False,
                "monetization_allowed": False,
                "revenue_share_required": not blocked,
                "policy_action": "BLOCK_OR_TAKEDOWN" if blocked else "MONETIZE_CLAIM",
            })

    elif dataset == "Public Domain Archive":
        # Tác phẩm đã hết bảo hộ, NHƯNG bản thu thì chưa chắc.
        recording_is_pd = bucket(rec_id, "pd") < 60
        row.update({
            "license_type": "PUBLIC_DOMAIN",
            "copyright_status": "PUBLIC_DOMAIN" if recording_is_pd else "PROTECTED",
            "attribution_required": False,
            "commercial_use_allowed": recording_is_pd,
            "modification_allowed": recording_is_pd,
            "monetization_allowed": recording_is_pd,
            "revenue_share_required": False,
            "recording_public_domain": recording_is_pd,
        })

    else:
        row.update({
            "license_type": "UNKNOWN",
            "copyright_status": "UNKNOWN",
            "attribution_required": True,
            "commercial_use_allowed": False,
            "modification_allowed": False,
            "monetization_allowed": False,
            "revenue_share_required": False,
        })

    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ in thống kê, không ghi file")
    args = parser.parse_args()

    metadata_csv = config.METADATA_CSV
    rights_csv = os.path.join(config.DATA_DIR, "rights_master.csv")

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        recordings = list(csv.DictReader(f))

    now = datetime.now().isoformat()
    rows = [build_row(rec, now) for rec in recordings]

    licenses = Counter(r["license_type"] for r in rows)
    statuses = Counter(r["copyright_status"] for r in rows)
    policies = Counter(r["policy_action"] for r in rows)

    print("=== PHÂN BỐ RIGHTS SAU KHI SINH LẠI ===")
    print(f"Tổng: {len(rows)} bản ghi\n")
    print("license_type:")
    for k, v in licenses.most_common():
        print(f"   {k:16} {v:>5}")
    print("copyright_status:")
    for k, v in statuses.most_common():
        print(f"   {k:16} {v:>5}")
    print("policy_action (Content ID):")
    for k, v in policies.most_common():
        print(f"   {k:18} {v:>5}")
    pd_split = Counter(r["recording_public_domain"] for r in rows
                       if r["license_type"] == "PUBLIC_DOMAIN")
    print(f"Public Domain — bản thu PD: {pd_split.get(True, 0)}, "
          f"bản thu còn bảo hộ: {pd_split.get(False, 0)}")
    print(f"Creator Music — đã mua license: "
          f"{sum(1 for r in rows if r['license_purchased'])}")

    if args.dry_run:
        print("\n(--dry-run: không ghi file)")
        return 0

    with open(rights_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Đã ghi {rights_csv} ({len(rows)} dòng, {len(FIELDNAMES)} cột)")
    print("⚠️  Toàn bộ là metadata MÔ PHỎNG (cột source ghi rõ 'SIMULATED').")
    print("Bước tiếp: python init_db.py  rồi  python scripts/check_data_integrity.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
