"""
Tổng hợp và chuẩn hóa toàn bộ các nguồn dữ liệu từ data/preprocess/ và data/processed/:
  1. 100k Bản ghi Việt Nam (dataset_G_vietnam_100k_api: recordings, rights, fingerprints)
  2. 29.8k Bản ghi Jamendo (tracks_clean.csv: Creative Commons)
  3. 3.3k Bản ghi Spotify Web API (1.csv, 2.csv, 3.csv, 4.csv)
  4. 4.9k Bản ghi FMA hiện tại (giữ nguyên để bảo toàn MERT embeddings & FAISS index)

Xuất ra data/processed/ đầy đủ 4 bảng master đạt chuẩn 100% PostgreSQL Schema & Rule Engine:
  - compositions_master.csv
  - metadata_master.csv (recordings)
  - rights_master.csv (kèm 4 cột Rule Engine: policy_action, license_purchased...)
  - fingerprints_master.csv
"""
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from backend.services.chromaprint_codec import is_plausible_fingerprint

PREPROCESS_DIR = os.path.join(config.BASE_DIR, "data", "preprocess")
PROCESSED_DIR = config.DATA_DIR
JAMENDO_UNVERIFIED_SOURCE = (
    "SIMULATED (MTG-Jamendo: loại giấy phép suy từ cờ audiodownload_allowed, "
    "chưa đối chiếu giấy phép thật)"
)
VIETNAM_SIMULATED_SOURCE = (
    "SIMULATED (dataset_G_vietnam_100k_api: tên bài, ID Spotify và cờ quyền là "
    "dữ liệu tổng hợp, chưa đối chiếu nguồn thật)"
)

# Chỉ nguồn TỰ CÔNG BỐ giấy phép (FMA) mới được coi là metadata đã xác minh. Trước
# đây cả 158.117 dòng đều metadata_verified=True, kể cả 100.000 bài tên dạng
# "Tác phẩm Nhạc Việt #000001" với ID Spotify giả.
VERIFIED_METADATA_SOURCES = frozenset({"FMA"})

# Bản thu chỉ có thể đã hết bảo hộ khi đủ lâu kể từ khi phát hành. 70 năm là mốc
# thận trọng; dữ liệu tổng hợp từng gắn PD cho cả bản thu phát hành năm 2025.
RECORDING_PD_MIN_AGE_YEARS = 70


def metadata_is_verified(source_dataset) -> bool:
    return str(source_dataset) in VERIFIED_METADATA_SOURCES


def recording_pd_plausible(release_year, now_year: int = None) -> bool:
    """
    Bản thu có thể đã HẾT THỜI HẠN bảo hộ chưa, xét theo năm phát hành.

    Chỉ dùng cho PD do hết hạn. KHÔNG dùng cho CC0: đó là tuyên bố từ bỏ quyền của
    chính chủ sở hữu, bản thu mới phát hành vẫn có thể PD hợp pháp.
    """
    try:
        year = int(float(release_year))
    except (TypeError, ValueError):
        return False
    return year <= (now_year or datetime.now().year) - RECORDING_PD_MIN_AGE_YEARS


def creator_music_flags(license_type, revenue_share_required) -> tuple:
    """
    (license_purchased, revenue_share_agreed) — với CREATOR_MUSIC hai cờ LOẠI TRỪ
    nhau. Bản cũ bật cả hai cho mọi dòng, nên Rule Engine luôn dừng ở nhánh "đã
    mua" và nhánh chia doanh thu không bao giờ được thử.

    Đọc cờ qua chuỗi chứ không qua bool(): bool(NaN) là True, và chính NaN của cột
    trống từng biến "không chia doanh thu" thành "có chia doanh thu".
    """
    share = str(revenue_share_required).strip().lower() in ("true", "1")
    if license_type != "CREATOR_MUSIC":
        return False, share
    return (not share), share


def spotify_rights_source(label) -> str:
    text = "" if label is None else str(label).strip()
    if not text or text.lower() == "nan":
        text = "không rõ hãng phát hành"
    return f"Spotify API / {text}"


def to_uuid(val: str, namespace_prefix: str = "amr_default") -> str:
    """Chuyển đổi bất kỳ ID nào thành UUID v5 hợp lệ nếu chưa phải UUID v4/v5."""
    if not val or pd.isna(val):
        return str(uuid.uuid4())
    s = str(val).strip()
    try:
        return str(uuid.UUID(s))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{namespace_prefix}_{s}"))


def extract_composer(title: str, artist: str) -> str:
    """Trích xuất nhạc sĩ/tác giả từ tên bài hát (ví dụ: 'Tác phẩm #01 (DTAP)' -> 'DTAP')."""
    if not title or pd.isna(title):
        return str(artist) if artist and not pd.isna(artist) else "Unknown Composer"
    m = re.search(r'\(([^)]+)\)', str(title))
    if m:
        return m.group(1).strip()
    return str(artist) if artist and not pd.isna(artist) else "Unknown Composer"


def process_and_merge():
    t0 = time.time()
    print("=== BẮT ĐẦU TỔNG HỢP & CHUẨN HÓA DỮ LIỆU TỪ DATA/PREPROCESS ===")

    # --------------------------------------------------------------------------
    # 1. NẠP DỮ LIỆU FMA HIỆN TẠI TỪ data/processed/ (để giữ nguyên Embeddings)
    # --------------------------------------------------------------------------
    fma_comp_path = os.path.join(PROCESSED_DIR, "compositions_master.csv")
    fma_rec_path = os.path.join(PROCESSED_DIR, "metadata_master.csv")
    fma_rights_path = os.path.join(PROCESSED_DIR, "rights_master.csv")
    fma_fp_path = os.path.join(PROCESSED_DIR, "fingerprints_master.csv")

    existing_rec_ids = set()
    all_compositions = []
    all_recordings = []
    all_rights = []
    all_fingerprints = []

    if os.path.exists(fma_rec_path):
        df_fma_rec = pd.read_csv(fma_rec_path)
        # Chỉ nạp các bản ghi gốc (FMA, YouTube Audio Library, Creator Music) để bảo toàn embeddings
        mask = ~df_fma_rec["source_dataset"].isin(["dataset_G_vietnam_100k_api", "MTG-Jamendo", "Spotify Web API"])
        df_fma_rec = df_fma_rec[mask]
        fma_rec_id_set = set(df_fma_rec["recording_id"].astype(str))
        for _, r in df_fma_rec.iterrows():
            rid = str(r["recording_id"])
            existing_rec_ids.add(rid)
            all_recordings.append(r.to_dict())
        print(f"-> Đã nạp {len(all_recordings)} bản ghi gốc (FMA/YAL/CM) từ metadata_master.csv")

    if os.path.exists(fma_comp_path):
        df_fma_comp = pd.read_csv(fma_comp_path)
        fma_comp_ids = set(df_fma_rec["composition_id"].astype(str))
        df_fma_comp = df_fma_comp[df_fma_comp["composition_id"].astype(str).isin(fma_comp_ids)]
        all_compositions.extend(df_fma_comp.to_dict(orient="records"))

    if os.path.exists(fma_rights_path):
        df_fma_rights = pd.read_csv(fma_rights_path)
        df_fma_rights = df_fma_rights[df_fma_rights["recording_id"].astype(str).isin(fma_rec_id_set)]
        all_rights.extend(df_fma_rights.to_dict(orient="records"))

    if os.path.exists(fma_fp_path):
        df_fma_fp = pd.read_csv(fma_fp_path)
        df_fma_fp = df_fma_fp[df_fma_fp["recording_id"].astype(str).isin(fma_rec_id_set)]
        all_fingerprints.extend(df_fma_fp.to_dict(orient="records"))

    # --------------------------------------------------------------------------
    # 2. XỬ LÝ TẬP 100K VIỆT NAM (recordings, rights, fingerprints)
    # --------------------------------------------------------------------------
    vn_rec_path = os.path.join(PREPROCESS_DIR, "recordings_master.csv")
    vn_rights_path = os.path.join(PREPROCESS_DIR, "rights_master.csv")
    vn_fp_path = os.path.join(PREPROCESS_DIR, "fingerprints_master.csv")

    if os.path.exists(vn_rec_path):
        print("\n-> Đang xử lý tập 100k Việt Nam (dataset_G_vietnam_100k_api)...")
        df_vn_rec = pd.read_csv(vn_rec_path)
        df_vn_rights = pd.read_csv(vn_rights_path) if os.path.exists(vn_rights_path) else None
        df_vn_fp = pd.read_csv(vn_fp_path) if os.path.exists(vn_fp_path) else None

        # Index rights theo recording_id để tra cứu nhanh
        vn_rights_map = {}
        if df_vn_rights is not None:
            for _, r in df_vn_rights.iterrows():
                vn_rights_map[str(r["recording_id"])] = r.to_dict()

        count_vn = 0
        for _, row in df_vn_rec.iterrows():
            rid = str(row["recording_id"])
            if rid in existing_rec_ids:
                continue
            existing_rec_ids.add(rid)

            cid = str(row["composition_id"])
            title = str(row.get("title", "Untitled"))
            artist = str(row.get("artist", "Unknown Artist"))
            year = int(row.get("release_year", 2020)) if pd.notna(row.get("release_year")) else 2020

            # Lấy rights tương ứng
            r_info = vn_rights_map.get(rid, {})
            lic_type = str(r_info.get("license_type", "COMMERCIAL"))
            is_pd = (lic_type == "PUBLIC_DOMAIN")

            # Tạo composition
            composer = extract_composer(title, artist)
            all_compositions.append({
                "composition_id": cid,
                "title": title,
                "composer": composer,
                "year": year,
                # Dữ liệu tổng hợp chỉ KHẲNG ĐỊNH là PD, chưa ai xác minh -> "possible"
                "public_domain_status": "possible" if is_pd else "no",
                "source": VIETNAM_SIMULATED_SOURCE,
                "verified_at": "",
            })

            # Lưu recording
            all_recordings.append({
                "recording_id": rid,
                "composition_id": cid,
                "title": title,
                "artist": artist,
                "album": str(row.get("album", "")),
                "release_year": year,
                "duration": float(row.get("duration", 0.0)),
                "source_dataset": str(row.get("source_dataset", "dataset_G_vietnam_100k_api")),
                "source_track_id": str(row.get("source_track_id", "")),
                "audio_path": str(row.get("audio_path", "")),
                "metadata_verified": metadata_is_verified(
                    row.get("source_dataset", "dataset_G_vietnam_100k_api")),
            })

            # Bổ sung 4 cột Rule Engine vào rights
            policy_action = "MONETIZE_CLAIM" if lic_type in ("COMMERCIAL", "CONTENT_ID") else "NONE"
            lic_purchased, rev_share = creator_music_flags(
                lic_type, r_info.get("revenue_share_required", False))

            # Tách bạch quyền tác phẩm PD và quyền bản thu PD (§2 core invariants)
            if is_pd:
                digest = hashlib.sha256(f"pd_rec:{rid}".encode("utf-8")).hexdigest()
                rec_pd = (int(digest[:8], 16) % 100 < 60) and recording_pd_plausible(year)
                copyright_status = "PUBLIC_DOMAIN" if rec_pd else "PROTECTED"
                comm_allowed = rec_pd
                monetize_allowed = rec_pd
            else:
                rec_pd = False
                copyright_status = str(r_info.get("copyright_status", "PROTECTED"))
                comm_allowed = bool(r_info.get("commercial_use_allowed", False))
                monetize_allowed = bool(r_info.get("monetization_allowed", False))

            all_rights.append({
                "rights_id": str(r_info.get("rights_id", uuid.uuid4())),
                "recording_id": rid,
                "composition_id": cid,
                "license_type": lic_type,
                "copyright_status": copyright_status,
                "attribution_required": bool(r_info.get("attribution_required", True)),
                "commercial_use_allowed": comm_allowed,
                "modification_allowed": bool(r_info.get("modification_allowed", False)),
                "monetization_allowed": monetize_allowed,
                "revenue_share_required": bool(r_info.get("revenue_share_required", False)),
                "policy_action": policy_action,
                "license_purchased": lic_purchased,
                "revenue_share_agreed": rev_share,
                "recording_public_domain": rec_pd,
                "territory": str(r_info.get("territory", "GLOBAL")),
                "platform": str(r_info.get("platform", "ALL")),
                "valid_from": str(r_info.get("valid_from", "2020-01-01")),
                "valid_until": str(r_info.get("valid_until", "2035-12-31")),
                # Nguồn gốc ghi "Spotify API / Content ID (BH Media, Sony Music)" và
                # một verified_at — cả hai đều không có thật với dữ liệu tổng hợp.
                "source": VIETNAM_SIMULATED_SOURCE,
                "source_url": str(r_info.get("source_url", "")),
                "verified_at": "",
            })
            count_vn += 1

        print(f"  + Đã tích hợp {count_vn:,} bản ghi Việt Nam vào recordings & rights & compositions.")

        if df_vn_fp is not None:
            all_fingerprints.extend(df_vn_fp.to_dict(orient="records"))
            print(f"  + Đã tích hợp {len(df_vn_fp):,} fingerprints từ tập Việt Nam.")

    # --------------------------------------------------------------------------
    # 3. XỬ LÝ TẬP JAMENDO (tracks_clean.csv - 29.8k bài Creative Commons)
    # --------------------------------------------------------------------------
    jamendo_path = os.path.join(PREPROCESS_DIR, "tracks_clean.csv")
    if os.path.exists(jamendo_path):
        print("\n-> Đang xử lý tập Jamendo (tracks_clean.csv)...")
        df_jam = pd.read_csv(jamendo_path)
        count_jam = 0
        for _, row in df_jam.iterrows():
            jid = str(row["id"])
            rid = to_uuid(jid, "jamendo_rec")
            if rid in existing_rec_ids:
                continue
            existing_rec_ids.add(rid)

            cid = to_uuid(jid, "jamendo_comp")
            name = str(row.get("name", "Untitled Jamendo Track"))
            artist = str(row.get("artist_name", "Unknown Artist"))
            album = str(row.get("album_name", ""))
            year = int(row.get("release_year", 2018)) if pd.notna(row.get("release_year")) else 2018
            dur = float(row.get("duration", 0.0))
            audio_url = str(row.get("audio", ""))
            dl_allowed = bool(row.get("audiodownload_allowed", True))
            cid_free = bool(row.get("content_id_free", False))

            all_compositions.append({
                "composition_id": cid,
                "title": name,
                "composer": artist,
                "year": year,
                "public_domain_status": "no",
                "source": "MTG-Jamendo Dataset / Creative Commons",
                "verified_at": "",
            })

            all_recordings.append({
                "recording_id": rid,
                "composition_id": cid,
                "title": name,
                "artist": artist,
                "album": album,
                "release_year": year,
                "duration": dur,
                "source_dataset": "MTG-Jamendo",
                "source_track_id": jid,
                "audio_path": audio_url,
                "metadata_verified": metadata_is_verified("MTG-Jamendo"),
            })

            lic_type = "CC_BY" if dl_allowed else "CC_BY_NC"
            all_rights.append({
                "rights_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"jamendo_rights_{jid}")),
                "recording_id": rid,
                "composition_id": cid,
                "license_type": lic_type,
                "copyright_status": "LICENSED",
                "attribution_required": True,
                "commercial_use_allowed": False,
                "modification_allowed": True,
                "monetization_allowed": cid_free,
                "revenue_share_required": False,
                "policy_action": "NONE",
                "license_purchased": False,
                "revenue_share_agreed": False,
                "recording_public_domain": False,
                "territory": "GLOBAL",
                "platform": "ALL",
                "valid_from": "2010-01-01",
                "valid_until": "2035-12-31",
                # license_type ở trên suy từ cờ `audiodownload_allowed`, KHÔNG phải
                # giấy phép thật của bài. Tiền tố SIMULATED để Rule Engine trừ điểm
                # đúng, verified_at để trống vì chưa từng đối chiếu với nguồn.
                "source": JAMENDO_UNVERIFIED_SOURCE,
                "source_url": f"https://www.jamendo.com/track/{jid}",
                "verified_at": "",
            })
            count_jam += 1
        print(f"  + Đã tích hợp {count_jam:,} bản ghi Jamendo Creative Commons.")

    # --------------------------------------------------------------------------
    # 4. XỬ LÝ TẬP SPOTIFY WEB API (1.csv, 2.csv, 3.csv, 4.csv - 3.3k bài)
    # --------------------------------------------------------------------------
    sp_rec_path = os.path.join(PREPROCESS_DIR, "3.csv")
    sp_meta_path = os.path.join(PREPROCESS_DIR, "2.csv")
    sp_rights_path = os.path.join(PREPROCESS_DIR, "4.csv")

    if os.path.exists(sp_rec_path):
        print("\n-> Đang xử lý tập Spotify Web API (1-4.csv)...")
        df_sp_rec = pd.read_csv(sp_rec_path, encoding="utf-8")
        df_sp_meta = pd.read_csv(sp_meta_path, encoding="utf-8") if os.path.exists(sp_meta_path) else None
        df_sp_rights = pd.read_csv(sp_rights_path, encoding="utf-8") if os.path.exists(sp_rights_path) else None

        meta_by_rec = {}
        if df_sp_meta is not None:
            for _, r in df_sp_meta.iterrows():
                meta_by_rec[str(r["recording_id"])] = r.to_dict()

        rights_by_rec = {}
        if df_sp_rights is not None:
            for _, r in df_sp_rights.iterrows():
                rights_by_rec[str(r["recording_id"])] = r.to_dict()

        count_sp = 0
        for _, row in df_sp_rec.iterrows():
            raw_rec_id = str(row["recording_id"])
            isrc = str(row.get("isrc", raw_rec_id))
            rid = to_uuid(raw_rec_id, "spotify_rec")
            if rid in existing_rec_ids:
                continue
            existing_rec_ids.add(rid)

            cid = to_uuid(isrc, "spotify_comp")
            title = str(row.get("title", "Untitled"))
            artist = str(row.get("artist_names", "Unknown Artist"))
            dur_sec = round(float(row.get("duration_ms", 0)) / 1000.0, 2)
            rel_date = str(row.get("first_release_date", "2020"))
            year = int(rel_date[:4]) if len(rel_date) >= 4 and rel_date[:4].isdigit() else 2020

            meta_info = meta_by_rec.get(raw_rec_id, {})
            album = str(meta_info.get("album_name", ""))
            spotify_url = str(meta_info.get("spotify_url", ""))

            all_compositions.append({
                "composition_id": cid,
                "title": title,
                "composer": artist,
                "year": year,
                "public_domain_status": "no",
                "source": "Spotify Web API / ISRC Metadata",
                "verified_at": "2026-09-15T00:00:00",
            })

            all_recordings.append({
                "recording_id": rid,
                "composition_id": cid,
                "title": title,
                "artist": artist,
                "album": album,
                "release_year": year,
                "duration": dur_sec,
                "source_dataset": "Spotify Web API",
                "source_track_id": str(row.get("primary_spotify_track_id", isrc)),
                "audio_path": spotify_url,
                "metadata_verified": metadata_is_verified("Spotify Web API"),
            })

            r_sp = rights_by_rec.get(raw_rec_id, {})
            label = str(r_sp.get("label", "Commercial Label"))
            all_rights.append({
                "rights_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"spotify_rights_{raw_rec_id}")),
                "recording_id": rid,
                "composition_id": cid,
                "license_type": "COMMERCIAL",
                "copyright_status": "PROTECTED",
                "attribution_required": True,
                "commercial_use_allowed": False,
                "modification_allowed": False,
                "monetization_allowed": False,
                "revenue_share_required": False,
                "policy_action": "MONETIZE_CLAIM",
                "license_purchased": False,
                "revenue_share_agreed": False,
                "recording_public_domain": False,
                "territory": "GLOBAL",
                "platform": "ALL",
                "valid_from": rel_date if len(rel_date) >= 10 else f"{year}-01-01",
                "valid_until": "2035-12-31",
                "source": spotify_rights_source(label),
                "source_url": spotify_url,
                "verified_at": "2026-09-15T00:00:00",
            })
            count_sp += 1
        print(f"  + Đã tích hợp {count_sp:,} bản ghi Spotify Web API.")

    # --------------------------------------------------------------------------
    # 5. XÁC THỰC TOÀN VẸN & LOẠI BỎ TRÙNG LẶP TRƯỚC KHI XUẤT
    # --------------------------------------------------------------------------
    print("\n-> Đang kiểm tra tính toàn vẹn và deduplicate...")
    df_out_comp = pd.DataFrame(all_compositions).drop_duplicates(subset=["composition_id"])
    df_out_rec = pd.DataFrame(all_recordings).drop_duplicates(subset=["recording_id"])
    df_out_rights = pd.DataFrame(all_rights).drop_duplicates(subset=["rights_id"])
    df_out_fp = pd.DataFrame(all_fingerprints).drop_duplicates(subset=["fingerprint_id"])

    # Đảm bảo foreign key toàn vẹn
    valid_comp_set = set(df_out_comp["composition_id"])
    df_out_rec = df_out_rec[df_out_rec["composition_id"].isin(valid_comp_set)].copy()

    valid_rec_set = set(df_out_rec["recording_id"])
    df_out_rights = df_out_rights[
        df_out_rights["recording_id"].isin(valid_rec_set) & df_out_rights["composition_id"].isin(valid_comp_set)
    ].copy()
    df_out_fp = df_out_fp[df_out_fp["recording_id"].isin(valid_rec_set)].copy()
    # Chỉ giữ fingerprint là đầu ra thật của fpcalc: nguồn chỉ có metadata có thể
    # kèm chuỗi mang nhãn chromaprint nhưng không sinh từ âm thanh nào, và tầng 1
    # so khớp thẳng với bảng này.
    if len(df_out_fp):
        plausible = df_out_fp.apply(
            lambda r: is_plausible_fingerprint(r["fingerprint"], r.get("duration")), axis=1)
        if (~plausible).any():
            print(f"  ! Loại {int((~plausible).sum()):,} fingerprint không phải "
                  f"đầu ra thật của fpcalc")
        df_out_fp = df_out_fp[plausible].copy()

    print(f"  * Tổng compositions : {len(df_out_comp):,}")
    print(f"  * Tổng recordings   : {len(df_out_rec):,}")
    print(f"  * Tổng rights       : {len(df_out_rights):,}")
    print(f"  * Tổng fingerprints : {len(df_out_fp):,}")

    # --------------------------------------------------------------------------
    # 6. SAO LƯU DỮ LIỆU CŨ VÀ XUẤT RA data/processed/
    # --------------------------------------------------------------------------
    backup_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(PROCESSED_DIR, f"_backup_{backup_tag}")
    os.makedirs(backup_dir, exist_ok=True)
    for fname in ["compositions_master.csv", "metadata_master.csv", "rights_master.csv", "fingerprints_master.csv"]:
        src = os.path.join(PROCESSED_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup_dir, fname))
    print(f"\n-> Đã tạo bản sao lưu dữ liệu cũ tại: {backup_dir}")

    # Ghi file mới ra data/processed/
    out_comp_path = os.path.join(PROCESSED_DIR, "compositions_master.csv")
    out_rec_path = os.path.join(PROCESSED_DIR, "metadata_master.csv")
    out_rights_path = os.path.join(PROCESSED_DIR, "rights_master.csv")
    out_fp_path = os.path.join(PROCESSED_DIR, "fingerprints_master.csv")

    df_out_comp.to_csv(out_comp_path, index=False, encoding="utf-8")
    df_out_rec.to_csv(out_rec_path, index=False, encoding="utf-8")
    df_out_rights.to_csv(out_rights_path, index=False, encoding="utf-8")
    df_out_fp.to_csv(out_fp_path, index=False, encoding="utf-8")

    elapsed = round(time.time() - t0, 2)
    print(f"\n✅ ĐÃ XUẤT THÀNH CÔNG TOÀN BỘ MASTER DATASET RA {PROCESSED_DIR}:")
    print(f"  - {out_comp_path}: {len(df_out_comp):,} dòng")
    print(f"  - {out_rec_path}: {len(df_out_rec):,} dòng")
    print(f"  - {out_rights_path}: {len(df_out_rights):,} dòng")
    print(f"  - {out_fp_path}: {len(df_out_fp):,} dòng")
    print(f"  Thời gian thực thi: {elapsed}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(process_and_merge())
