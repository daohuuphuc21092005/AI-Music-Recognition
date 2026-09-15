"""
Kiểm tra toàn vẹn dữ liệu tham chiếu — chạy trước khi khởi động server.

Script này mã hoá lại đúng những lỗi đã phát hiện ở Giai đoạn 0 để chúng không
tái diễn trong im lặng:
  - FAISS index đọc được và khớp số lượng với bản đồ ID
  - embeddings_master.csv dùng recording_id là UUID join được với recordings
  - mọi vector đúng số chiều
  - khoá ngoại giữa các file CSV master
  - fingerprint trùng lặp (cảnh báo, không chặn)
  - độ phủ license_type so với 5 nhóm của Rule Engine (cảnh báo)

Dùng:  python scripts/check_data_integrity.py
Mã thoát khác 0 nếu có mục FAIL.
"""
import csv
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config

csv.field_size_limit(10 ** 9)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))


def load_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_uuid(value: str) -> bool:
    return len(str(value)) == 36 and str(value).count("-") == 4


def main() -> int:
    data_dir = config.DATA_DIR
    paths = {
        "compositions": os.path.join(data_dir, "compositions_master.csv"),
        "recordings": os.path.join(data_dir, "metadata_master.csv"),
        "rights": os.path.join(data_dir, "rights_master.csv"),
        "fingerprints": os.path.join(data_dir, "fingerprints_master.csv"),
        "embeddings": config.EMBEDDINGS_CSV,
    }

    missing = [name for name, p in paths.items() if not os.path.exists(p)]
    if missing:
        record(FAIL, "Có đủ file CSV master", f"thiếu: {', '.join(missing)}")
        return report()

    compositions = load_csv(paths["compositions"])
    recordings = load_csv(paths["recordings"])
    rights = load_csv(paths["rights"])
    fingerprints = load_csv(paths["fingerprints"])
    embeddings = load_csv(paths["embeddings"])

    comp_ids = {r["composition_id"] for r in compositions}
    rec_ids = {r["recording_id"] for r in recordings}

    record(PASS, "Đọc được 5 file CSV master",
           f"{len(compositions)} composition, {len(recordings)} recording, "
           f"{len(rights)} rights, {len(fingerprints)} fingerprint, "
           f"{len(embeddings)} embedding")

    # --- Khoá ngoại giữa các CSV -------------------------------------------
    def check_fk(rows, column, valid, label):
        bad = [r for r in rows if r.get(column) and r[column] not in valid]
        if bad:
            record(FAIL, label, f"{len(bad)}/{len(rows)} giá trị không tồn tại "
                                f"(vd. {bad[0][column]})")
        else:
            record(PASS, label, f"{len(rows)}/{len(rows)} hợp lệ")

    check_fk(recordings, "composition_id", comp_ids, "recordings.composition_id -> compositions")
    check_fk(fingerprints, "recording_id", rec_ids, "fingerprints.recording_id -> recordings")
    check_fk(rights, "recording_id", rec_ids, "rights.recording_id -> recordings")
    check_fk(embeddings, "recording_id", rec_ids, "embeddings.recording_id -> recordings")

    # --- Định dạng khoá của embeddings -------------------------------------
    non_uuid = [r for r in embeddings if not is_uuid(r["recording_id"])]
    if non_uuid:
        record(FAIL, "embeddings.recording_id là UUID",
               f"{len(non_uuid)} dòng còn dùng ID nguồn dataset "
               f"(vd. {non_uuid[0]['recording_id']}). "
               f"Chạy: python scripts/rebuild_faiss_index.py")
    else:
        record(PASS, "embeddings.recording_id là UUID", f"{len(embeddings)} dòng")

    # --- Số chiều vector ----------------------------------------------------
    dims = Counter(len(r["vector"].split(",")) for r in embeddings)
    wrong = {d: n for d, n in dims.items() if d != config.EMBEDDING_DIM}
    if wrong:
        record(FAIL, f"Mọi vector đúng {config.EMBEDDING_DIM} chiều",
               f"số chiều lạ: {wrong}")
    else:
        record(PASS, f"Mọi vector đúng {config.EMBEDDING_DIM} chiều",
               f"{len(embeddings)} vector")

    # --- FAISS index + bản đồ ID -------------------------------------------
    try:
        import faiss
        index = faiss.read_index(config.FAISS_INDEX_PATH)
        record(PASS, "Đọc được FAISS index", f"ntotal={index.ntotal}, d={index.d}")

        with open(config.FAISS_ID_MAP_PATH, "r", encoding="utf-8") as f:
            id_map = json.load(f)
        map_ids = id_map.get("recording_ids", [])

        if index.ntotal != len(map_ids):
            record(FAIL, "index.ntotal khớp bản đồ ID",
                   f"{index.ntotal} vector nhưng {len(map_ids)} ID")
        else:
            record(PASS, "index.ntotal khớp bản đồ ID",
                   f"{index.ntotal} vector / {len(set(map_ids))} recording")

        if index.d != config.EMBEDDING_DIM:
            record(FAIL, "Số chiều index", f"{index.d} != {config.EMBEDDING_DIM}")

        unknown = {i for i in map_ids if i not in rec_ids}
        if unknown:
            record(FAIL, "Bản đồ ID -> recordings",
                   f"{len(unknown)} ID không có trong metadata")
        else:
            record(PASS, "Bản đồ ID -> recordings", f"{len(set(map_ids))} recording")
    except FileNotFoundError as e:
        record(FAIL, "Đọc được FAISS index",
               f"{e}. Chạy: python scripts/rebuild_faiss_index.py")
    except Exception as e:
        record(FAIL, "Đọc được FAISS index",
               f"{type(e).__name__}: {str(e)[:120]}. "
               f"Chạy: python scripts/rebuild_faiss_index.py")

    # --- Cảnh báo về chất lượng dữ liệu (không chặn) ------------------------
    fp_values = [r["fingerprint"] for r in fingerprints]
    n_unique = len(set(fp_values))
    if n_unique < len(fp_values):
        dup = Counter(fp_values).most_common(1)[0][1]
        record(WARN, "Fingerprint trùng lặp",
               f"{len(fp_values)} dòng nhưng chỉ {n_unique} giá trị duy nhất "
               f"(một chuỗi lặp tới {dup} lần) -> ground truth không rõ ràng")
    else:
        record(PASS, "Fingerprint trùng lặp", "không có")

    orphan_comp = comp_ids - {r["composition_id"] for r in recordings}
    if orphan_comp:
        record(WARN, "Composition không có recording nào",
               f"{len(orphan_comp)}/{len(comp_ids)} composition mồ côi")

    emb_coverage = len({r["recording_id"] for r in embeddings})
    record(WARN if emb_coverage < len(rec_ids) else PASS,
           "Độ phủ embedding của reference DB",
           f"{emb_coverage}/{len(rec_ids)} recording có embedding "
           f"-> tầng 2 (MERT) chỉ tìm được trong {emb_coverage} bản ghi")

    # Rule Engine có 5 nhóm ưu tiên; dữ liệu hiện tại phủ được bao nhiêu nhóm?
    licenses = Counter(r["license_type"] for r in rights)
    statuses = Counter(r["copyright_status"] for r in rights)

    def group_of(license_type: str) -> str:
        if license_type == "AUDIO_LIBRARY":
            return "1_AUDIO_LIBRARY"
        if license_type == "CREATOR_MUSIC":
            return "2_CREATOR_MUSIC"
        if license_type in ("CONTENT_ID", "COMMERCIAL"):
            return "3_CONTENT_ID"
        if license_type.startswith("CC") or license_type == "CREATIVE_COMMONS":
            return "4_CREATIVE_COMMONS"
        if license_type == "PUBLIC_DOMAIN":
            return "5_PUBLIC_DOMAIN"
        return "6_UNCATEGORIZED"

    groups = Counter(group_of(r["license_type"]) for r in rights)
    required = {"1_AUDIO_LIBRARY", "2_CREATOR_MUSIC", "3_CONTENT_ID",
                "4_CREATIVE_COMMONS", "5_PUBLIC_DOMAIN"}
    absent = sorted(required - set(groups))
    if absent:
        record(WARN, "Độ phủ 5 nhóm của Rule Engine",
               f"thiếu {absent} -> các nhánh này chưa test được")
    else:
        record(PASS, "Độ phủ 5 nhóm của Rule Engine",
               ", ".join(f"{g}={n}" for g, n in sorted(groups.items())))

    if len(statuses) == 1:
        record(WARN, "Độ phủ copyright_status",
               f"toàn bộ {len(rights)} dòng đều là {list(statuses)[0]} "
               f"-> nhánh PUBLIC_DOMAIN không bao giờ kích hoạt")
    else:
        record(PASS, "Độ phủ copyright_status", str(dict(statuses)))

    # composition PD và recording PD phải được lưu ĐỘC LẬP (§2)
    if "recording_public_domain" in rights[0]:
        pd_rows = [r for r in rights if r["license_type"] == "PUBLIC_DOMAIN"]
        pd_recording = sum(1 for r in pd_rows
                           if str(r["recording_public_domain"]).lower() == "true")
        if pd_rows and 0 < pd_recording < len(pd_rows):
            record(PASS, "Tách bạch PD tác phẩm / PD bản thu",
                   f"{pd_recording}/{len(pd_rows)} bản thu là PD, phần còn lại "
                   f"vẫn được bảo hộ")
        else:
            record(WARN, "Tách bạch PD tác phẩm / PD bản thu",
                   "mọi bản thu PD giống nhau -> không kiểm được nhánh "
                   "RECORDING_PERMISSION_REQUIRED")
    else:
        record(WARN, "Cột recording_public_domain",
               "chưa có -> chạy scripts/enrich_rights_metadata.py")

    return report()


def report() -> int:
    width = max(len(name) for _, name, _ in results) + 2
    icon = {PASS: "✅", WARN: "⚠️ ", FAIL: "❌"}
    print("\n=== KIỂM TRA TOÀN VẸN DỮ LIỆU ===\n")
    for status, name, detail in results:
        print(f"{icon[status]} {status:4} | {name:<{width}} | {detail}")

    counts = Counter(status for status, _, _ in results)
    print(f"\nTổng kết: {counts[PASS]} PASS, {counts[WARN]} WARN, {counts[FAIL]} FAIL")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
