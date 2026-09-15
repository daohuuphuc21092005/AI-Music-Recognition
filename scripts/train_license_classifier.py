"""
Train bộ phân loại giấy phép từ embedding MERT, dùng cho bài NGOÀI reference DB.

⚠️ ĐỌC TRƯỚC KHI DÙNG SỐ LIỆU CỦA MODEL NÀY

EXP-09 đã đo chính xác khả năng của cách tiếp cận này và kết quả là KHÔNG ĐẠT:

  * Split theo nghệ sĩ (đúng tình huống "bài lạ"): accuracy 0.382 so với baseline
    "luôn đoán lớp phổ biến nhất" 0.569 — tức TỆ HƠN đoán mò 18,7 điểm.
  * MLP hội tụ về đúng baseline từng con số, nghĩa là không tìm được tín hiệu nào.
  * Đối chứng quyết định: cùng embedding đó dự đoán NGHỆ SĨ đạt 0.481 (gấp 4,4 lần
    baseline), nhưng rơi về 0.000 khi nghệ sĩ trong test là người mới. Model nhận
    ra *ai hát* rồi tra bảng, chứ không đọc được gì về *giấy phép*.

Model vẫn được xây theo yêu cầu rõ ràng của chủ dự án, và mọi đầu ra của nó đều
mang kèm cảnh báo trên. Lý do gốc rễ (§2): giấy phép là thuộc tính PHÁP LÝ gắn với
hợp đồng, không gắn với tín hiệu — cùng một file WAV có thể là CC-BY hôm nay và
độc quyền thương mại tháng sau mà không đổi một bit.

Chọn LogisticRegression chứ không phải MLP: MLP thoái hoá hoàn toàn về baseline nên
không dùng được; logistic ít nhất còn cho macro-F1 nhỉnh hơn baseline (+0.075) nhờ
`class_weight="balanced"`, tức là có phân biệt được chút ít giữa các lớp hiếm.

Dùng:
    python scripts/train_license_classifier.py
    python scripts/train_license_classifier.py --target commercial_use
"""
import argparse
import csv
import io
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from backend import config

csv.field_size_limit(10 ** 9)

MODEL_DIR = os.path.join(config.BASE_DIR, "models", "license")
MODEL_VERSION = "license_clf_v1"
FOLDS = 5
SEED = 42


def load_dataset(target: str):
    """Trả (X, y, groups=artist, ids). Chỉ lấy bản ghi có audio thật."""
    import faiss

    meta = {}
    with io.open(config.METADATA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["recording_id"]] = row

    labels = {}
    rights_csv = os.path.join(config.DATA_DIR, "rights_master.csv")
    with io.open(rights_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rec = meta.get(row["recording_id"])
            if not rec or not rec.get("audio_path"):
                continue
            if target == "commercial_use":
                value = ("cho_phep"
                         if str(row["commercial_use_allowed"]).lower() == "true"
                         else "cam")
            else:
                value = row["license_type"]
            labels[row["recording_id"]] = {
                "artist": (rec.get("artist") or "").strip() or "(không rõ)",
                "label": value,
            }

    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with io.open(config.FAISS_ID_MAP_PATH, encoding="utf-8") as f:
        recording_ids = json.load(f)["recording_ids"]
    matrix = index.reconstruct_n(0, index.ntotal).astype("float32")

    # Gộp segment của cùng bản ghi thành một vector (giống EXP-09, để con số
    # kiểm định của EXP-09 áp dụng được cho đúng model này)
    grouped = defaultdict(list)
    for vector, rec_id in zip(matrix, recording_ids):
        if rec_id in labels:
            grouped[rec_id].append(vector)

    ids, X, y, groups = [], [], [], []
    for rec_id, vectors in grouped.items():
        mean = np.mean(vectors, axis=0)
        mean = mean / (np.linalg.norm(mean) or 1.0)
        ids.append(rec_id)
        X.append(mean)
        y.append(labels[rec_id]["label"])
        groups.append(labels[rec_id]["artist"])
    return ids, np.vstack(X).astype("float32"), np.array(y), np.array(groups)


def build_model():
    return LogisticRegression(max_iter=2000, C=1.0,
                              class_weight="balanced", random_state=SEED)


def validate_by_artist(X, y, groups) -> dict:
    """
    Kiểm định bằng split theo NGHỆ SĨ — con số duy nhất phản ánh tình huống thật.

    Không báo cáo split ngẫu nhiên ở đây: nó cho điểm cao hơn chỉ vì nghệ sĩ nằm
    ở cả hai bên, và người đọc rất dễ tưởng đó là năng lực thật của model.
    """
    counts = Counter(y)
    keep = np.array([counts[v] >= FOLDS for v in y])
    X, y, groups = X[keep], y[keep], groups[keep]

    splitter = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED)
    true_all, pred_all = [], []
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = build_model()
        model.fit(X[train_idx], y[train_idx])
        true_all.extend(y[test_idx])
        pred_all.extend(model.predict(X[test_idx]))

    majority = Counter(y).most_common(1)[0][1] / len(y)
    accuracy = float(accuracy_score(true_all, pred_all))
    return {
        "protocol": "StratifiedGroupKFold theo nghệ sĩ (nghệ sĩ trong test là mới)",
        "folds": FOLDS,
        "n_eval": len(true_all),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(f1_score(true_all, pred_all,
                                         average="macro", zero_division=0)), 4),
        "baseline_accuracy": round(majority, 4),
        "accuracy_vs_baseline": round(accuracy - majority, 4),
        "beats_baseline": bool(accuracy > majority),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="license_type",
                        choices=["license_type", "commercial_use"])
    args = parser.parse_args()

    ids, X, y, groups = load_dataset(args.target)
    if len(ids) < FOLDS * 2:
        print(f"❌ Chỉ có {len(ids)} bản ghi có embedding — quá ít để train.")
        return 1

    print(f"Dữ liệu: {X.shape[0]} bản ghi x {X.shape[1]} chiều | "
          f"{len(set(y))} lớp | {len(set(groups))} nghệ sĩ")
    for label, count in Counter(y).most_common():
        print(f"  {label:<16}{count:>5}  ({count / len(y) * 100:.1f}%)")

    print("\nKiểm định (split theo nghệ sĩ)...")
    validation = validate_by_artist(X, y, groups)
    for key in ("accuracy", "macro_f1", "baseline_accuracy", "accuracy_vs_baseline"):
        print(f"  {key:<22} {validation[key]}")

    if not validation["beats_baseline"]:
        print("\n⚠️  MODEL KHÔNG VƯỢT BASELINE khi gặp nghệ sĩ chưa từng thấy.")
        print("   Vẫn train và lưu theo yêu cầu, nhưng cảnh báo này được ghi vào")
        print("   metadata và sẽ đi kèm MỌI dự đoán mà service trả ra.")

    print("\nTrain trên toàn bộ dữ liệu...")
    model = build_model()
    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{MODEL_VERSION}__{args.target}.joblib")
    joblib.dump(model, model_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "target": args.target,
        "algorithm": "LogisticRegression(class_weight=balanced, C=1.0)",
        "features": "MERT-v1-95M mean-pooled theo bản ghi, chuẩn hoá L2",
        "embedding_dim": int(X.shape[1]),
        "mert_model_version": config.MERT_MODEL_VERSION,
        "n_train": int(X.shape[0]),
        "n_artists": len(set(groups)),
        "classes": sorted(set(y.tolist())),
        "class_distribution": {k: int(v) for k, v in Counter(y).items()},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "validation": validation,
        "warning": (
            "EXP-09: bộ phân loại này KHÔNG vượt baseline khi gặp nghệ sĩ chưa từng "
            "thấy — đúng tình huống 'bài ngoài cơ sở dữ liệu' mà nó được dùng. "
            "Giấy phép là thuộc tính pháp lý gắn với hợp đồng, không gắn với tín "
            "hiệu âm thanh. Mọi dự đoán chỉ nên đọc như một gợi ý cần người kiểm tra."
        ),
    }
    meta_path = os.path.join(MODEL_DIR, f"{MODEL_VERSION}__{args.target}.json")
    with io.open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Model    : {model_path}")
    print(f"💾 Metadata : {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
