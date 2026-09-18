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

Độ phức tạp (`--sweep`): quét C — nghịch đảo độ mạnh regularization — theo đánh đổi
bias–variance, chọn C có tổng lỗi kiểm định thấp nhất rồi mới train model cuối.

Dùng:
    python scripts/train_license_classifier.py --sweep
    python scripts/train_license_classifier.py --C 0.1
    python scripts/train_license_classifier.py --target commercial_use --sweep
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
from experiments.common import save_result

csv.field_size_limit(10 ** 9)

MODEL_DIR = os.path.join(config.BASE_DIR, "models", "license")
MODEL_VERSION = "license_clf_v1"
FOLDS = 5
SEED = 42
DEFAULT_C = 1.0
# C nhỏ ép trọng số về 0: mô hình đơn giản, lỗi độ chênh cao. C lớn để mô hình bám
# sát tập train: lỗi phương sai cao. Thang log vì hiệu ứng của C là theo bậc độ lớn.
# Nới tới 3000 (2026-09-18): lưới cũ dừng ở 100 trong khi tổng lỗi vẫn giảm tới đúng
# mức đó, nên "chọn C = 100" có thể chỉ là giới hạn của lưới chứ không phải tối ưu.
COMPLEXITY_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
                   300.0, 1000.0, 3000.0)
MAX_ITER = 2000
SWEEP_EXPERIMENT_ID = "exp09_license_complexity_sweep"
# Nhãn từ các nguồn này không phải giấy phép thật — học từ chúng là học nhiễu.
UNTRUSTED_LABEL_SOURCES = ("SIMULATED", "PREDICTED")


def load_dataset(target: str):
    """Trả (X, y, groups=artist, ids). Chỉ lấy bản ghi có audio thật và nhãn thật."""
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
            if str(row.get("source") or "").upper().startswith(UNTRUSTED_LABEL_SOURCES):
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


def build_model(C: float = DEFAULT_C):
    return LogisticRegression(max_iter=MAX_ITER, C=C,
                              class_weight="balanced", random_state=SEED)


def validate_by_artist(X, y, groups, C: float = DEFAULT_C) -> dict:
    """
    Kiểm định bằng split theo NGHỆ SĨ — con số duy nhất phản ánh tình huống thật.

    Không báo cáo split ngẫu nhiên ở đây: nó cho điểm cao hơn chỉ vì nghệ sĩ nằm
    ở cả hai bên, và người đọc rất dễ tưởng đó là năng lực thật của model.
    Điểm trên tập train được đo kèm để thấy khoảng cách train↔kiểm định.
    """
    counts = Counter(y)
    keep = np.array([counts[v] >= FOLDS for v in y])
    X, y, groups = X[keep], y[keep], groups[keep]

    splitter = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED)
    true_all, pred_all = [], []
    train_accuracy, train_macro_f1, iterations = [], [], []
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = build_model(C)
        model.fit(X[train_idx], y[train_idx])
        iterations.append(int(np.max(model.n_iter_)))
        fitted = model.predict(X[train_idx])
        train_accuracy.append(accuracy_score(y[train_idx], fitted))
        train_macro_f1.append(f1_score(y[train_idx], fitted, average="macro", zero_division=0))
        true_all.extend(y[test_idx])
        pred_all.extend(model.predict(X[test_idx]))

    majority_label, majority_count = Counter(y).most_common(1)[0]
    majority = majority_count / len(y)
    accuracy = float(accuracy_score(true_all, pred_all))
    macro_f1 = float(f1_score(true_all, pred_all, average="macro", zero_division=0))
    baseline_macro_f1 = float(f1_score(true_all, [majority_label] * len(true_all),
                                       average="macro", zero_division=0))
    return {
        "protocol": "StratifiedGroupKFold theo nghệ sĩ (nghệ sĩ trong test là mới)",
        "folds": FOLDS,
        "C": C,
        "n_eval": len(true_all),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "train_accuracy": round(float(np.mean(train_accuracy)), 4),
        "train_macro_f1": round(float(np.mean(train_macro_f1)), 4),
        "baseline_accuracy": round(majority, 4),
        "baseline_macro_f1": round(baseline_macro_f1, 4),
        "accuracy_vs_baseline": round(accuracy - majority, 4),
        "beats_baseline": bool(accuracy > majority),
        "macro_f1_beats_baseline": bool(macro_f1 > baseline_macro_f1),
        # Chạm trần vòng lặp thì con số là của một mô hình CHƯA hội tụ — thường gặp ở
        # C lớn (regularization yếu). Ghi lại để không đọc nhầm thành tối ưu thật.
        "max_iterations_used": max(iterations),
        "converged": bool(max(iterations) < MAX_ITER),
    }


def complexity_sweep(X, y, groups) -> dict:
    """
    Quét độ phức tạp theo đánh đổi bias–variance, chọn C có tổng lỗi thấp nhất.

    Đây là xấp xỉ, không phải phân rã chính xác: lỗi trên tập train đại diện cho
    lỗi độ chênh, khoảng cách train↔kiểm định đại diện cho lỗi phương sai, lỗi
    kiểm định (nghệ sĩ mới) là tổng lỗi. Tính theo macro-F1 vì các lớp giấy phép
    lệch mạnh — accuracy sẽ thưởng cho việc luôn đoán lớp phổ biến.
    """
    grid = []
    for C in COMPLEXITY_GRID:
        result = validate_by_artist(X, y, groups, C)
        bias = 1.0 - result["train_macro_f1"]
        total = 1.0 - result["macro_f1"]
        grid.append({**result,
                     "bias_error": round(bias, 4),
                     "variance_error": round(total - bias, 4),
                     "total_error": round(total, 4)})
        print(f"  C={C:<7g} train_f1={result['train_macro_f1']:.4f}  "
              f"val_f1={result['macro_f1']:.4f}  lỗi độ chênh={bias:.4f}  "
              f"lỗi phương sai={total - bias:.4f}  tổng lỗi={total:.4f}"
              f"{'' if result['converged'] else '  (CHƯA hội tụ)'}", flush=True)
    # Hoà điểm thì chọn C nhỏ hơn: cùng lỗi thì mô hình đơn giản hơn an toàn hơn.
    selected = min(grid, key=lambda row: (row["total_error"], row["C"]))
    at_edge = selected["C"] == max(row["C"] for row in grid)
    if at_edge:
        print(f"⚠️  C được chọn ({selected['C']:g}) nằm ở RÌA lưới — tổng lỗi có thể còn giảm "
              f"tiếp; nới lưới trước khi tin đây là tối ưu.")
    return {"grid": grid, "selected": selected, "selected_at_grid_edge": at_edge}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="license_type",
                        choices=["license_type", "commercial_use"])
    parser.add_argument("--C", type=float, default=DEFAULT_C,
                        help="Nghịch đảo độ mạnh regularization (bỏ qua khi --sweep)")
    parser.add_argument("--sweep", action="store_true",
                        help="Quét độ phức tạp theo bias–variance rồi train với C tốt nhất")
    args = parser.parse_args()

    ids, X, y, groups = load_dataset(args.target)
    if len(ids) < FOLDS * 2:
        print(f"❌ Chỉ có {len(ids)} bản ghi có embedding — quá ít để train.")
        return 1

    print(f"Dữ liệu: {X.shape[0]} bản ghi x {X.shape[1]} chiều | "
          f"{len(set(y))} lớp | {len(set(groups))} nghệ sĩ")
    for label, count in Counter(y).most_common():
        print(f"  {label:<16}{count:>5}  ({count / len(y) * 100:.1f}%)")

    sweep = None
    if args.sweep:
        print(f"\nSweep độ phức tạp: {len(COMPLEXITY_GRID)} mức C, split theo nghệ sĩ...")
        sweep = complexity_sweep(X, y, groups)
        validation = sweep["selected"]
        path = save_result(
            SWEEP_EXPERIMENT_ID,
            params={
                "target": args.target,
                "model": "LogisticRegression(class_weight=balanced)",
                "grid_C": list(COMPLEXITY_GRID),
                "folds": FOLDS,
                "seed": SEED,
                "features": "MERT-v1-95M mean-pooled theo bản ghi, chuẩn hoá L2",
                "n_recordings": int(X.shape[0]),
                "n_artists": len(set(groups)),
                "label_sources": "loại nhãn có source SIMULATED/PREDICTED",
            },
            metrics={"selected_C": validation["C"], "selected": validation,
                     "selected_at_grid_edge": sweep["selected_at_grid_edge"],
                     "grid": sweep["grid"]},
            notes=("Lỗi độ chênh ≈ 1 − macro-F1 trên tập train; lỗi phương sai ≈ khoảng "
                   "cách train↔kiểm định; tổng lỗi = 1 − macro-F1 kiểm định với nghệ sĩ "
                   "mới. Chọn C có tổng lỗi thấp nhất, hoà thì chọn C nhỏ hơn."),
        )
        print(f"\n💾 Kết quả sweep: {path}")
    else:
        print("\nKiểm định (split theo nghệ sĩ)...")
        validation = validate_by_artist(X, y, groups, args.C)

    chosen_C = validation["C"]
    for key in ("C", "accuracy", "macro_f1", "train_macro_f1", "baseline_accuracy",
                "baseline_macro_f1", "accuracy_vs_baseline"):
        print(f"  {key:<22} {validation[key]}")

    if not validation["beats_baseline"]:
        print("\n⚠️  MODEL KHÔNG VƯỢT BASELINE khi gặp nghệ sĩ chưa từng thấy.")
        print("   Vẫn train và lưu theo yêu cầu, nhưng cảnh báo này được ghi vào")
        print("   metadata và sẽ đi kèm MỌI dự đoán mà service trả ra.")

    print(f"\nTrain trên toàn bộ dữ liệu với C={chosen_C:g}...")
    model = build_model(chosen_C)
    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{MODEL_VERSION}__{args.target}.joblib")
    joblib.dump(model, model_path)

    verdict = "VƯỢT" if validation["beats_baseline"] else "KHÔNG vượt"
    metadata = {
        "model_version": MODEL_VERSION,
        "target": args.target,
        "algorithm": f"LogisticRegression(class_weight=balanced, C={chosen_C:g})",
        "features": "MERT-v1-95M mean-pooled theo bản ghi, chuẩn hoá L2",
        "embedding_dim": int(X.shape[1]),
        "mert_model_version": config.MERT_MODEL_VERSION,
        "n_train": int(X.shape[0]),
        "n_artists": len(set(groups)),
        "classes": sorted(set(y.tolist())),
        "class_distribution": {k: int(v) for k, v in Counter(y).items()},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "validation": validation,
        "complexity_sweep": ({"experiment_id": SWEEP_EXPERIMENT_ID,
                              "grid_C": list(COMPLEXITY_GRID),
                              "selected_C": chosen_C} if sweep else None),
        "warning": (
            f"Kiểm định theo nghệ sĩ chưa từng thấy: accuracy {validation['accuracy']} so "
            f"với baseline {validation['baseline_accuracy']} ({verdict} baseline) — đúng "
            "tình huống 'bài ngoài cơ sở dữ liệu' mà bộ phân loại được dùng. Giấy phép là "
            "thuộc tính pháp lý gắn với hợp đồng, không gắn với tín hiệu âm thanh. Mọi dự "
            "đoán chỉ nên đọc như một gợi ý cần người kiểm tra."
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
