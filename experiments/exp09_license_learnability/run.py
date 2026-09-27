"""
EXP-09 — Bản quyền có học được từ âm thanh không?

Câu hỏi (do người dùng đặt ra, và đáng được trả lời bằng số liệu chứ không bằng
nguyên tắc): nếu gắn nhãn giấy phép vào đặc trưng MERT rồi train, hệ thống có
đoán được tình trạng bản quyền của một bài KHÔNG có trong reference database không?

Đây chính là giả thuyết mà §2 bác bỏ ("license không phải một acoustic class") và
là nền của đóng góp C3. Cho tới nay nó mới chỉ được LẬP LUẬN. Thí nghiệm này đo nó.

Thiết kế — điểm mấu chốt nằm ở cách chia dữ liệu:

  A. Split ngẫu nhiên theo bản ghi
     Nghệ sĩ xuất hiện ở CẢ train lẫn test. Đây là cách chia mà một pipeline học
     máy thông thường sẽ dùng, và là con số mà người ta hay đem đi báo cáo.

  B. Split theo NGHỆ SĨ (StratifiedGroupKFold, groups = artist)
     Nghệ sĩ trong test chưa từng xuất hiện lúc train. Đây mới đúng tình huống
     "upload một bài lạ": hệ thống chưa từng nghe nghệ sĩ đó bao giờ.

  C. Baseline: luôn đoán lớp phổ biến nhất.

Vì sao khoảng cách A - B chính là thước đo của rò rỉ dữ liệu: trong corpus FMA
hiện tại 96,6% nghệ sĩ chỉ có ĐÚNG MỘT loại giấy phép. Nên một model chỉ cần nhận
ra nghệ sĩ là suy được giấy phép — mà MERT nhận diện bản ghi cực giỏi (EXP-02:
Recall@1 = 0.9515). Ở giao thức A nó tra bảng được; ở giao thức B thì không.

Thí nghiệm còn train thêm một bộ phân loại NGHỆ SĨ từ chính embedding đó. Nếu bộ
này chính xác cao, ta có bằng chứng TRỰC TIẾP cho cơ chế rò rỉ nói trên, chứ không
phải chỉ suy đoán.

Hai bài toán nhãn được thử:
  * license_type   — nhiều lớp (CC_BY, CC_BY_NC_SA, ...)
  * commercial_use — nhị phân, gộp mọi biến thể NC lại; dễ hơn hẳn, nên nếu ngay
                     cả bài toán này cũng không học được thì kết luận càng chắc.

Metrics: accuracy, macro-F1, và khoảng cách so với baseline.
"""
import csv
import io
import os
import sys
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier

from backend import config
from experiments.common import load_embedding_matrix, print_table, save_result

csv.field_size_limit(10 ** 9)

EXPERIMENT_ID = "exp09_license_learnability"
SEED = 42
FOLDS = 5


def load_labels() -> dict:
    """recording_id -> {artist, license_type, commercial_use}."""
    meta = {}
    with io.open(config.METADATA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["recording_id"]] = row

    rights_csv = os.path.join(config.DATA_DIR, "rights_master.csv")
    labels = {}
    with io.open(rights_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rec = meta.get(row["recording_id"])
            # Chỉ lấy bản ghi CÓ AUDIO THẬT: bản ghi mô phỏng không có embedding,
            # và giấy phép của chúng cũng là bịa nên đưa vào là làm hỏng nhãn.
            if not rec or not rec.get("audio_path"):
                continue
            labels[row["recording_id"]] = {
                "artist": (rec.get("artist") or "").strip() or "(không rõ)",
                "license_type": row["license_type"],
                "commercial_use": str(row["commercial_use_allowed"]).lower() == "true",
            }
    return labels


def per_recording_vectors(matrix, recording_ids, labels):
    """
    Gộp các segment của cùng bản ghi thành MỘT vector (trung bình, chuẩn hoá lại).

    Không dùng từng segment làm một mẫu độc lập: các segment của cùng bài cực
    giống nhau, để chúng nằm rải cả train lẫn test là thêm một dạng rò rỉ nữa,
    chồng lên đúng thứ mà thí nghiệm này đang muốn đo (§2 cấm split theo segment).
    """
    grouped = defaultdict(list)
    for vector, rec_id in zip(matrix, recording_ids):
        if rec_id in labels:
            grouped[rec_id].append(vector)

    ids, vectors = [], []
    for rec_id, group in grouped.items():
        mean = np.mean(group, axis=0)
        norm = np.linalg.norm(mean) or 1.0
        ids.append(rec_id)
        vectors.append(mean / norm)
    return ids, np.vstack(vectors).astype("float32")


def build_model(name: str):
    if name == "logistic":
        return LogisticRegression(max_iter=2000, C=1.0,
                                  class_weight="balanced", random_state=SEED)
    return MLPClassifier(hidden_layer_sizes=(256,), max_iter=600,
                         early_stopping=True, random_state=SEED)


MODELS = ("logistic", "mlp")


def cross_validate(X, y, groups, splitter, factory) -> dict:
    """
    Chạy CV, gộp toàn bộ dự đoán out-of-fold rồi mới tính chỉ số.

    StratifiedKFold bỏ qua `groups` (và cảnh báo nếu ta cứ truyền vào), còn
    StratifiedGroupKFold thì bắt buộc phải có — nên chỉ truyền khi splitter
    thật sự dùng tới.
    """
    uses_groups = isinstance(splitter, StratifiedGroupKFold)
    true_all, pred_all = [], []
    for train_idx, test_idx in splitter.split(X, y, groups if uses_groups else None):
        model = factory()
        model.fit(X[train_idx], y[train_idx])
        true_all.extend(y[test_idx])
        pred_all.extend(model.predict(X[test_idx]))
    return {
        "accuracy": round(float(accuracy_score(true_all, pred_all)), 4),
        "macro_f1": round(float(f1_score(true_all, pred_all,
                                         average="macro", zero_division=0)), 4),
        "n_test": len(true_all),
    }


def run_task(name: str, X, y, groups) -> dict:
    """Chạy đủ ba giao thức cho một bài toán nhãn."""
    print("\n" + "=" * 78)
    print(f"  Bài toán: {name}")
    print("=" * 78, flush=True)

    distribution = Counter(y)
    majority = distribution.most_common(1)[0]
    print(f"  {len(y)} mẫu, {len(distribution)} lớp | lớp lớn nhất chiếm "
          f"{majority[1] / len(y) * 100:.1f}%")

    results = {"n_samples": int(len(y)), "n_classes": len(distribution),
               "class_distribution": {str(k): int(v) for k, v in distribution.items()},
               "protocols": {}}

    random_split = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    artist_split = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED)

    baseline = cross_validate(X, y, groups, random_split,
                              lambda: DummyClassifier(strategy="most_frequent"))
    results["protocols"]["C_baseline"] = baseline
    print(f"  C. Baseline (đoán lớp phổ biến nhất): acc={baseline['accuracy']} "
          f"macroF1={baseline['macro_f1']}", flush=True)

    for model_name in MODELS:
        a = cross_validate(X, y, groups, random_split,
                           lambda: build_model(model_name))
        b = cross_validate(X, y, groups, artist_split,
                           lambda: build_model(model_name))

        results["protocols"][f"A_random_{model_name}"] = a
        results["protocols"][f"B_by_artist_{model_name}"] = b
        results["protocols"][f"gap_{model_name}"] = {
            "accuracy": round(a["accuracy"] - b["accuracy"], 4),
            "macro_f1": round(a["macro_f1"] - b["macro_f1"], 4),
        }
        print(f"  A. Ngẫu nhiên   [{model_name:8}]: acc={a['accuracy']} "
              f"macroF1={a['macro_f1']}")
        print(f"  B. Theo nghệ sĩ [{model_name:8}]: acc={b['accuracy']} "
              f"macroF1={b['macro_f1']}")
        print(f"     rò rỉ A-B: {a['accuracy'] - b['accuracy']:+.4f} | "
              f"B vượt baseline: {b['accuracy'] - baseline['accuracy']:+.4f}",
              flush=True)
    return results


def main() -> int:
    labels = load_labels()
    if not labels:
        print("❌ Không có bản ghi nào vừa có audio thật vừa có nhãn giấy phép.")
        return 1

    matrix, recording_ids, _segments = load_embedding_matrix()
    ids, X = per_recording_vectors(matrix, recording_ids, labels)
    if len(ids) < FOLDS * 2:
        print(f"❌ Chỉ có {len(ids)} bản ghi có embedding — quá ít để chia fold.")
        return 1

    print(f"Đặc trưng: {X.shape[0]} bản ghi x {X.shape[1]} chiều "
          f"(MERT mean-pooled, gộp theo bản ghi)")

    artists = np.array([labels[i]["artist"] for i in ids])
    by_artist = defaultdict(set)
    for rec_id in ids:
        by_artist[labels[rec_id]["artist"]].add(labels[rec_id]["license_type"])
    spread = sum(1 for licenses in by_artist.values() if len(licenses) > 1)
    print(f"Nghệ sĩ: {len(by_artist)} | có nhiều hơn 1 loại giấy phép: {spread} "
          f"({spread / len(by_artist) * 100:.1f}%)")

    tasks = {
        "license_type": np.array([labels[i]["license_type"] for i in ids]),
        "commercial_use": np.array(
            ["cho_phep" if labels[i]["commercial_use"] else "cam" for i in ids]),
        # Đối chứng: embedding có mã hoá NGHỆ SĨ không? Nếu có, đó chính là cơ
        # chế khiến giao thức A cho điểm cao mà không học được gì về bản quyền.
        "artist_doi_chung": artists,
    }

    metrics = {
        "n_recordings": len(ids),
        "n_artists": len(by_artist),
        "artists_with_multiple_licenses": spread,
        "tasks": {},
    }

    for name, y in tasks.items():
        counts = Counter(y)
        keep = np.array([counts[value] >= FOLDS for value in y])
        dropped = int(len(y) - keep.sum())
        if dropped:
            print(f"\n[{name}] bỏ {dropped} mẫu thuộc lớp có dưới {FOLDS} bản ghi "
                  f"(không chia nổi {FOLDS} fold)")
        if len(set(y[keep])) < 2:
            print(f"[{name}] còn dưới 2 lớp sau khi lọc -> bỏ qua")
            continue
        metrics["tasks"][name] = run_task(name, X[keep], y[keep], artists[keep])

    rows = []
    for name, task in metrics["tasks"].items():
        base = task["protocols"]["C_baseline"]["accuracy"]
        base_f1 = task["protocols"]["C_baseline"]["macro_f1"]
        for model_name in MODELS:
            a = task["protocols"][f"A_random_{model_name}"]
            b = task["protocols"][f"B_by_artist_{model_name}"]
            rows.append([name, model_name,
                         f"{base} / {base_f1}",
                         f"{a['accuracy']} / {a['macro_f1']}",
                         f"{b['accuracy']} / {b['macro_f1']}",
                         round(a["accuracy"] - b["accuracy"], 4),
                         f"{round(b['accuracy'] - base, 4):+} / "
                         f"{round(b['macro_f1'] - base_f1, 4):+}"])

    # Hiện CẢ accuracy lẫn macro-F1: với nhãn lệch 87/13 thì accuracy một mình
    # là con số dễ gây hiểu nhầm theo cả hai chiều — baseline "luôn đoán lớp
    # lớn" đạt accuracy rất cao nhưng macro-F1 thấp, còn model dùng
    # class_weight="balanced" thì ngược lại. Muốn kết luận trung thực phải nhìn
    # cả hai.
    print_table(
        "EXP-09 — Bản quyền có học được từ âm thanh không?  (accuracy / macro-F1)",
        rows,
        ["Bài toán", "Model", "C. Baseline", "A. Ngẫu nhiên", "B. Theo nghệ sĩ",
         "Rò rỉ acc", "B vượt baseline"],
    )

    path = save_result(
        EXPERIMENT_ID,
        params={
            "features": "MERT-v1-95M mean-pooled, gộp theo bản ghi, chuẩn hoá L2",
            "embedding_dim": int(X.shape[1]),
            "folds": FOLDS,
            "seed": SEED,
            "protocol_A": "StratifiedKFold theo bản ghi (nghệ sĩ có ở cả hai bên)",
            "protocol_B": "StratifiedGroupKFold theo NGHỆ SĨ (nghệ sĩ trong test là mới)",
            "protocol_C": "DummyClassifier(most_frequent)",
            "models": list(MODELS),
        },
        metrics=metrics,
        notes=(
            "Thí nghiệm trả lời bằng số liệu câu hỏi 'gắn nhãn bản quyền vào đặc "
            "trưng rồi train thì có đoán được bài lạ không'. Con số cần đọc là cột "
            "'B vượt baseline': đó là phần model THỰC SỰ học được về giấy phép khi "
            "gặp nghệ sĩ chưa từng thấy. Cột 'Rò rỉ (A-B)' là phần hiệu năng đến từ "
            "việc nhận ra nghệ sĩ rồi tra bảng chứ không phải từ bản quyền — bài "
            "toán đối chứng 'artist_doi_chung' cho thấy embedding mã hoá danh tính "
            "nghệ sĩ mạnh tới mức nào. Kết quả này là bằng chứng thực nghiệm cho "
            "đóng góp C3 (§1) và cho nguyên tắc §2 'license không phải acoustic class'."
        ),
    )
    print(f"\n💾 Đã lưu kết quả: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
