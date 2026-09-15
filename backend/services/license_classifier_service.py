"""
Dự đoán giấy phép từ embedding MERT, cho bài KHÔNG có trong reference database.

⚠️ GIỚI HẠN ĐÃ ĐO, PHẢI ĐỌC TRƯỚC KHI TIN CON SỐ

EXP-09 đo đúng năng lực của cách tiếp cận này ở đúng tình huống nó được dùng
(nghệ sĩ chưa từng xuất hiện lúc train) và kết quả là KHÔNG ĐẠT: accuracy 0.382
so với baseline "luôn đoán lớp phổ biến nhất" 0.569 — tệ hơn đoán mò 18,7 điểm.
Bằng chứng cho cơ chế: cùng embedding đó dự đoán NGHỆ SĨ đạt 0.481 (gấp 4,4 lần
baseline) nhưng rơi về 0.000 với nghệ sĩ mới. Model nhận ra *ai hát* rồi tra bảng
giấy phép của người đó, chứ không đọc được gì về giấy phép từ âm thanh.

Vì vậy module này:

  * KHÔNG BAO GIỜ tự nâng/hạ mức rủi ro. Nó chỉ sinh một khối bằng chứng.
  * Luôn kèm `warning` và `beats_baseline=False` trong mọi output, để tầng trên
    và người đọc không thể vô tình dùng nó như một kết luận.
  * Trả `None` khi chưa train model — tuyệt đối không đoán bừa để lấp chỗ trống.

Model được xây theo yêu cầu rõ ràng của chủ dự án sau khi đã xem số liệu EXP-09.
Train lại bằng: python scripts/train_license_classifier.py
"""
import io
import json
import os

import numpy as np

from backend import config

MODEL_DIR = os.path.join(config.BASE_DIR, "models", "license")
MODEL_VERSION = "license_clf_v1"
DEFAULT_TARGET = "license_type"

_cache = {}


def _paths(target: str) -> tuple:
    stem = os.path.join(MODEL_DIR, f"{MODEL_VERSION}__{target}")
    return stem + ".joblib", stem + ".json"


def is_available(target: str = DEFAULT_TARGET) -> bool:
    model_path, meta_path = _paths(target)
    return os.path.exists(model_path) and os.path.exists(meta_path)


def load_model(target: str = DEFAULT_TARGET):
    """
    Nạp lười (model, metadata); trả (None, None) nếu chưa train.

    Nạp lười vì cùng lý do với `embedding_service`: chỉ `import backend.main` mà
    đã kéo theo joblib + sklearn thì /health và pytest đều phải trả giá, và lỗi
    nạp model sẽ làm sập ứng dụng ngay từ khâu import.
    """
    if target in _cache:
        return _cache[target]

    model_path, meta_path = _paths(target)
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        _cache[target] = (None, None)
        return _cache[target]

    try:
        import joblib
        model = joblib.load(model_path)
        with io.open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        _cache[target] = (model, metadata)
    except Exception:
        # Model hỏng thì coi như không có: thà thiếu dự đoán còn hơn dự đoán rác
        _cache[target] = (None, None)
    return _cache[target]


def predict(query_vector, target: str = DEFAULT_TARGET, top_n: int = 3) -> dict:
    """
    Dự đoán giấy phép từ một vector MERT đã chuẩn hoá L2.

    Trả None nếu chưa có model hoặc vector không hợp lệ. Khi có kết quả, output
    luôn mang theo phần `validation` và `warning` lấy thẳng từ metadata lúc train
    — không tách rời được con số khỏi giới hạn của nó.
    """
    model, metadata = load_model(target)
    if model is None or query_vector is None:
        return None

    vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
    expected = metadata.get("embedding_dim")
    if expected and vector.shape[1] != expected:
        return None

    norm = np.linalg.norm(vector) or 1.0
    vector = vector / norm

    try:
        probabilities = model.predict_proba(vector)[0]
    except Exception:
        return None

    order = np.argsort(probabilities)[::-1][:top_n]
    ranked = [{"license_type": str(model.classes_[i]),
               "probability": round(float(probabilities[i]), 4)} for i in order]

    validation = metadata.get("validation") or {}
    return {
        "predicted_license_type": ranked[0]["license_type"],
        "probability": ranked[0]["probability"],
        "top_n": ranked,
        "model_version": metadata.get("model_version", MODEL_VERSION),
        "target": target,
        "trained_on_recordings": metadata.get("n_train"),
        # Hai trường dưới đây là bắt buộc: chúng ngăn con số ở trên bị đọc như
        # một kết luận về bản quyền.
        "validation": {
            "protocol": validation.get("protocol"),
            "accuracy": validation.get("accuracy"),
            "baseline_accuracy": validation.get("baseline_accuracy"),
            "accuracy_vs_baseline": validation.get("accuracy_vs_baseline"),
            "beats_baseline": validation.get("beats_baseline", False),
        },
        "warning": metadata.get("warning"),
        "advisory_only": True,
    }
