"""
Trích xuất embedding MERT-v1-95M (768 chiều) cho một đoạn audio.

Thay đổi so với bản cũ: model được nạp LƯỜI (lazy) thay vì nạp ngay lúc import.
Trước đây chỉ cần `import backend.main` là đã kéo theo vài giây nạp model, khiến
cả /health lẫn pytest đều phải trả giá — và nếu tải model lỗi thì toàn bộ ứng
dụng sập ngay từ khâu import.
"""
import gc
import os
import sys

# Xử lý tương thích môi trường: nếu torchvision bị lỗi nhị phân C++ (ví dụ: operator torchvision::nms does not exist),
# cô lập torchvision để transformers không bị crash khi nạp dynamic module (audio retrieval không cần torchvision).
if "torchvision" not in sys.modules:
    try:
        import torchvision
    except Exception:
        sys.modules["torchvision"] = None

import librosa
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from backend import config

MODEL_NAME = config.MERT_MODEL

_processor = None
_model = None
_device = None


def is_loaded() -> bool:
    return _model is not None


def get_device() -> str:
    """
    Thiết bị chạy MERT: tự dò (có GPU CUDA thì dùng), ép bằng biến môi trường
    DEVICE=cpu|cuda.

    Vì sao cần: dựng embedding cho 24.375 bài trên 6 luồng CPU mất ~4 s/bài, tức
    khoảng 24 giờ. Cùng model trên GPU chỉ còn phần giải mã audio là đáng kể —
    và phần đó vẫn nằm ở CPU, nên đừng kỳ vọng nhanh hơn ~0,3 s/bài.
    """
    global _device
    if _device is None:
        choice = os.environ.get("DEVICE", "auto").strip().lower()
        if choice not in ("cpu", "cuda"):
            choice = "cuda" if torch.cuda.is_available() else "cpu"
        elif choice == "cuda" and not torch.cuda.is_available():
            print("⚠️  DEVICE=cuda nhưng torch không thấy GPU nào -> dùng CPU")
            choice = "cpu"
        _device = choice
    return _device


def load_model():
    """Nạp processor + model (idempotent). Gọi lúc startup để 'làm nóng'."""
    global _processor, _model
    if _model is None:
        _processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        _model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _model.eval()
        _model.to(get_device())
    return _processor, _model


def extract_mert_embedding(audio_path: str, target_sr: int = None,
                           max_duration: float = None):
    """
    Trả vector 768 chiều đã chuẩn hoá L2 (mean pooling theo trục thời gian),
    hoặc None nếu không xử lý được file.
    """
    target_sr = target_sr or config.MERT_SAMPLE_RATE
    max_duration = max_duration or config.MERT_MAX_DURATION

    try:
        processor, model = load_model()

        audio_array, _ = librosa.load(
            audio_path, sr=target_sr, mono=True, duration=max_duration
        )

        inputs = processor(
            audio_array, sampling_rate=target_sr, return_tensors="pt"
        ).to(get_device())

        # inference_mode: không lưu gradient/trạng thái trung gian -> tiết kiệm RAM
        with torch.inference_mode():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state

            # Mean pooling theo thời gian (chiến lược P1)
            track_embedding = torch.mean(embeddings, dim=1)

            # Chuẩn hoá L2 để inner product = cosine similarity
            norm_embedding = torch.nn.functional.normalize(track_embedding, p=2, dim=1)

            result_vector = norm_embedding.squeeze().cpu().numpy()

        del inputs, outputs, embeddings, track_embedding, norm_embedding, audio_array
        gc.collect()

        return result_vector
    except Exception as e:
        print(f"Lỗi extract MERT embedding: {e}")
        return None
