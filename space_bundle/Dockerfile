# Image cho Hugging Face Spaces (Docker SDK, gói CPU miễn phí: 2 vCPU, 16 GB RAM).
# Khác Dockerfile ở gốc repo (dùng cho docker compose trên máy có GPU) ở bốn điểm:
#   - torch bản CPU: Space miễn phí không có GPU, bản CUDA trên PyPI nặng thêm vài GB;
#   - chạy bằng uid 1000, đúng uid Hugging Face dùng để chạy container;
#   - trọng số MERT được tải sẵn vào image, Space ngủ dậy không phải tải lại ~380 MB;
#   - chỉ mục FAISS không nằm trong image: deploy/space/start.sh tải từ một dataset
#     repo trên Hugging Face lúc khởi động (file 150 MB, không vào được git).
# Build context là thư mục do deploy/build_space_bundle.py dựng, không phải gốc repo.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libchromaprint-tools \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user

WORKDIR /app

# torch CPU cài TRƯỚC, đúng phiên bản ghim trong requirements.txt; lượt cài sau coi
# torch==X đã thoả (bản +cpu cùng số hiệu) nên không kéo bản CUDA về nữa.
COPY requirements.txt .
RUN TORCH_VERSION="$(grep -i '^torch==' requirements.txt | cut -d= -f3)" \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
    && pip install -r requirements.txt

COPY . .
RUN mkdir -p /app/temp_uploads /app/data/processed "$HF_HOME" \
    && chown -R user:user /app/temp_uploads /app/data "$HOME"

USER user

# Kiểm sớm lúc build: thiếu fpcalc thì tầng 1 không chạy
RUN fpcalc -version
# Tải MERT đúng revision ghim trong backend/config.py
RUN DEVICE=cpu python -c "from backend.services import embedding_service as e; e.load_model()"

# Cấu hình cho trang công khai (ghi đè được bằng Variables của Space):
#   EVIDENCE_DETAIL=public   ẩn tên file người khác đã gửi (sổ) và ngưỡng nội bộ
#   TRUSTED_PROXY_HOPS=1     Space đứng sau một reverse proxy của Hugging Face
ENV DEVICE=cpu \
    PORT=7860 \
    EVIDENCE_DETAIL=public \
    TRUSTED_PROXY_HOPS=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import json, os, sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '7860') + '/health', timeout=8); sys.exit(0 if json.load(r).get('status') == 'ONLINE' else 1)"

CMD ["sh", "deploy/space/start.sh"]
