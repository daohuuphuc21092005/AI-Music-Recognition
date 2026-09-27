FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# ffmpeg               -> tách âm thanh từ video (AudioService)
# libchromaprint-tools -> fpcalc cho tầng 1 (Chromaprint). Thiếu nó thì nhận diện
#                         chính xác không hoạt động.
# Không cần gcc/libpq-dev: requirements dùng psycopg2-binary, đã kèm sẵn libpq.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libchromaprint-tools \
    && rm -rf /var/lib/apt/lists/*

# torch trên PyPI cho Linux là bản CUDA (kèm thư viện nvidia-*), nên image nặng vài
# GB. Nó vẫn chạy được trên máy không có GPU; DEVICE=auto tự chọn CPU khi đó.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Chạy bằng user thường. Mã nguồn thuộc root nên tiến trình web không sửa được
# chính nó; chỉ thư mục upload tạm và cache model là ghi được.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/temp_uploads /home/app/.cache/huggingface \
    && chown -R app:app /app/temp_uploads /home/app/.cache

COPY . .

USER app

# Kiểm tra sớm: thiếu fpcalc thì báo ngay lúc build thay vì lúc chạy
RUN fpcalc -version

EXPOSE 8000

# /health luôn trả HTTP 200 kể cả khi DEGRADED (CSDL sập, thiếu index) — phải đọc
# trường status thì healthcheck mới có nghĩa.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import json, sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8); sys.exit(0 if json.load(r).get('status') == 'ONLINE' else 1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
