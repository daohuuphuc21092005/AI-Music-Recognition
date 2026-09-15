FROM python:3.12-slim

WORKDIR /app

# ffmpeg            -> tách âm thanh từ video (AudioService)
# libchromaprint-tools -> cung cấp fpcalc cho tầng 1 (Chromaprint).
#                      Thiếu nó thì nhận diện chính xác không hoạt động.
# libpq-dev, gcc    -> build psycopg2
RUN apt-get update && apt-get install -y \
        ffmpeg \
        libchromaprint-tools \
        libpq-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Kiểm tra sớm: thiếu fpcalc thì báo ngay lúc build thay vì lúc chạy
RUN fpcalc -version

# Khởi chạy FastAPI với Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
