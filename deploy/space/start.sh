#!/bin/sh
# Điểm vào của container trên Hugging Face Spaces.
# 1. Tải chỉ mục FAISS (và chỉ mục Cover nếu có) từ dataset repo RUNTIME_DATA_REPO.
#    Tải hỏng thì server vẫn lên: /health báo DEGRADED kèm lý do, thay vì container
#    khởi động lại liên tục mà không ai thấy vì sao.
# 2. Chạy uvicorn. --no-proxy-headers: uvicorn KHÔNG tự thay IP client bằng
#    X-Forwarded-For; việc đó do backend/security.get_client_ip làm, theo
#    TRUSTED_PROXY_HOPS.
set -e
cd /app
python deploy/fetch_runtime_data.py || echo "fetch_runtime_data thất bại, server vẫn khởi động (xem /health)"
exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-7860}" --no-proxy-headers
