# # Dockerfile
# FROM python:3.11-slim

# # ===== Install ODBC Driver 18 (bookworm) =====
# # 使用 /etc/apt/keyrings + signed-by，避免 apt-key 被棄用造成的 GPG 錯誤
# RUN apt-get update && apt-get install -y --no-install-recommends \
#       curl gnupg ca-certificates apt-transport-https unixodbc-dev \
#  && mkdir -p /etc/apt/keyrings \
#  && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
#     | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg \
#  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
#     > /etc/apt/sources.list.d/microsoft-prod.list \
#  && apt-get update \
#  && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
#  && apt-get clean && rm -rf /var/lib/apt/lists/*

# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .
# ENV PORT=8000
# EXPOSE 8000

# # CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8000"]
# # CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["bash","-lc","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]

FROM python:3.12-slim

# 基本優化
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式
COPY . .

# （可選）告訴人類這個容器預期對外 8000
EXPOSE 8000

# 關鍵：綁 0.0.0.0，吃 Azure 的 PORT（沒給就用 8000）
CMD ["bash","-lc","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]

