# Admate AI Project Root Dockerfile (for Hugging Face Spaces)
FROM python:3.11-slim

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# backend 폴더의 데이터 복사 및 설치
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# backend 소스 코드 전체 복사
COPY backend/ .

# 서버 실행 (포트 7860 고정)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
