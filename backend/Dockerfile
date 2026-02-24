# Admate AI Backend Dockerfile
FROM python:3.11-slim

# 시스템 의존성 설치 (PyMuPDF 등에 필요)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY . .

# 서버 실행 (Hugging Face Spaces는 기본적으로 7860 포트를 사용하거나 PORT 환경변수 제공)
# 0.0.0.0으로 열어야 외부 접속이 가능합니다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
