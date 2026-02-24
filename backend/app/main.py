import os
import ssl
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, docs
from app.core.config import settings

# 윈도우/사내망 환경의 SSL 인증서 오류 전역 우회
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['PYTHONHTTPSVERIFY'] = '0'
except Exception:
    pass

app = FastAPI(
    title="Admate RAG Chatbot API",
    description="광고 운영 효율화를 위한 RAG 기반 문서 검색 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(docs.router, prefix="/api/v1/docs", tags=["Documents"])

@app.get("/")
async def root():
    return {"message": "Admate RAG Chatbot API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
