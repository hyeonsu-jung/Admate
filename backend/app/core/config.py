from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "rag-chatbot"
    
    # RAG Settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # App Settings
    PROJECT_NAME: str = "Admate RAG Chatbot"
    DEBUG: bool = True

    # .env 파일 로드 설정
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# 진단용 출력
if __name__ == "__main__":
    print(f"DEBUG: settings.PINECONE_API_KEY length = {len(settings.PINECONE_API_KEY)}")
    print(f"DEBUG: PINECONE_KEY starts with: {settings.PINECONE_API_KEY[:10]}...")
