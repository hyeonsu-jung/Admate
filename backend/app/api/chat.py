from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from app.services.rag_engine import get_rag_engine

router = APIRouter()
# 공유 인스턴스 사용
engine = get_rag_engine()

class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = "public"
    history: Optional[List[dict]] = []
    filter_source: Optional[str] = None

@router.post("/ask")
async def ask_question(request: ChatRequest):
    """사용자 질문에 대해 RAG 엔진을 사용해 답변 생성 (동기)"""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")
        
    result = await engine.get_answer(
        query=request.query, 
        history=request.history,
        filter_source=request.filter_source, 
        user_id=request.user_id
    )
    return result

@router.post("/ask/stream")
async def ask_question_stream(request: ChatRequest):
    """사용자 질문에 대해 RAG 엔진을 사용해 답변 생성 (스트리밍)"""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")
        
    return StreamingResponse(
        engine.get_streaming_answer(
            query=request.query, 
            history=request.history,
            filter_source=request.filter_source, 
            user_id=request.user_id
        ),
        media_type="text/event-stream"
    )
