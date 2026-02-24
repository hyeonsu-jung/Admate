import logging
import asyncio
import time
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.services.vector_db import VectorDBService
from app.services.rag_chain import RagChain
from app.services.vision import get_vision_service
from app.services.monitor import get_monitor_service

logger = logging.getLogger(__name__)

class RagEngine:
    """텍스트 분할 및 RAG 핵심 로직을 담당하는 엔진 (싱글톤 패턴)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RagEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
        self.vector_db = VectorDBService()
        self.rag_chain = RagChain()
        self.monitor = get_monitor_service()
        self._initialized = True

    async def index_text(self, documents: List[Dict[str, Any]]):
        """텍스트 데이터를 즉시 인덱싱하여 사이드바에 반영 (빠른 응답용)"""
        all_chunks_to_index = []
        
        for doc in documents:
            if doc.get("content") and doc["content"].strip():
                texts = self.text_splitter.split_text(doc["content"])
                for i, chunk_text in enumerate(texts):
                    all_chunks_to_index.append({
                        "content": chunk_text,
                        "metadata": {
                            **doc["metadata"], 
                            "chunk_index": i, 
                            "data_type": "text",
                            "status": "indexing_images" # 텍스트는 완료, 이미지는 아직 분석 전 상태
                        }
                    })
        
        if all_chunks_to_index:
            # upsert_documents 내부에서 기존 source 삭제 로직이 있으므로 주의 (첫 텍스트 업로드 시에만 삭제 필요)
            count = await self.vector_db.upsert_documents(all_chunks_to_index)
            return count
        return 0

    async def index_images_background(self, documents: List[Dict[str, Any]]):
        """이미지를 백그라운드에서 분석하여 벡터 DB에 추가 (비동기 수행)"""
        vision = get_vision_service()
        vision_tasks = []
        semaphore = asyncio.Semaphore(5)
        
        for doc in documents:
            if doc.get("images"):
                for img_idx, img_bytes in enumerate(doc["images"]):
                    metadata = {
                        **doc["metadata"], 
                        "chunk_index": f"img_{img_idx}", 
                        "data_type": "image",
                        "status": "indexed" # 이미지까지 포함되면 최종 완료
                    }
                    async def analyze(bytes_data, meta):
                        async with semaphore:
                            try:
                                logger.info(f"Background analysis starting: {meta.get('source')} p.{meta.get('page')}")
                                desc = await vision.describe_image(bytes_data)
                                return {"content": f"[이미지 분석 내용]\n{desc}", "metadata": meta}
                            except Exception as e:
                                logger.error(f"Background vision error: {str(e)}")
                                return {"content": "[이미지 분석 실패: 서버 부하]", "metadata": meta}
                    
                    vision_tasks.append(analyze(img_bytes, metadata))

        if vision_tasks:
            logger.info(f"Starting background vision analysis for {len(vision_tasks)} images...")
            analyzed_images = await asyncio.gather(*vision_tasks)
            # 이미 텍스트 업로드 시 source가 삭제되었으므로, 여기서는 단순히 추가(Append)만 수행되어야 함
            # VectorDBService.upsert_documents 내부에서 source 별 삭제 로직이 있으므로 
            # 백그라운드 업로드 시에는 'status'가 'indexed'인 문서 전용 업로드 메서드나 옵션 필요
            # (현재 VectorDBService 구조상 source가 같으면 또 삭제하므로, 
            #  VectorDBService의 upsert_documents를 수정하여 partial update 지원하도록 하거나,
            #  이곳에서 직접 처리 로직 고민)
            
            # 임시 해결: upsert_documents에 skip_delete 옵션이 있다고 가정하고 호출 (뒤에 VectorDB 수정 예정)
            await self.vector_db.upsert_documents(analyzed_images, skip_delete=True)
            logger.info("Background vision analysis and indexing completed.")

    async def get_answer(self, query: str, filter_source: str = None):
        """질문에 대한 답변 생성 (벡터 검색 -> LLM 답변)"""
        start_time = time.time()
        try:
            # 1. 유사 문서 검색 (필터 적용)
            matches = await self.vector_db.search(query, top_k=3, filter_source=filter_source)
            context = "\n\n".join([m.metadata["text"] for m in matches]) if matches else "검색된 관련 문서가 없습니다."
            
            # 2. LLM 답변 생성
            answer = await self.rag_chain.generate_answer(query, context)
            
            # 3. 출처 및 신뢰도 정리
            sources = []
            if matches:
                for m in matches:
                    sources.append({
                        "name": m.metadata["source"],
                        "page": m.metadata.get("page"),
                        "score": round(float(m.score), 4) if hasattr(m, 'score') else 0.8
                    })
            
            # 4. 성능 로깅
            duration = time.time() - start_time
            self.monitor.log_request(query, duration, source_count=len(sources))
                
            return {"answer": answer, "sources": sources}
        except Exception as e:
            logger.error(f"Error in get_answer: {str(e)}")
            return {"answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}", "sources": []}

    async def get_streaming_answer(self, query: str, filter_source: str = None):
        """스트리밍 방식으로 답변 및 출처 정보 전달"""
        start_time = time.time()
        try:
            # 1. 유사 문서 검색
            matches = await self.vector_db.search(query, top_k=3, filter_source=filter_source)
            context = "\n\n".join([m.metadata["text"] for m in matches]) if matches else "검색된 관련 문서가 없습니다."
            
            # 2. 출처 정보 선제적 전송
            sources = []
            if matches:
                for m in matches:
                    sources.append({
                        "name": m.metadata["source"],
                        "page": m.metadata.get("page"),
                        "score": round(float(m.score), 4) if hasattr(m, 'score') else 0.8
                    })
            
            import json
            yield f"__SOURCES__:{json.dumps(sources)}\n"
            
            # 3. LLM 스트리밍 답변 생성
            async for chunk in self.rag_chain.astream_answer(query, context):
                yield chunk
            
            # 4. 성능 로깅
            duration = time.time() - start_time
            self.monitor.log_request(query, duration, source_count=len(sources))
                
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield f"Error: {str(e)}"

# 유틸리티 함수로 전역 인스턴스 제공
def get_rag_engine():
    return RagEngine()
