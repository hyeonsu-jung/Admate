import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings
import os
import pickle
import logging
import httpx
import ssl
import sys
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# 윈도우/사내망 환경의 SSL 인증서 오류(Self-signed certificate)를 전역적으로 해결
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['PYTHONHTTPSVERIFY'] = '0'  # 파이썬 전역 SSL 검증 비활성화
except Exception:
    pass

# 로그 설정
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class VectorDBService:
    """Pinecone 및 영구 로컬 캐시 병행 서비스"""
    
    def __init__(self):
        self.index = None
        self.enabled = False
        self.local_cache = []
        # 파일 경로 절대화 (서버 실행 위치에 구애받지 않도록)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.cache_file = os.path.join(base_dir, "vector_cache.pkl")
        logger.info(f"Initialized VectorDBService. Cache file path: {self.cache_file}")
        
        # 1. 임베딩 엔진 준비 (OpenAI)
        # Windows/사내망 환경의 SSL 인증서 오류 대응 (동기/비동기 모두 우회)
        # httpx 클라이언트를 생성하여 수동으로 전달
        async_client = httpx.AsyncClient(verify=False, timeout=60.0)
        sync_client = httpx.Client(verify=False, timeout=60.0)
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL,
            http_async_client=async_client,
            http_client=sync_client
        )

        # 2. 로컬 파일 캐시 로드 (서버 재시작 대비)
        self._load_cache()

        # 3. Pinecone 연결 시도
        try:
            pk = settings.PINECONE_API_KEY
            if pk and "your_pinecone" not in pk:
                self.pc = Pinecone(api_key=pk)
                self.index_name = settings.PINECONE_INDEX_NAME
                
                indexes = [i.name for i in self.pc.list_indexes()]
                if self.index_name in indexes:
                    self.index = self.pc.Index(self.index_name)
                    self.enabled = True
                    logger.info(f"Connected to Pinecone: {self.index_name}")
                else:
                    logger.warning(f"Index '{self.index_name}' not found in Pinecone. Available: {indexes}")
        except Exception as e:
            logger.error(f"Pinecone init failed: {str(e)}")

    def _save_cache(self):
        """로컬 캐시를 파일로 저장하여 영속성 유지"""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.local_cache, f)
            logger.info(f"Knowledge cache saved to file: {len(self.local_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {str(e)}")

    def _load_cache(self):
        """저장된 파일에서 캐시 로드"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self.local_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.local_cache)} entries from local cache file.")
            except Exception as e:
                logger.error(f"Failed to load cache: {str(e)}")

    async def upsert_documents(self, chunks: List[Dict[str, Any]], skip_delete: bool = False):
        """문서를 벡터로 변환하여 저장 (skip_delete 옵션으로 부분 업데이트 지원)"""
        if not chunks:
            logger.warning("No chunks provided for upsert.")
            return 0
            
        # 0. 동일 출처(source)의 기존 데이터 삭제 (최초 업로드 시에만 수행)
        if not skip_delete:
            sources = list(set(c["metadata"]["source"] for c in chunks))
            for source in sources:
                self.delete_document(source)
            
        count = 0
        batch_size = 20
        total_chunks = len(chunks)
        upload_time = datetime.now().isoformat()
        
        logger.info(f"Starting upsert for {total_chunks} chunks in batches of {batch_size}")
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = [c["content"] for c in batch_chunks]
            
            try:
                logger.info(f"Processing batch {(i // batch_size) + 1}/{(total_chunks + batch_size - 1) // batch_size}...")
                
                # 배치 임베딩 생성
                vectors = await self.embeddings.aembed_documents(batch_texts)
                
                vectors_to_upsert = []
                for j, vector in enumerate(vectors):
                    chunk = batch_chunks[j]
                    data_node = {
                        "id": f"{chunk['metadata']['source']}_{i + j}",
                        "values": vector,
                        "metadata": {
                            "text": chunk["content"], 
                            **chunk["metadata"],
                            "uploaded_at": upload_time,
                            "status": "indexed"
                        }
                    }
                    self.local_cache.append(data_node)
                    
                    if self.enabled:
                        vectors_to_upsert.append({
                            "id": data_node["id"],
                            "values": vector,
                            "metadata": data_node["metadata"]
                        })
                    count += 1

                # 각 배치 완료 후 파일에 즉시 중간 저장 (안전성)
                self._save_cache()

                # Pinecone 업서트
                if self.enabled and vectors_to_upsert:
                    self.index.upsert(vectors=vectors_to_upsert)
                    logger.info(f"Batch {i//batch_size + 1} synced to Pinecone.")

            except Exception as e:
                logger.error(f"Error in batch starting at index {i}: {str(e)}")
                if "SSL" in str(e):
                    logger.error("SSL Verification Failed. Please check your network/VPN.")
                # 한 배치가 실패하더라도 다음 배치 계속 시도 (또는 예외 전파 선택)
                continue

        logger.info(f"Upsert process finished. Total successfully added: {count}")
        return count

    def delete_document(self, source_name: str):
        """특정 소스의 문서를 캐시 및 Pinecone에서 삭제"""
        logger.info(f"Attempting to delete document segments for: {source_name}")
        
        # 1. 로컬 캐시에서 삭제
        initial_count = len(self.local_cache)
        self.local_cache = [item for item in self.local_cache if item["metadata"].get("source") != source_name]
        deleted_count = initial_count - len(self.local_cache)
        
        # 2. Pinecone에서 삭제
        if self.enabled:
            try:
                # 메타데이터 'source' 필터를 사용하여 삭제
                self.index.delete(filter={"source": {"$eq": source_name}})
                logger.info(f"Deleted from Pinecone: {source_name}")
            except Exception as e:
                logger.error(f"Pinecone delete failed for {source_name}: {str(e)}")
        
        self._save_cache()
        logger.info(f"Deleted {deleted_count} chunks from local cache for {source_name}")
        return True

    async def search(self, query: str, top_k: int = 3, filter_source: str = None):
        """검색 (Pinecone 우선, 로컬 폴백) - 필터링 지원 추가"""
        try:
            logger.info(f"Searching: '{query}' (Filter: {filter_source})")
            query_vector = self.embeddings.embed_query(query)
            
            # 1. Pinecone 검색
            if self.enabled:
                try:
                    # 필터 구성
                    filter_params = {"source": {"$eq": filter_source}} if filter_source else None
                    
                    res = self.index.query(
                        vector=query_vector, 
                        top_k=top_k, 
                        include_metadata=True,
                        filter=filter_params
                    )
                    if res.matches:
                        return res.matches
                except Exception as e:
                    logger.error(f"Pinecone query error: {str(e)}")

            # 2. 로컬 캐시 검색
            if not self.local_cache:
                logger.warning("Empty knowledge base.")
                return []
            
            def cosine_similarity(v1, v2):
                return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            similarities = []
            # 필터 적용된 캐시 대상 선별
            target_cache = self.local_cache
            if filter_source:
                target_cache = [item for item in self.local_cache if item["metadata"].get("source") == filter_source]

            for item in target_cache:
                score = cosine_similarity(query_vector, item["values"])
                class Match:
                    def __init__(self, id, score, metadata):
                        self.id, self.score, self.metadata = id, score, metadata
                similarities.append(Match(item["id"], score, item["metadata"]))

            similarities.sort(key=lambda x: x.score, reverse=True)
            results = similarities[:top_k]
            
            if results:
                logger.info(f"Top Score: {results[0].score:.4f} for ID: {results[0].id}")
            return results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
