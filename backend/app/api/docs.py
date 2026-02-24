from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.services.parser import DocumentParser
from app.services.rag_engine import get_rag_engine
from typing import List

router = APIRouter()
parser = DocumentParser()
# 공유 인스턴스 사용
engine = get_rag_engine()

@router.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks, 
    files: List[UploadFile] = File(...),
    user_id: str = "public"
):
    """다중 파일 업로드, 파싱 및 벡터 저장 실행 (v2.2 비동기 최적화)"""
    results = []
    
    for file in files:
        try:
            content = await file.read()
            # 1. 문서 파싱 (텍스트 + 이미지 추출)
            raw_docs = parser.parse_file(content, file.filename)
            
            # 사용자 ID 메타데이터 추가
            for doc in raw_docs:
                doc["metadata"]["user_id"] = user_id
            
            # 2. 텍스트 즉시 인덱싱 (매우 빠름)
            count = await engine.index_text(raw_docs)
            
            # 3. 이미지 백그라운드 인덱싱 등록 (무거운 작업)
            # 사용자는 기다리지 않고 즉시 응답을 받음
            background_tasks.add_task(engine.index_images_background, raw_docs)
            
            results.append({
                "filename": file.filename,
                "status": "Success",
                "message": "Text indexed. Image analysis is running in background.",
                "upserted_count": count
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "Error",
                "error": str(e)
            })
            
    return {"results": results}

@router.get("/list")
async def list_documents(user_id: str = "public"):
    """등록된 문서 목록 조회 (사용자별 필터링)"""
    # 벡터 DB에 저장된 실제 문서 소스 목록 추출
    docs_info = {}
    if engine.vector_db.local_cache:
        for item in engine.vector_db.local_cache:
            # 사용자 ID가 일치하거나 공용 문서인 경우만 반환
            if item["metadata"].get("user_id") != user_id:
                continue
                
            source = item["metadata"]["source"]
            current_status = item["metadata"].get("status", "indexed")
            
            if source not in docs_info:
                docs_info[source] = {
                    "filename": source,
                    "uploaded_at": item["metadata"].get("uploaded_at"),
                    "status": current_status,
                    "type": item["metadata"].get("type", "unknown")
                }
            else:
                # 'indexed' 상태가 하나라도 있으면 최종 상태로 간주 (이미지 분석 완료 대응)
                if current_status == "indexed":
                    docs_info[source]["status"] = "indexed"
    
    return {"documents": list(docs_info.values())}

@router.delete("/delete/{filename}")
async def delete_document(filename: str, user_id: str = "public"):
    """특정 문서 삭제 (본인 확인)"""
    try:
        success = engine.vector_db.delete_document(filename, user_id=user_id)
        if success:
            return {"status": "Success", "message": f"Document {filename} deleted."}
        else:
            return {"status": "Error", "message": "Failed to delete document."}
    except Exception as e:
        return {"status": "Error", "message": str(e)}

@router.get("/debug")
async def debug_status():
    """RAG 엔진 및 지식 베이스 상태 정밀 진단"""
    db = engine.vector_db
    return {
        "pinecone_enabled": db.enabled,
        "pinecone_index": db.index_name if db.enabled else None,
        "local_cache_size": len(db.local_cache),
        "loaded_documents": list(set(item["metadata"]["source"] for item in db.local_cache)),
        "openai_key_configured": bool(db.embeddings.openai_api_key)
    }
