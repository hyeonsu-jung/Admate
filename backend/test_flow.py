import sys
import os

# 환경 변수 Mocking (Settings 로드 전 실행)
os.environ["OPENAI_API_KEY"] = "mock-key"
os.environ["ANTHROPIC_API_KEY"] = "mock-key"
os.environ["PINECONE_API_KEY"] = "mock-key"

# 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.parser import DocumentParser
from app.services.rag_engine import RagEngine

def test_parsing_flow():
    print("=== Testing Document Parsing Flow ===")
    parser = DocumentParser()
    engine = RagEngine()

    # Mock Data: Simple Text for testing (Simulation of parsed PDF)
    mock_raw_docs = [
        {
            "content": "이것은 광고 단가표 테스트 문서입니다. 네이버 브랜드검색 단가는 1,000만원입니다. " * 30,
            "metadata": {"source": "test.pdf", "page": 1}
        }
    ]

    print(f"Original Text Length: {len(mock_raw_docs[0]['content'])}")
    
    # Testing Chunking
    chunks = engine.split_documents(mock_raw_docs)
    print(f"Total Chunks Created: {len(chunks)}")
    
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n[Chunk {i}]")
        print(f"Length: {len(chunk['content'])}")
        print(f"Preview: {chunk['content'][:100]}...")
        print(f"Metadata: {chunk['metadata']}")

    print("\n=== Test Completed Successfully ===")

if __name__ == "__main__":
    test_parsing_flow()
