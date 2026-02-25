import base64
import logging
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.core.config import settings

logger = logging.getLogger(__name__)

class VisionService:
    """GPT-4o Vision을 사용하여 이미지의 내용을 분석하고 설명하는 서비스"""
    
    def __init__(self):
        # SSL 우회를 위한 커스텀 클라이언트 적용 (VectorDBService와 동일 전략)
        custom_client = httpx.AsyncClient(verify=False, timeout=60.0)
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=500,
            http_async_client=custom_client
        )

    @staticmethod
    def _sniff_mime(image_bytes: bytes) -> str:
        """바이트 헤더 기반으로 data URL mime을 추정 (OpenAI Vision 디코딩 안정화)"""
        if not image_bytes:
            return "image/jpeg"
        # JPEG
        if image_bytes[:2] == b"\xff\xd8":
            return "image/jpeg"
        # PNG
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        # GIF
        if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        # WEBP: RIFF....WEBP
        if len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"

    async def describe_image(self, image_bytes: bytes) -> str:
        """이미지 바이트 데이터를 받아 GPT-4o Vision으로 분석된 텍스트 반환"""
        try:
            # 1. 이미지를 Base64로 인코딩
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            mime = self._sniff_mime(image_bytes)
            
            # 2. 멀티모달 메시지 구성
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "이 이미지는 광고 운영 가이드 문서의 일부입니다. 이 이미지에 포함된 데이터, 도표의 수치, 또는 시각적 설명을 상세히 텍스트로 요약해줘. RAG 검색에 활용될 정보이므로 중요한 키워드와 숫자 위주로 작성해줘."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{base64_image}"},
                    },
                ]
            )
            
            # 3. LLM 호출
            response = await self.llm.ainvoke([message])
            description = response.content
            
            logger.info("Image analysis completed successfully.")
            return description
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {str(e)}")
            return f"[이미지 분석 실패: {str(e)}]"

# 싱글톤 인스턴스 제공
_vision_service = None

def get_vision_service():
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service
