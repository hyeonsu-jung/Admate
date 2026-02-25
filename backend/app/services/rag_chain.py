from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
from typing import List, Dict, Any

class RagChain:
    """GPT-4o를 사용한 RAG 답변 생성 체인 (Claude 404 에러 대응)"""
    
    def __init__(self):
        # Anthropic 대신 제공해주신 OpenAI 키를 사용하여 GPT-4o로 전환
        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0
        )
        
        self.prompt = ChatPromptTemplate.from_template("""
당신은 광고 운영팀을 돕는 유능한 AI 어시스턴트입니다. 
아래 제공된 [문서 내용]을 바탕으로 사용자의 질문에 친절하고 정확하게 답변하세요.
제공된 내용에는 문서의 텍스트뿐만 아니라, 문서 내 이미지를 AI가 분석한 내용([이미지 분석 내용])도 포함되어 있을 수 있습니다. 모든 정보를 조화롭게 참고하여 답변하세요.

[문서 내용]
{context}

[질문]
{question}

[답변 지침]
1. 반드시 제공된 [문서 내용]에 근거하여 답변하세요.
2. **절대로 마크다운(Markdown) 기호를 사용하지 마세요.**
   - 텍스트 앞뒤에 **볼드(`**`)**나 **글머리 기호(`-`)**를 넣지 마세요.
   - 오직 **명확한 줄바꿈(Enter)**과 **빈 줄(Empty Line)**만을 사용하여 정보를 구분하세요.
3. 문서에 정보가 없는 경우, 억지로 답변하지 말고 "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 정중히 안내하세요.
4. 숫자가 포함된 경우(단가, 수수료 등) 정확하게 표기하세요.
5. 답변 끝에 반드시 활성화된 출처(문서명, 페이지 등)를 언급하세요.
""")
        self.chain = self.prompt | self.llm | StrOutputParser()

    async def generate_answer(self, question: str, context: str) -> str:
        """컨텍스트와 질문을 결합하여 답변 생성 (동기식 호출용)"""
        response = await self.chain.ainvoke({
            "context": context,
            "question": question
        })
        return response

    async def astream_answer(self, question: str, context: str):
        """토큰 단위 스트리밍 답변 출력"""
        async for chunk in self.chain.astream({
            "context": context,
            "question": question
        }):
            yield chunk
