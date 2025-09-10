# Agent 4 스타일 보정 Agent
from openai import AsyncOpenAI


class Refiner:
    def __init__(self, client: AsyncOpenAI) -> None:
        self.client = client 
    
    async def refine_answer(self, answer: str) -> str:
        """답변을 면접 톤으로 최종 다듬기 (길면 줄이고, 핵심 강조)"""
        prompt = f"""
        아래는 면접 답변 초안입니다:
        {answer}

        이 답변을 다음 기준으로 다듬어주세요:
        - 반드시 한국어로 대답한다.
        - 실제 면접 대화처럼 자연스럽고 자신감 있는 어투로 바꾼다.
        - 답변이 너무 길면 핵심만 담아 5문장 이내로 줄인다.
        - 성과와 핵심 경험을 명확히 강조한다.
        - 글을 읽는 듯한 어투 대신, 구어체 면접 답변처럼 자연스럽게 표현한다.
        """
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content
