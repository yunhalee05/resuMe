

import json

from openai import AsyncOpenAI

# Agent 1: 질문 분류기
class Classifier:
    def __init__(self, client: AsyncOpenAI) -> None:
        self.client = client 

    async def classify_question(self, question: str) -> str:
        """질문을 카테고리 + 검색 메타데이터 필드로 분류"""
        categories = ["프로젝트 경험", "기술스택", "협업", "자기소개", "학습 경험"]
        prompt = f"""
        분류할 질문: "{question}"
        이력서 데이터 메타데이터 필드:
        - company
        - role
        - period, period_from, period_to
        - tech_stack
        - topic_tags (고정값: {{"문제 해결", "프로젝트 경험", "학습 경험", "기술 스택", "지원 동기", "협업"}})
        
        카테고리 → 메타데이터 매핑:
        - 프로젝트 경험 → [company, role, period, period_from, period_to, tech_stack]
        - 기술 스택 → [tech_stack]
        - 학습 경험 → [topic_tags]
        - 협업 → [topic_tags]
        - 지원 동기 → [topic_tags]
        - 자기소개 → [summary]

        카테고리 → doc_type 매핑:
        - 프로젝트 경험, 기술 스택 → "projects"
        - 학습 경험, 협업, 지원 동기 → "qna"
        - 자기소개 → "summary"

        추가 규칙:
        - 항상 filter에 doc_type 포함 (예: {{"doc_type": "projects"}})
        - 질문에 특정 값이 있으면 해당 메타데이터 필드에 regex 조건 추가 가능
        - 프로젝트 경험 질문에 "최근/마지막" → time_condition="recent"
        - 프로젝트 경험 질문에 "처음/첫번째" → time_condition="first"
        - 그 외 → time_condition="none"
        - summary 카테고리의 경우 filter에는 doc_type만 포함
        
        출력 형식(JSON):
        {{
            "category": "<카테고리>",
            "time_condition": "<recent|first|none>",
            "filters": {{
                "doc_type": "<projects|qna|summary>",
                "<필드명>": {{"$regex": ".*<값>.*"}}  
            }}
        }}
        """

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"} 
        )

        return json.loads(response.choices[0].message.content.strip())