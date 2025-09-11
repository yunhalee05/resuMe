




import datetime
from openai import AsyncOpenAI

from resume.db.vector_store import VectorStore

# Agent 2: 지식 검색기 (RAG Agent)
class Retriever:
    def __init__(self, client: AsyncOpenAI, store: VectorStore) -> None:
        self.client = client 
        self.store = store 

    async def retrieve_context(self, question: str, category_info: dict) -> str:
        """질문과 가장 유사한 카테고리 정보에 맞는 메타데이터 기반 Resume/summary 부분을 검색"""
        time_condition = category_info.get("time_condition", "none")
        filters = category_info.get("filters", {})  
        category = category_info.get("category")
        k = 5

        try:
            if(time_condition != "none" and category == "프로젝트 경험"):
                results = self.store.get_similar_data(
                    question,
                    20,
                    filters if filters else None
                )
                if time_condition == "recent":
                    results = sorted(results, key=lambda r: self._parse_date_safe(r.metadata.get("period_from")), reverse=True)[:k]
                elif time_condition == "first":
                    results = sorted(results, key=lambda r: self._parse_date_safe(r.metadata.get("period_from")), reverse=False)[:k]
            else :
                results = self.store.get_similar_data(
                    question,
                    k,
                    filters if filters else None
                )
        except Exception:
            results = self.store.get_similar_data(question, k=3)

        # for r in results:
        #     print(r.page_content)
        #     print("META:", r.metadata)
        #     print("-" * 50)

        if not results:
            return ""

        return "\n".join([r.page_content for r in results])

    def _parse_date_safe(val):
        try:
            return datetime.fromisoformat(val)
        except Exception:
            return datetime.min

    async def is_context_valid(self, question: str, threshold: float = 0.2) -> bool:
        score= self.store.get_context_score(question, threshold)
        return score >= threshold