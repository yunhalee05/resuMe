from openai import AsyncOpenAI
from resume.agents.classifier import Classifier
from resume.agents.persona import Persona
from resume.agents.refiner import Refiner
from resume.agents.retriever import Retriever
from resume.db.cache_store import CacheStore
from resume.db.vector_store import VectorStore
from resume.repository.answer_repository import AnswerRepository
from resume.repository.history_repository import HistoryRepository
from dotenv import load_dotenv


class ResumeChatbot:
    def __init__(self, gcs_bucket: str, gcs_projects_path: str, gcs_qna_path: str, gcs_introduce_path: str, use_gcs = True, cache_file: str = "answer_cache.json"):
        load_dotenv(override=True)
        self.client = AsyncOpenAI()
        db = VectorStore(
            gcs_bucket, 
            gcs_projects_path,
            gcs_qna_path,
            gcs_introduce_path,
            use_gcs,
            cache_file,
        )
        cache = CacheStore()
        self.answer_repository = AnswerRepository(cache, self.client)
        self.history_repository = HistoryRepository(cache, self.client)
        self.classifier = Classifier(self.client)
        self.retriever = Retriever(self.client, db)
        self.persona = Persona(self.client, self.history_repository)
        self.refiner = Refiner(self.client)

    async def chat(self, message: str, history: list, session_id: str):
        # 캐시에 있다면 답변 
        cached = self.answer_repository.get_answer(message)
        if cached:
            return cached
        
        if not await self.retriever.is_context_valid(message):
            return "제 이력서나 요약에는 해당 정보가 포함되어 있지 않아서 답변드리기 어려워요."

        # 1) 질문 분류
        category_info = await self.classifier.classify_question(message)
        category = category_info["category"]

        # 2) 관련 컨텍스트 검색
        context = await self.retriever.retrieve_context(message, category_info)
        if not context.strip():
            final_answer = "제 이력서나 요약에는 해당 정보가 포함되어 있지 않아서 답변드리기 어려워요."
            return final_answer

        # 3) Persona 답변 생성
        # recent_history = self.history_store.get_summary(session_id, self.summarize_history)
        draft_answer = await self.persona.persona_answer(message, category, context, session_id)
        if "제 이력서에는 해당 정보가 없습니다." in draft_answer:
            return "제 이력서나 요약에는 해당 정보가 포함되어 있지 않아서 답변드리기 어려워요."

        # 4) 스타일 보정
        final_answer = await self.refiner.refine_answer(draft_answer)

        # 5) 대화 기록 저장
        self.history_repository.save(session_id, message, final_answer)
        self.answer_repository.save(message, final_answer, category)

        return final_answer


