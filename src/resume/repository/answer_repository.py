import datetime
import hashlib
from typing import Any, Optional

from openai import AsyncOpenAI
from resume.agents.summarizer import Summarizer
from resume.db.cache_store import CacheStore

class AnswerRepository:
    def __init__(self, redis: CacheStore, client: AsyncOpenAI, ttl=3600):
        self.redis = redis
        self.ttl = ttl
        self.summarizer = Summarizer(client)

    def save(self, question: str, answer: str, category: str) -> None:
        """새로운 응답을 기록"""
        key = self._get_question_key(question)
        entry = {
            "category": category,
            "question": question, 
            "answer": answer,
            "timestamp": datetime.datetime.now().isoformat()
        }
        histories = self._get(category)
        histories.append(entry)
        self.redis.save(key, self.ttl, histories)

    def get_answer(self, question: str) -> Optional[Any]:
        value = self._get(question)
        return value if value else None
    
    def _get_question_key(self, question: str) -> str:
        return hashlib.sha256(question.encode()).hexdigest()
    
    def _get(self, question: str) -> list[dict[str, Any]]:
        key = self._get_question_key(question)
        return self.redis.get(key)