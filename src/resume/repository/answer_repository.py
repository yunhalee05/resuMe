import datetime
import hashlib
from resume.agents.summarizer import Summarizer
from resume.db.cache_store import CacheStore

class AnswerRepository:
    def __init__(self, redis: CacheStore, ttl=3600):
        self.redis = redis
        self.ttl = ttl
        self.summarizer = Summarizer()

    def save(self, question: str, answer: str, category: str):
        """새로운 응답을 기록"""
        key = self._get_question_key(question)
        entry = {
            "category": category,
            "question": question, 
            "answer": answer,
            "category": category,
            "timestamp": datetime.now().isoformat()
        }
        histories = self._get(category)
        histories.append(entry)
        self.redis.save(key, self.ttl, histories)

    def get_answer(self, question: str):
        value = self._get(question)
        return value if value else None
    
    def _get_question_key(self, question: str) -> str:
        return hashlib.sha256(question.encode()).hexdigest()
    
    def _get(self, question: str):
        key = self._get_question_key(question)
        return self.redis.get(key)