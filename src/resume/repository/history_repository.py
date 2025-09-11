from resume.agents.summarizer import Summarizer
from resume.db.cache_store import CacheStore

class HistoryRepository:
    def __init__(self, redis: CacheStore, ttl=3600):
        self.redis = redis
        self.ttl = ttl
        self.summarizer = Summarizer()

    def save(self, session_id, q, a):
        """새로운 질답을 기록"""
        key = f"chat:{session_id}:history"
        history = self.get(session_id)
        history.append({"q": q, "a": a})
        self.redis.save(key, self.ttl, history)
    
    def set(self, session_id, history):
        """history 전체를 갱신"""
        key = f"chat:{session_id}:history"
        self.redis.save(key, self.ttl, history)

    def get(self, session_id):
        key = f"chat:{session_id}:history"
        return self.redis.get(key)
        

    def get_window(self, session_id, n=5):
        history = self.get(session_id)
        return history[-n:]

    async def get_summary(self, session_id, threshold=10):
        history = self.get(session_id)
        if len(history) > threshold:
            summary = await self.summarizer(history[:-5])  
            new_history = [{"summary": summary}] + history[-5:]  
            self.set(session_id, new_history) 
            return new_history
        return history

