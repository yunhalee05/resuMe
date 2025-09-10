import redis, json
from resume.agents.summarizer import Summarizer

class HistoryStore:
    def __init__(self, ttl=3600):
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.ttl = ttl
        self.summarizer = Summarizer()

    def save(self, session_id, q, a):
        """새로운 질답을 기록"""
        key = f"chat:{session_id}:history"
        history = self.get(session_id)
        history.append({"q": q, "a": a})
        self.redis.setex(key, self.ttl, json.dumps(history))
    
    def set(self, session_id, history):
        """history 전체를 갱신"""
        key = f"chat:{session_id}:history"
        self.redis.setex(key, self.ttl, json.dumps(history))

    def get(self, session_id):
        key = f"chat:{session_id}:history"
        data = self.redis.get(key)
        return json.loads(data) if data else []

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