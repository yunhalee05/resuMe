import json
import redis


class CacheStore:
    def __init__(self) -> None:
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
    
    def save(self, key: str, ttl: int, data: str):
        self.redis.setex(key, ttl, json.dumps(data))
    
    def get(self, key:str):
        data = self.redis.get(key)
        return json.loads(data) if data else [] 
    
    
