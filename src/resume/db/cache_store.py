import json
from typing import Any, Union
import redis


class CacheStore:
    def __init__(self) -> None:
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
    
    def save(self, key: str, ttl: int, data: Any) -> None:
        self.redis.setex(key, ttl, json.dumps(data, ensure_ascii=False)) # 한글 깨지지 않게 설정 
    
    def get(self, key: str) -> Union[list[Any], dict[str, Any]]:
        data = self.redis.get(key)
        return json.loads(data) if data else [] 
    
    
