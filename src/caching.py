
"""
Caching utility for query results. Supports Redis or local file cache.
"""
import os
import json
import hashlib

try:
    import redis
except ImportError:
    redis = None


class Cache:
    def __init__(self, use_redis=False, redis_url="redis://localhost:6379/0", cache_file="./data/cache.json"):
        self.use_redis = use_redis and redis is not None
        self.cache_file = cache_file

        if self.use_redis:
            self.client = redis.Redis.from_url(redis_url)
        else:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            if not os.path.exists(cache_file):
                with open(cache_file, "w") as f:
                    json.dump({}, f)

    def _make_key(self, query: str, backend="faiss", llm_model="distilgpt2", k=5):
        key_string = f"{query}:{backend}:{llm_model}:{k}"
        return hashlib.md5(key_string.encode("utf-8")).hexdigest()

    def get(self, query: str):
        key = self._make_key(query)
        if self.use_redis:
            val = self.client.get(key)
            return json.loads(val) if val else None
        else:
            with open(self.cache_file, "r") as f:
                data = json.load(f)
            return data.get(key)

    def set(self, query: str, result: dict):
        key = self._make_key(query)
        if self.use_redis:
            self.client.set(key, json.dumps(result))
        else:
            with open(self.cache_file, "r") as f:
                data = json.load(f)
            data[key] = result
            with open(self.cache_file, "w") as f:
                json.dump(data, f, indent=2)
