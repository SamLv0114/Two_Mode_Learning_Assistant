"""Short-lived, public-corpus-only retrieval cache backed by Redis."""
import hashlib
import json
import logging
import math
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

TTL_SECONDS = 900
MAX_SCOPE_ENTRIES = 64
# A twelve-pair adversarial probe found a negation false reuse at 0.98.
# 0.997 admitted only exact duplicates in that small set. Keep the feature
# disabled by default until a larger labelled and production-traffic study.
SIMILARITY_THRESHOLD = 0.997


def cosine(a, b) -> float:
    if len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else -1.0


class SemanticRetrievalCache:
    """Caches exact and near-duplicate queries for explicitly public filters.

    Uses a small bounded Redis list rather than Redis Stack vector indexes, so
    it works with the project's redis:7-alpine image. Entries are versioned by
    corpus count and retrieval configuration, and expire after 15 minutes.
    Private user documents are never eligible.
    """

    def __init__(self, redis_client, embed: Callable[[str], List[float]],
                 similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.redis = redis_client
        self.embed = embed
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def eligible(filter_type: Optional[str]) -> bool:
        return filter_type in ("paper", "article")

    @staticmethod
    def scope_key(config: Dict) -> str:
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        return f"retrieval_cache:{digest}"

    def _vector(self, query: str) -> List[float]:
        value = self.embed(query)
        return value.tolist() if hasattr(value, "tolist") else list(value)

    def get(self, query: str, config: Dict):
        scope = self.scope_key(config)
        exact = hashlib.sha256(query.strip().lower().encode()).hexdigest()
        try:
            raw = self.redis.get(f"{scope}:exact:{exact}")
            if raw is not None:
                return json.loads(raw)
            vector = self._vector(query)
            for row in self.redis.lrange(f"{scope}:vectors", 0, -1):
                entry = json.loads(row)
                if cosine(vector, entry["embedding"]) >= self.similarity_threshold:
                    result = self.redis.get(entry["result_key"])
                    if result is not None:
                        return json.loads(result)
        except Exception as exc:
            logger.debug("Retrieval cache read failed: %s", exc)
        return None

    def put(self, query: str, config: Dict, results: List[Dict]) -> None:
        scope = self.scope_key(config)
        exact = hashlib.sha256(query.strip().lower().encode()).hexdigest()
        result_key = f"{scope}:exact:{exact}"
        try:
            vector = self._vector(query)
            with self.redis.pipeline(transaction=True) as pipe:
                pipe.setex(result_key, TTL_SECONDS, json.dumps(results))
                pipe.rpush(f"{scope}:vectors", json.dumps({"embedding": vector, "result_key": result_key}))
                pipe.ltrim(f"{scope}:vectors", -MAX_SCOPE_ENTRIES, -1)
                pipe.expire(f"{scope}:vectors", TTL_SECONDS)
                pipe.execute()
        except Exception as exc:
            logger.debug("Retrieval cache write failed: %s", exc)
