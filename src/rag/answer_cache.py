"""Conservative, opt-in answer-cache experiment for cold research questions."""
import hashlib
from typing import Dict, List

from src.rag.semantic_cache import SemanticRetrievalCache


ALLOWED_TOOLS = {"search_knowledge_base"}


def eligible_answer(intent: str, history: List[Dict], facts: str, agent_name: str) -> bool:
    return intent == "research_qa" and agent_name == "ReflectionAgent" and not history and not facts


def cacheable_result(reply: str, tools_called: List[str], citations: List[Dict]) -> bool:
    return bool(reply and citations and tools_called and set(tools_called) <= ALLOWED_TOOLS)


class SemanticAnswerCache:
    def __init__(self, redis_client, embed):
        self.inner = SemanticRetrievalCache(redis_client, embed, similarity_threshold=0.997)
        self.redis = redis_client

    def scope(self, user_id: int, model: str, agent_prompt: str, corpus_count: int) -> Dict:
        return {
            "user_id": user_id, "model": model,
            "agent_prompt_sha256": hashlib.sha256(agent_prompt.encode()).hexdigest(),
            "corpus_count": corpus_count,
            "corpus_revision": self.redis.get("rag:public_corpus_revision") or "0",
            "answer_cache_version": 1,
        }

    def get(self, query: str, scope: Dict):
        return self.inner.get(query, scope)

    def put(self, query: str, scope: Dict, result: Dict):
        self.inner.put(query, scope, result)
