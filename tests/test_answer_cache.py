import unittest

from src.rag.answer_cache import SemanticAnswerCache, cacheable_result, eligible_answer
from test_semantic_cache import _Redis


class AnswerCacheTests(unittest.TestCase):
    def test_personalization_and_corpus_revision_are_in_scope(self):
        redis = _Redis()
        cache = SemanticAnswerCache(redis, lambda query: [1.0, 0.0] if "RRF" in query else [0.0, 1.0])
        scope = cache.scope(1, "gpt-4o-mini", "prompt", 100)
        reply = {"reply": "RRF combines rankings", "citations": [{"url": "https://example.org/a"}]}
        cache.put("Explain RRF", scope, reply)
        self.assertEqual(cache.get("Explain RRF", scope), reply)
        self.assertEqual(cache.get("Describe RRF", scope), reply)
        self.assertIsNone(cache.get("Explain RRF", cache.scope(2, "gpt-4o-mini", "prompt", 100)))
        redis.values["rag:public_corpus_revision"] = "2"
        self.assertIsNone(cache.get("Explain RRF", cache.scope(1, "gpt-4o-mini", "prompt", 100)))

    def test_only_cold_cited_kb_answers_are_cached(self):
        self.assertTrue(eligible_answer("research_qa", [], "", "ReflectionAgent"))
        self.assertFalse(eligible_answer("research_qa", [{"role": "user"}], "", "ReflectionAgent"))
        self.assertFalse(eligible_answer("research_qa", [], "private fact", "ReflectionAgent"))
        self.assertFalse(cacheable_result("answer", ["search_web"], [{"url": "x"}]))
        self.assertTrue(cacheable_result("answer", ["search_knowledge_base"], [{"url": "x"}]))


if __name__ == "__main__":
    unittest.main()
