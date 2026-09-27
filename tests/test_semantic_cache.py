"""Public retrieval cache must not mix scopes or private documents."""

import unittest

from src.rag.semantic_cache import SemanticRetrievalCache


class _Redis:
    def __init__(self):
        self.values = {}
        self.lists = {}

    def get(self, key):
        return self.values.get(key)

    def lrange(self, key, start, stop):
        return self.lists.get(key, [])[start:] if stop == -1 else self.lists.get(key, [])[start:stop + 1]

    def pipeline(self, transaction=True):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def setex(self, key, _ttl, value):
        self.values[key] = value

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    def ltrim(self, key, start, stop):
        self.lists[key] = self.lrange(key, start, stop)

    def expire(self, *_args):
        pass

    def execute(self):
        pass


class SemanticCacheTests(unittest.TestCase):
    def test_exact_and_semantic_hits_respect_corpus_version(self):
        vectors = {"RRF retrieval": [1.0, 0.0], "RRF search": [0.999, 0.001], "unrelated": [0.0, 1.0]}
        cache = SemanticRetrievalCache(_Redis(), lambda query: vectors[query])
        config = {"filter_type": "paper", "corpus_count": 10, "rerank": True, "user_id": 1}
        expected = [{"id": "paper1"}]
        cache.put("RRF retrieval", config, expected)
        self.assertEqual(cache.get("RRF retrieval", config), expected)
        self.assertEqual(cache.get("RRF search", config), expected)
        self.assertIsNone(cache.get("unrelated", config))
        self.assertIsNone(cache.get("RRF retrieval", {**config, "corpus_count": 11}))
        self.assertIsNone(cache.get("RRF retrieval", {**config, "corpus_revision": "2"}))
        self.assertIsNone(cache.get("RRF retrieval", {**config, "user_id": 2}))

    def test_private_filter_is_never_eligible(self):
        self.assertFalse(SemanticRetrievalCache.eligible("user_doc"))
        self.assertFalse(SemanticRetrievalCache.eligible(None))
        self.assertTrue(SemanticRetrievalCache.eligible("paper"))


if __name__ == "__main__":
    unittest.main()
