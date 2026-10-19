"""Shared BM25 index: paged reads, scoring parity, scoping, sharing and backoff."""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.rag import retriever as retriever_module
from src.rag.retriever import Retriever, _build_index, _tokenize


class _Collection:
    """Minimal ChromaDB stand-in that fails like SQLite on oversized reads."""

    def __init__(self, records, max_variables=5, fail=False):
        self.id = f"fake-{id(self)}"
        self.records = records  # list of (id, document, metadata)
        self.max_variables = max_variables
        self.fail = fail
        self.page_reads = 0
        self.id_reads = 0

    @staticmethod
    def _match(metadata, where):
        if "$and" in where:
            return all(_Collection._match(metadata, clause) for clause in where["$and"])
        key, condition = next(iter(where.items()))
        if isinstance(condition, dict) and "$in" in condition:
            return metadata.get(key) in condition["$in"]
        return metadata.get(key) == condition

    def get(self, ids=None, where=None, limit=None, offset=None, include=None):
        if self.fail:
            raise RuntimeError("database unavailable")
        if ids is not None:
            self.id_reads += 1
            rows = [r for r in self.records if r[0] in set(ids)]
        else:
            self.page_reads += 1
            rows = [r for r in self.records if where is None or self._match(r[2], where)]
            if limit is None or limit > self.max_variables:
                raise RuntimeError("too many SQL variables")
            rows = rows[offset or 0:(offset or 0) + limit]
        return {"ids": [r[0] for r in rows], "documents": [r[1] for r in rows],
                "metadatas": [r[2] for r in rows]}


def _paper(i, text):
    return (f"paper_{i}", text, {"type": "paper", "paper_id": f"2601.{i:05d}", "title": f"Paper {i}"})


CORPUS = [
    _paper(0, "Reciprocal rank fusion combines lexical and dense rankings."),
    _paper(1, "LoRA adapts large language models with low-rank updates."),
    _paper(2, "Dense retrieval with contrastive training for open-domain QA."),
    _paper(3, "Arxiv 2401.09876, a benchmark for retrieval augmented generation."),
    _paper(4, "Diffusion models for image generation and editing."),
    _paper(5, "Reinforcement learning from human feedback aligns language models."),
    _paper(6, "Low-rank adaptation compared with full fine-tuning of language models."),
    ("paperfull_2601.00001_0", "rank fusion rank fusion rank fusion full text",
     {"type": "paper_fulltext", "arxiv_id": "2601.00001"}),
    ("userdoc_userdoc_7_ab_0", "My private notes on rank fusion experiments.",
     {"type": "user_doc", "user_id": 7, "title": "notes"}),
]


def _retriever(collection):
    return Retriever(embedding_manager=SimpleNamespace(collection=collection))


class BM25IndexTests(unittest.TestCase):
    def setUp(self):
        retriever_module._BM25_CACHE.reset()
        self.page_patch = patch.object(retriever_module, "BM25_PAGE_SIZE", 3)
        self.page_patch.start()

    def tearDown(self):
        self.page_patch.stop()
        retriever_module._BM25_CACHE.reset()

    def test_scores_match_rank_bm25_okapi(self):
        from rank_bm25 import BM25Okapi
        texts = [doc for _, doc, _ in CORPUS]
        reference = BM25Okapi([_tokenize(t) for t in texts])
        index = _build_index(CORPUS)
        for query in ("rank fusion", "language models low-rank", "2401.09876", "unseen words"):
            np.testing.assert_allclose(index.score(query), reference.get_scores(_tokenize(query)),
                                       rtol=1e-5, atol=1e-6)

    def test_corpus_larger_than_one_read_is_loaded_in_pages(self):
        collection = _Collection(CORPUS, max_variables=3)
        with self.assertRaisesRegex(RuntimeError, "too many SQL variables"):
            collection.get(include=["documents", "metadatas"])  # the old unpaginated read
        results = _retriever(collection)._bm25_search("low-rank language models", 10, "paper")
        self.assertEqual({r["id"] for r in results} >= {"paper_1", "paper_6"}, True)
        self.assertEqual(len(retriever_module._BM25_CACHE.get(collection)), 7)  # papers only
        self.assertGreaterEqual(collection.page_reads, 3)

    def test_fulltext_and_private_chunks_stay_out_of_shared_index(self):
        collection = _Collection(CORPUS)
        index = retriever_module._BM25_CACHE.get(collection)
        self.assertNotIn("paperfull_2601.00001_0", index.doc_ids)
        self.assertNotIn("userdoc_userdoc_7_ab_0", index.doc_ids)

        anonymous = _retriever(collection)._bm25_search("rank fusion", 10, None, user_id=None)
        stranger = _retriever(collection)._bm25_search("rank fusion", 10, None, user_id=8)
        owner = _retriever(collection)._bm25_search("rank fusion", 10, None, user_id=7)
        private_only = _retriever(collection)._bm25_search("rank fusion", 10, "user_doc", user_id=7)
        for results in (anonymous, stranger):
            self.assertNotIn("userdoc_userdoc_7_ab_0", [r["id"] for r in results])
        self.assertIn("userdoc_userdoc_7_ab_0", [r["id"] for r in owner])
        self.assertEqual([r["id"] for r in private_only], ["userdoc_userdoc_7_ab_0"])
        self.assertEqual(_retriever(collection)._bm25_search("rank fusion", 10, "user_doc", user_id=None), [])

    def test_results_keep_the_shape_fusion_and_rerank_expect(self):
        collection = _Collection(CORPUS)
        results = _retriever(collection)._bm25_search("2401.09876", 5, "paper")
        self.assertEqual(results[0]["id"], "paper_3")
        self.assertEqual(results[0]["metadata"]["paper_id"], "2601.00003")
        self.assertIn("benchmark", results[0]["document"])
        self.assertGreater(results[0]["bm25_score"], 0)

    def test_index_is_built_once_and_shared_by_every_retriever(self):
        collection = _Collection(CORPUS)
        _retriever(collection)._bm25_search("rank fusion", 5, "paper")
        reads_after_first = collection.page_reads
        for _ in range(5):
            _retriever(collection)._bm25_search("language models", 5, "paper")
        self.assertEqual(collection.page_reads, reads_after_first)

    def test_failed_build_backs_off_instead_of_retrying_every_request(self):
        collection = _Collection(CORPUS, fail=True)
        self.assertEqual(_retriever(collection)._bm25_search("rank fusion", 5, "paper"), [])
        self.assertEqual(collection.page_reads, 0)  # failed before counting
        attempts = []
        original_get = collection.get
        collection.get = lambda **kwargs: attempts.append(1) or original_get(**kwargs)
        for _ in range(3):
            self.assertEqual(_retriever(collection)._bm25_search("rank fusion", 5, "paper"), [])
        self.assertEqual(attempts, [])
        self.assertEqual(retriever_module.BM25_INDEX_AVAILABLE._value.get(), 0)

        collection.fail = False
        later = time.time() + retriever_module.BM25_RETRY_BACKOFF_SECONDS + 1
        with patch.object(retriever_module.time, "time", return_value=later):
            results = _retriever(collection)._bm25_search("rank fusion", 5, "paper")
        self.assertEqual(results[0]["id"], "paper_0")
        self.assertEqual(retriever_module.BM25_INDEX_AVAILABLE._value.get(), 1)
        self.assertEqual(retriever_module.BM25_INDEX_DOCUMENTS._value.get(), 7)

    def test_expired_index_keeps_serving_while_refreshing(self):
        collection = _Collection(CORPUS)
        first = retriever_module._BM25_CACHE.get(collection)
        expired = time.time() + retriever_module.BM25_INDEX_TTL_SECONDS + 1
        with patch.object(retriever_module.time, "time", return_value=expired):
            served = retriever_module._BM25_CACHE.get(collection)
        self.assertIs(served, first)
        for thread in [t for t in __import__("threading").enumerate() if t.name == "bm25-refresh"]:
            thread.join(timeout=5)
        self.assertIsNot(retriever_module._BM25_CACHE.get(collection), first)

    def test_ids_deleted_after_build_are_skipped(self):
        collection = _Collection(CORPUS)
        retriever_module._BM25_CACHE.get(collection)
        collection.records = [r for r in collection.records if r[0] != "paper_0"]
        results = _retriever(collection)._bm25_search("rank fusion", 5, "paper")
        self.assertNotIn("paper_0", [r["id"] for r in results])


if __name__ == "__main__":
    unittest.main()
