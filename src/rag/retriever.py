"""
Hybrid (dense + lexical) retriever for RAG.
"""
from typing import List, Dict, Optional
import logging
import string
import threading
import time

from src.models.embeddings import EmbeddingManager
from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# How long a lazily-built BM25 index is trusted before being rebuilt from
# ChromaDB's current contents. Bounds staleness against papers the nightly
# indexer adds while this process keeps running, without wiring an explicit
# refresh call into that job — matches the cache-TTL idiom already used
# elsewhere in this project (e.g. What's Hot's 6h Redis cache) rather than
# inventing a new invalidation scheme.
BM25_INDEX_TTL_SECONDS = 6 * 3600

# How many candidates the fused (vector + BM25) shortlist carries into the
# Cross-Encoder rerank stage. Bounds rerank cost (it scores every candidate
# against the query individually, unlike vector/BM25 which score against a
# precomputed index) regardless of how wide the two upstream searches are.
RERANK_CANDIDATE_POOL = 30


class Retriever:
    """
    Retrieves relevant documents via hybrid search: dense vector similarity
    (ChromaDB / sentence-transformers) fused with sparse lexical matching
    (BM25) using Reciprocal Rank Fusion, optionally refined by a
    Cross-Encoder reranker.

    Pure vector search misses exact-term hits embeddings don't represent
    well — arXiv IDs, acronyms, exact paper titles — which BM25 catches
    trivially. The two ranked lists are combined with RRF rather than a
    hand-tuned score blend: BM25 scores are unbounded and cosine
    similarities are [0,1], so averaging them directly needs a weight with
    no principled value — the same failure mode that made the recommendation
    pipeline's first calibration draft fragile (see daily_feed.py's
    _calibrate_top_k docstring). RRF sidesteps that by only using rank
    position, not raw score, so nothing needs tuning.

    Cross-Encoder reranking (settings.RAG_RERANK_ENABLED, off by default) is
    a further precision pass over the fused shortlist using real
    query-document attention instead of precomputed embeddings. It is gated
    off rather than always-on: in local dev testing it crashed the process
    with a native SIGBUS on the simplest possible input (device="cpu"
    explicitly set) — a failure mode try/except cannot catch, since no
    Python exception is ever raised. Whether that's specific to this dev
    machine (macOS ARM64, torch 2.11.0) or would also hit the Linux
    production container (python:3.11-slim) was never confirmed — verify it
    doesn't crash the actual deployment target before enabling it there.

    The BM25 index and the reranker both degrade to vector-only automatically
    if their dependency is missing or fails to *load* (see
    _ensure_bm25_index and _get_reranker) — that degradation path is for
    ImportError/load failures, not for a reranker that loads fine and then
    crashes the process on predict(), which is exactly why reranking needs
    the separate opt-in flag rather than relying on the same fallback.
    """

    _bm25_lock = threading.Lock()
    _reranker_instance = None
    _reranker_lock = threading.Lock()

    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self._bm25 = None  # None = not built yet, False = tried, unavailable
        self._bm25_doc_ids: List[str] = []
        self._bm25_docs: List[str] = []
        self._bm25_metas: List[Dict] = []
        self._bm25_built_at: float = 0.0

    # ── Lexical (BM25) index ────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Strip surrounding punctuation per token (trailing commas/periods
        # from sentence structure) without touching punctuation *inside* a
        # token — a bare .split() left "2401.09876," never matching a query
        # for "2401.09876", which is exactly the exact-ID-match case BM25 is
        # here for. Internal punctuation (arXiv IDs, hyphenated terms) stays
        # intact since strip() only trims the ends.
        return [
            stripped
            for raw in (text or "").lower().split()
            if (stripped := raw.strip(string.punctuation))
        ]

    def _ensure_bm25_index(self) -> None:
        """(Re)build the in-memory BM25 index from ChromaDB's current corpus
        if it's never been built, or has gone stale."""
        if self._bm25 is not None and (time.time() - self._bm25_built_at) < BM25_INDEX_TTL_SECONDS:
            return

        with self._bm25_lock:
            if self._bm25 is not None and (time.time() - self._bm25_built_at) < BM25_INDEX_TTL_SECONDS:
                return  # another thread rebuilt it while we waited for the lock

            try:
                from rank_bm25 import BM25Okapi
            except ImportError:
                logger.warning("rank_bm25 not installed — hybrid search falls back to vector-only")
                self._bm25 = False
                self._bm25_built_at = time.time()
                return

            try:
                corpus = self.embedding_manager.collection.get(include=["documents", "metadatas"])
            except Exception as e:
                logger.warning(f"BM25 index build failed to read ChromaDB corpus: {e}")
                self._bm25 = False
                self._bm25_built_at = time.time()
                return

            ids = corpus.get("ids") or []
            documents = corpus.get("documents") or []
            metadatas = corpus.get("metadatas") or []
            if not ids:
                self._bm25 = False
                self._bm25_built_at = time.time()
                return

            self._bm25 = BM25Okapi([self._tokenize(doc) for doc in documents])
            self._bm25_doc_ids = ids
            self._bm25_docs = documents
            self._bm25_metas = metadatas
            self._bm25_built_at = time.time()
            logger.info(f"BM25 index built: {len(ids)} documents")

    def _bm25_search(self, query: str, n_results: int, filter_type: Optional[str]) -> List[Dict]:
        self._ensure_bm25_index()
        if not self._bm25:
            return []

        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[Dict] = []
        for i in ranked:
            if scores[i] <= 0:
                break  # BM25Okapi scores sort descending; 0 means no term overlap at all
            meta = self._bm25_metas[i] or {}
            if filter_type and meta.get("type") != filter_type:
                continue
            results.append({
                "id": self._bm25_doc_ids[i],
                "document": self._bm25_docs[i],
                "metadata": meta,
                "bm25_score": float(scores[i]),
            })
            if len(results) >= n_results:
                break
        return results

    # ── Fusion ──────────────────────────────────────────────────────────

    @staticmethod
    def _reciprocal_rank_fusion(ranked_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
        """
        Merge multiple already-ranked result lists into one by reciprocal
        rank rather than raw score, so BM25's unbounded scores and cosine
        similarity's [0,1] range never need a cross-method weight.

        Attaches rrf_score and a "distance" (lower = better) field to each
        returned item so the result has the same shape whether or not
        RAG_RERANK_ENABLED goes on to replace it with a rerank_score.
        """
        fused: Dict[str, float] = {}
        by_id: Dict[str, Dict] = {}
        for results in ranked_lists:
            for rank, item in enumerate(results):
                doc_id = item.get("id")
                if not doc_id:
                    continue
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
                by_id.setdefault(doc_id, item)

        ordered_ids = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)
        merged = []
        for doc_id in ordered_ids:
            item = dict(by_id[doc_id])
            item["rrf_score"] = fused[doc_id]
            item["distance"] = -fused[doc_id]
            merged.append(item)
        return merged

    # ── Cross-Encoder rerank ────────────────────────────────────────────

    @classmethod
    def _get_reranker(cls):
        if cls._reranker_instance is not None:
            return cls._reranker_instance
        with cls._reranker_lock:
            if cls._reranker_instance is not None:
                return cls._reranker_instance
            try:
                from sentence_transformers import CrossEncoder
                cls._reranker_instance = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
                )
                logger.info("Cross-encoder reranker loaded: ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                logger.warning(f"Cross-encoder reranker unavailable, skipping rerank: {e}")
                cls._reranker_instance = False
            return cls._reranker_instance

    def _rerank(self, query: str, candidates: List[Dict], top_n: int) -> List[Dict]:
        reranker = self._get_reranker()
        if not reranker or not candidates:
            return candidates[:top_n]

        pairs = [(query, c.get("document", "") or "") for c in candidates]
        scores = reranker.predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)

        reranked = []
        for i in order[:top_n]:
            c = dict(candidates[i])
            c["rerank_score"] = float(scores[i])
            # Preserve "distance" (lower = better) for callers written against
            # the original vector-only contract — e.g. tools.py's
            # _exec_search_knowledge_base merges results from several query
            # variants and re-sorts the pool by this field. Cross-encoder
            # logits aren't perfectly comparable across separately-reranked
            # queries, but neither were the cosine distances that field held
            # before hybrid search existed — same approximation, not a new one.
            c["distance"] = -c["rerank_score"]
            reranked.append(c)
        return reranked

    # ── Public API ──────────────────────────────────────────────────────

    def retrieve(
        self, query: str, n_results: int = 5, filter_type: Optional[str] = None,
        use_hybrid: bool = True,
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.

        use_hybrid=True (default): dense vector search + BM25 keyword search,
        fused with Reciprocal Rank Fusion. use_hybrid=False keeps the
        original vector-only path. Cross-Encoder reranking on top of the
        fused list additionally requires settings.RAG_RERANK_ENABLED — see
        that setting's comment for why it isn't on by default.
        """
        vector_pool = self.embedding_manager.search(
            query=query, n_results=RERANK_CANDIDATE_POOL, filter_type=filter_type
        )

        if not use_hybrid:
            results = vector_pool[:n_results]
            logger.info(f"Retrieved {len(results)} documents (vector-only) for query: {query[:50]}...")
            return results

        bm25_pool = self._bm25_search(query, RERANK_CANDIDATE_POOL, filter_type)
        fused = self._reciprocal_rank_fusion([vector_pool, bm25_pool])

        if settings.RAG_RERANK_ENABLED:
            results = self._rerank(query, fused[:RERANK_CANDIDATE_POOL], n_results)
            stage = "reranked"
        else:
            results = fused[:n_results]
            stage = "RRF-only"

        logger.info(
            f"Retrieved {len(results)} documents (hybrid: {len(vector_pool)} vector + "
            f"{len(bm25_pool)} bm25 -> {len(fused)} fused -> {stage}) for query: {query[:50]}..."
        )
        return results

    def retrieve_with_scores(self, query: str, n_results: int = 5,
                            min_score: float = 0.0) -> List[Dict]:
        """
        Retrieve documents with similarity scores.

        Stays vector-only (use_hybrid=False): min_score is a cosine-similarity
        threshold, which only means something against ChromaDB's own
        distance metric — hybrid search's fused/reranked results don't carry
        a comparable [0,1] similarity score to threshold against.
        """
        results = self.retrieve(query, n_results=n_results, use_hybrid=False)

        # Filter by minimum score (distance is inverse of similarity)
        filtered = []
        for result in results:
            # Convert distance to similarity (1 - distance for cosine)
            if result.get("distance") is not None:
                similarity = 1 - result["distance"]
                if similarity >= min_score:
                    result["similarity"] = similarity
                    filtered.append(result)
            else:
                filtered.append(result)

        return filtered
