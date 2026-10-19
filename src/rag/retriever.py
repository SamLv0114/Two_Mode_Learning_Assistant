"""
Hybrid (dense + lexical) retriever for RAG.
"""
from array import array
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging
import string
import threading
import time

import numpy as np
from prometheus_client import Gauge
from scipy import sparse

from src.models.embeddings import EmbeddingManager
from src.utils.config import settings
from src.utils.tracing import traceable
from src.rag.semantic_cache import SemanticRetrievalCache
from src.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

# How long a lazily-built BM25 index is trusted before being rebuilt from
# ChromaDB's current contents. Bounds staleness against papers the nightly
# indexer adds while this process keeps running, without wiring an explicit
# refresh call into that job — matches the cache-TTL idiom already used
# elsewhere in this project (e.g. What's Hot's 6h Redis cache) rather than
# inventing a new invalidation scheme.
BM25_INDEX_TTL_SECONDS = 6 * 3600

# After a failed build, wait this long before reading the corpus again. A
# failure used to be retried by every new Retriever, i.e. on every chat
# request, and each attempt loaded the whole corpus into memory first.
BM25_RETRY_BACKOFF_SECONDS = 10 * 60

# Records per ChromaDB read. Chroma binds one SQLite variable per record when
# it loads metadata, and SQLite rejects a statement with more than ~32k
# variables — an unpaginated collection.get() started failing in production
# with "too many SQL variables" once the corpus passed that size (36k rows),
# which silently turned hybrid search into vector-only search.
BM25_PAGE_SIZE = 2000

# Only public, whole-document content goes into the shared index.
# paper_fulltext chunks exist for fetch_full_paper's per-paper lookups, and
# user_doc chunks are private, so both are kept out of it.
BM25_PUBLIC_TYPES = ("paper",)

# Private uploads are scored per request from the requesting user's own
# chunks. Bounded so an unusually large upload set cannot turn one search
# into an unbounded read.
BM25_USER_DOC_MAX_CHUNKS = 5000

# Okapi BM25 parameters, identical to rank_bm25.BM25Okapi's defaults so
# scores stay comparable with the evaluation baselines built on it.
BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25

# How many candidates the fused (vector + BM25) shortlist carries into the
# Cross-Encoder rerank stage. Bounds rerank cost (it scores every candidate
# against the query individually, unlike vector/BM25 which score against a
# precomputed index) regardless of how wide the two upstream searches are.
RERANK_CANDIDATE_POOL = 30

BM25_INDEX_AVAILABLE = Gauge(
    "rag_bm25_index_available",
    "1 when the shared BM25 index is built and serving, 0 otherwise",
)
BM25_INDEX_DOCUMENTS = Gauge(
    "rag_bm25_index_documents",
    "Documents in the shared BM25 index",
)


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


@dataclass
class _BM25Index:
    """
    Immutable Okapi BM25 index over document ids only.

    Each cell holds a term's precomputed BM25 contribution to one document,
    so scoring a query is a sparse column gather plus a sum. Document text
    and metadata are not kept — callers fetch them for the few ids that
    rank, which is what keeps the index small enough for a 2 GB host.
    """
    doc_ids: List[str]
    doc_types: np.ndarray
    type_codes: Dict[str, int]
    vocabulary: Dict[str, int]
    weights: sparse.csc_matrix

    def __len__(self) -> int:
        return len(self.doc_ids)

    @property
    def nbytes(self) -> int:
        w = self.weights
        return w.data.nbytes + w.indices.nbytes + w.indptr.nbytes + self.doc_types.nbytes

    def score(self, query: str) -> np.ndarray:
        counts = Counter(t for t in _tokenize(query) if t in self.vocabulary)
        if not counts or not self.doc_ids:
            return np.zeros(len(self.doc_ids), dtype=np.float32)
        columns = [self.vocabulary[t] for t in counts]
        repeats = np.array([counts[t] for t in counts], dtype=np.float32)
        return np.asarray(self.weights[:, columns] @ repeats).ravel()

    def top(self, query: str, n: int, doc_type: Optional[str] = None) -> List[Tuple[int, float]]:
        scores = self.score(query)
        if doc_type is not None:
            code = self.type_codes.get(doc_type)
            if code is None:
                return []
            scores = np.where(self.doc_types == code, scores, 0.0)
        # With the idf floor every idf is positive, so a zero score means the
        # document shares no term with the query.
        matched = np.flatnonzero(scores > 0)
        if matched.size == 0:
            return []
        if matched.size > n:
            matched = matched[np.argpartition(-scores[matched], n - 1)[:n]]
        order = matched[np.argsort(-scores[matched], kind="stable")]
        return [(int(i), float(scores[i])) for i in order]


def _build_index(records: Iterable[Tuple[str, Optional[str], Optional[Dict]]],
                 okapi_idf: bool = True) -> _BM25Index:
    """
    Build an index from (id, document, metadata) records in one streaming
    pass. Term ids and frequencies accumulate in typed arrays rather than
    Python lists, so peak memory stays near the final matrix size instead of
    several times it.

    okapi_idf=True reproduces rank_bm25.BM25Okapi exactly (used for the
    shared corpus, so scores match the evaluation baselines). Its idf is
    negative for any term in more than half the documents and its floor is
    a fraction of the *mean* idf, which is itself negative for a corpus of a
    handful of documents — so a single uploaded file could never match
    anything. okapi_idf=False uses Lucene's always-positive
    log(1 + (N - df + 0.5) / (df + 0.5)) for those small per-user indexes.
    """
    doc_ids: List[str] = []
    type_codes: Dict[str, int] = {}
    doc_types = array("b")
    vocabulary: Dict[str, int] = {}
    rows, cols, tfs = array("i"), array("i"), array("f")
    lengths = array("f")

    for doc_id, document, metadata in records:
        row = len(doc_ids)
        doc_ids.append(doc_id)
        doc_type = (metadata or {}).get("type") or ""
        doc_types.append(type_codes.setdefault(doc_type, len(type_codes)))
        tokens = _tokenize(document)
        lengths.append(len(tokens))
        for token, tf in Counter(tokens).items():
            rows.append(row)
            cols.append(vocabulary.setdefault(token, len(vocabulary)))
            tfs.append(tf)

    n_docs, n_terms = len(doc_ids), len(vocabulary)
    rows_np = np.frombuffer(rows, dtype=np.int32) if rows else np.zeros(0, np.int32)
    cols_np = np.frombuffer(cols, dtype=np.int32) if cols else np.zeros(0, np.int32)
    tf_np = np.frombuffer(tfs, dtype=np.float32) if tfs else np.zeros(0, np.float32)
    length_np = np.frombuffer(lengths, dtype=np.float32) if lengths else np.zeros(0, np.float32)

    if n_terms:
        df = np.bincount(cols_np, minlength=n_terms).astype(np.float64)
        if okapi_idf:
            idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
            # rank_bm25's floor: terms in more than half the corpus would get
            # a negative idf; they are given a small share of the mean.
            idf = np.where(idf < 0, BM25_EPSILON * idf.mean(), idf)
        else:
            idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        avg_length = float(length_np.mean()) if n_docs else 0.0
        norm = BM25_K1 * (1 - BM25_B + BM25_B * length_np / (avg_length or 1.0))
        data = (idf[cols_np] * tf_np * (BM25_K1 + 1) / (tf_np + norm[rows_np])).astype(np.float32)
    else:
        data = np.zeros(0, np.float32)

    weights = sparse.csc_matrix((data, (rows_np, cols_np)), shape=(n_docs, n_terms), dtype=np.float32)
    return _BM25Index(
        doc_ids=doc_ids,
        doc_types=np.frombuffer(doc_types, dtype=np.int8).copy() if doc_types else np.zeros(0, np.int8),
        type_codes=type_codes,
        vocabulary=vocabulary,
        weights=weights,
    )


def _paged_records(collection, where: Dict, page_size: Optional[int] = None,
                   max_records: Optional[int] = None):
    """Yield (id, document, metadata) from ChromaDB one bounded page at a time."""
    page_size = page_size or BM25_PAGE_SIZE
    offset = 0
    while True:
        limit = page_size if max_records is None else min(page_size, max_records - offset)
        if limit <= 0:
            return
        page = collection.get(where=where, limit=limit, offset=offset,
                              include=["documents", "metadatas"])
        ids = page.get("ids") or []
        documents = page.get("documents") or [None] * len(ids)
        metadatas = page.get("metadatas") or [None] * len(ids)
        yield from zip(ids, documents, metadatas)
        if len(ids) < limit:
            return
        offset += len(ids)


class _SharedIndexCache:
    """
    One public BM25 index per Chroma collection, shared by every Retriever
    in the process. The chat endpoint constructs a Retriever per request, so
    an index held on the instance was rebuilt for every conversation.
    """

    @dataclass
    class _Entry:
        index: Optional[_BM25Index] = None
        built_at: float = 0.0
        failed_at: float = 0.0
        lock: Any = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, "_SharedIndexCache._Entry"] = {}

    @staticmethod
    def _key(collection) -> str:
        return str(getattr(collection, "id", None) or id(collection))

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
        BM25_INDEX_AVAILABLE.set(0)
        BM25_INDEX_DOCUMENTS.set(0)

    def get(self, collection) -> Optional[_BM25Index]:
        with self._lock:
            entry = self._entries.setdefault(self._key(collection), self._Entry(lock=threading.Lock()))
        now = time.time()
        if entry.index is not None and now - entry.built_at < BM25_INDEX_TTL_SECONDS:
            return entry.index
        if now - entry.failed_at < BM25_RETRY_BACKOFF_SECONDS:
            return entry.index  # stale index or None; do not hammer a failing corpus read
        stale = entry.index
        if stale is not None:
            # Keep serving the stale index while one background thread refreshes it.
            if entry.lock.acquire(blocking=False):
                threading.Thread(target=self._refresh, args=(entry, collection, True),
                                 daemon=True, name="bm25-refresh").start()
            return stale
        with entry.lock:
            if entry.index is None and time.time() - entry.failed_at >= BM25_RETRY_BACKOFF_SECONDS:
                self._refresh(entry, collection, False)
        return entry.index

    @staticmethod
    def _refresh(entry, collection, release_lock: bool) -> None:
        started = time.time()
        try:
            index = _build_index(_paged_records(collection, {"type": {"$in": list(BM25_PUBLIC_TYPES)}}))
            entry.index, entry.built_at, entry.failed_at = index, time.time(), 0.0
            BM25_INDEX_AVAILABLE.set(1)
            BM25_INDEX_DOCUMENTS.set(len(index))
            logger.info(
                "BM25 index built: %d documents, %d terms, %d nonzeros, %.1f MB, %.1fs",
                len(index), len(index.vocabulary), index.weights.nnz,
                index.nbytes / 2**20, time.time() - started,
            )
        except Exception as e:
            entry.failed_at = time.time()
            BM25_INDEX_AVAILABLE.set(1 if entry.index is not None else 0)
            logger.warning(
                "BM25 index build failed (%s) — %s; next attempt in %ds",
                e, "serving the previous index" if entry.index is not None else "hybrid search is vector-only",
                BM25_RETRY_BACKOFF_SECONDS,
            )
        finally:
            if release_lock:
                entry.lock.release()


_BM25_CACHE = _SharedIndexCache()


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

    The BM25 half is a process-wide index over public papers (see
    _SharedIndexCache), read from ChromaDB in pages and stored as a sparse
    matrix of ids and term weights; private uploads are scored per request.

    Cross-Encoder reranking (settings.RAG_RERANK_ENABLED, on by default) is a
    further precision pass over the fused shortlist using real
    query-document attention instead of precomputed embeddings. It is behind
    a flag rather than unconditional because in local dev testing it once
    crashed the process with a native SIGBUS on the simplest possible input
    (device="cpu" explicitly set) — a failure mode try/except cannot catch,
    since no Python exception is ever raised. Root-caused to a macOS-only
    Accelerate-BLAS-vs-bundled-OpenBLAS conflict under memory pressure (see
    settings.RAG_RERANK_ENABLED's comment for the full diagnostic chain) —
    confirmed safe by running the exact CrossEncoder.predict call directly on
    the Linux production container (python:3.11-slim): returned a real
    score, no crash. Enabled by default on that evidence; the flag stays so
    it can be switched off again without a code change if a similar failure
    ever surfaces on a different deployment target.

    Both the BM25 index and the reranker degrade to vector-only if they fail
    to build or load; the BM25 failure is logged and exported as the
    rag_bm25_index_available gauge so it is not silent.
    """

    _reranker_instance = None
    _reranker_lock = threading.Lock()

    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.cache = None
        if settings.SEMANTIC_CACHE_ENABLED:
            client = get_redis()
            if client is not None:
                self.cache = SemanticRetrievalCache(client, self.embedding_manager.generate_embedding)

    # ── Lexical (BM25) search ───────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return _tokenize(text)

    def _public_bm25(self, query: str, n_results: int, filter_type: Optional[str]) -> List[Dict]:
        collection = self.embedding_manager.collection
        index = _BM25_CACHE.get(collection)
        if index is None:
            return []
        hits = index.top(query, n_results, doc_type=filter_type)
        if not hits:
            return []
        ids = [index.doc_ids[i] for i, _ in hits]
        fetched = collection.get(ids=ids, include=["documents", "metadatas"])
        by_id = {
            doc_id: (document, metadata)
            for doc_id, document, metadata in zip(
                fetched.get("ids") or [], fetched.get("documents") or [], fetched.get("metadatas") or [])
        }
        results = []
        for (_, score), doc_id in zip(hits, ids):
            if doc_id in by_id:  # skip ids deleted since the index was built
                document, metadata = by_id[doc_id]
                results.append({"id": doc_id, "document": document,
                                "metadata": metadata or {}, "bm25_score": score})
        return results

    def _user_doc_bm25(self, query: str, n_results: int, user_id: int) -> List[Dict]:
        """Score only this user's uploads; nothing private enters the shared index."""
        where = {"$and": [{"type": "user_doc"}, {"user_id": user_id}]}
        records = list(_paged_records(self.embedding_manager.collection, where,
                                      max_records=BM25_USER_DOC_MAX_CHUNKS))
        if not records:
            return []
        hits = _build_index(records, okapi_idf=False).top(query, n_results)
        return [{"id": records[i][0], "document": records[i][1],
                 "metadata": records[i][2] or {}, "bm25_score": score} for i, score in hits]

    def _bm25_search(
        self, query: str, n_results: int, filter_type: Optional[str],
        user_id: Optional[int] = None,
    ) -> List[Dict]:
        results: List[Dict] = []
        if filter_type in (None, *BM25_PUBLIC_TYPES):
            try:
                results.extend(self._public_bm25(query, n_results, filter_type))
            except Exception as e:
                logger.warning("BM25 public search failed: %s", e)
        # "user_doc" chunks are private per-user — they are only ever scored
        # for their owner (mirrors the scoping EmbeddingManager.search applies
        # on the vector side), and never without an identity.
        if filter_type in (None, "user_doc") and user_id is not None:
            try:
                results.extend(self._user_doc_bm25(query, n_results, user_id))
            except Exception as e:
                logger.warning("BM25 user-document search failed: %s", e)
        # Scores from the two indexes are not strictly comparable (different
        # idf statistics); RRF downstream only uses the resulting rank order.
        results.sort(key=lambda r: r["bm25_score"], reverse=True)
        return results[:n_results]

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

    @traceable(run_type="retriever", name="retrieve",
               process_inputs=lambda i: {k: i.get(k) for k in ("query", "n_results", "filter_type", "use_hybrid")})
    def retrieve(
        self, query: str, n_results: int = 5, filter_type: Optional[str] = None,
        use_hybrid: bool = True, user_id: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.

        use_hybrid=True (default): dense vector search + BM25 keyword search,
        fused with Reciprocal Rank Fusion. use_hybrid=False keeps the
        original vector-only path. Cross-Encoder reranking on top of the
        fused list is controlled by settings.RAG_RERANK_ENABLED (on by
        default; see that setting's comment).

        user_id: requesting user's id, used to scope "user_doc" (private
        upload) results to their owner — see EmbeddingManager.search. Pass it
        on every user-facing call; only a trusted system/background path
        should omit it.
        """
        cache_config = None
        if self.cache and self.cache.eligible(filter_type):
            try:
                cache_config = {
                    "corpus_count": self.embedding_manager.collection.count(),
                    "corpus_revision": self.cache.redis.get("rag:public_corpus_revision") or "0",
                    "user_id": user_id,
                    "filter_type": filter_type, "n_results": n_results,
                    "use_hybrid": use_hybrid, "rerank": settings.RAG_RERANK_ENABLED,
                    "embedding_model": settings.EMBEDDING_MODEL, "version": 1,
                }
                cached = self.cache.get(query, cache_config)
                if cached is not None:
                    return cached
            except Exception as exc:
                logger.debug("Semantic cache lookup skipped: %s", exc)
                cache_config = None

        vector_pool = self.embedding_manager.search(
            query=query, n_results=RERANK_CANDIDATE_POOL, filter_type=filter_type,
            user_id=user_id,
        )

        if not use_hybrid:
            results = vector_pool[:n_results]
            if cache_config is not None:
                self.cache.put(query, cache_config, results)
            logger.info(f"Retrieved {len(results)} documents (vector-only) for query: {query[:50]}...")
            return results

        bm25_pool = self._bm25_search(query, RERANK_CANDIDATE_POOL, filter_type, user_id=user_id)
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
        if cache_config is not None:
            self.cache.put(query, cache_config, results)
        return results
