"""
Recall channels for a profile-driven feed, and the quota merge that turns them into one shortlist.

  content   the research profile searched against the corpus by meaning (ChromaDB)
  graph     Semantic Scholar's recommendations from the papers the user pinned, saved or
            implemented, with dismissed papers as negatives. This is the collaborative signal
            of the whole research community, which this app cannot collect from its own few
            users. Papers the corpus does not have yet are added to it, so this channel reaches
            beyond what the nightly arXiv job indexed.
  trending  this week's Hugging Face trending papers, kept only when close to the profile

Each channel returns nothing on failure; the content channel alone still makes a feed.
Channels only propose candidates. Ranking and the final choice happen after the merge.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import httpx
import numpy as np
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.collectors import semantic_scholar
from src.collectors.semantic_scholar import S2Paper
from src.database.models import Paper
from src.utils.redis_client import JsonCache

logger = logging.getLogger(__name__)

# The shortlist handed to ranking and the model's judgements: each channel's best papers
# up to its quota. Slots a channel cannot fill go to the others (see merge).
CHANNEL_QUOTAS = {"content": 18, "graph": 9, "trending": 3}
GRAPH_RECALL_LIMIT = 60
GRAPH_MAX_POSITIVES = 20
TRENDING_FETCH = 50
TRENDING_MIN_SIMILARITY = 0.35   # cosine to the profile; below this a trending paper is off-topic
HF_TRENDING_URL = "https://huggingface.co/api/papers"

_trending_cache = JsonCache("feed_trending", ttl_seconds=6 * 3600)


@dataclass
class OutsidePaper:
    """A paper that may not be in the corpus yet, with what ingest needs to add it."""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    published: Optional[datetime] = None
    citation_count: int = 0

    @classmethod
    def from_s2(cls, paper: S2Paper) -> "OutsidePaper":
        return cls(paper.arxiv_id, paper.title, paper.abstract, paper.authors, paper.published, paper.citation_count)


# ── Vectors and the corpus ────────────────────────────────────────────────

def normalise(matrix: np.ndarray) -> np.ndarray:
    """Rows (or a single vector) scaled to unit length; zero vectors are left as they are."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def query_vector(em, text: str) -> np.ndarray:
    return normalise(np.asarray(em.generate_embedding(text), dtype=float))


def paper_vectors(em, arxiv_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    """Stored vectors for corpus papers, unit length, keyed by arXiv id (missing ids left out)."""
    if not arxiv_ids:
        return {}
    try:
        stored = em.collection.get(ids=[f"paper_{a}" for a in arxiv_ids], include=["embeddings"])
    except Exception as exc:
        logger.warning("Could not read paper vectors: %s", exc)
        return {}
    embeddings = stored.get("embeddings")
    if embeddings is None:  # Chroma returns an array here, so test against None, not truthiness
        return {}
    return {doc_id.removeprefix("paper_"): normalise(np.asarray(emb, dtype=float))
            for doc_id, emb in zip(stored.get("ids") or [], embeddings)}


def search_corpus(db: Session, em, query: str, limit: int) -> List[Paper]:
    """The corpus papers closest to the query by meaning, closest first."""
    try:
        hits = em.search(query, n_results=limit, filter_type="paper")
    except Exception as exc:
        logger.warning("Corpus search failed: %s", exc)
        return []
    ids = [h["metadata"]["paper_id"] for h in hits if (h.get("metadata") or {}).get("paper_id")]
    rows = {p.arxiv_id: p for p in db.query(Paper).filter(Paper.arxiv_id.in_(ids)).all()} if ids else {}
    return [rows[a] for a in ids if a in rows]


def ingest(db: Session, em, papers: List[OutsidePaper]) -> List[str]:
    """
    Add the papers the corpus lacks (a Paper row and a vector), so they can be ranked, shown,
    saved and summarised like any other. Papers without an arXiv id, title or abstract are
    skipped. Returns the arXiv ids of the papers now in the corpus, in the order given.
    """
    usable: Dict[str, OutsidePaper] = {}
    for p in papers:
        if p.arxiv_id and p.title and p.abstract:
            usable.setdefault(p.arxiv_id, p)
    if not usable:
        return []

    def present() -> set:
        return {a for (a,) in db.query(Paper.arxiv_id).filter(Paper.arxiv_id.in_(list(usable))).all()}

    existing = present()
    new = [p for a, p in usable.items() if a not in existing]
    if new:
        for p in new:
            db.add(Paper(arxiv_id=p.arxiv_id, title=p.title[:500], abstract=p.abstract,
                         authors=", ".join(p.authors), categories="", published_date=p.published,
                         citation_count=p.citation_count, arxiv_url=f"https://arxiv.org/abs/{p.arxiv_id}",
                         pdf_url=f"https://arxiv.org/pdf/{p.arxiv_id}"))
        try:
            db.commit()
        except IntegrityError:  # another request added one of them first
            db.rollback()
            existing = present()
            return [a for a in usable if a in existing]
        try:
            em.collection.upsert(
                ids=[f"paper_{p.arxiv_id}" for p in new],
                embeddings=em.embed_papers([p.title for p in new], [p.abstract for p in new]),
                documents=[f"{p.title}\n\n{p.abstract}" for p in new],
                metadatas=[{"type": "paper", "paper_id": p.arxiv_id, "title": p.title[:500],
                            "url": f"https://arxiv.org/abs/{p.arxiv_id}", "published_date": str(p.published or ""),
                            "citation_count": p.citation_count} for p in new])
        except Exception as exc:
            logger.warning("Added %s papers to the database but not the vector store: %s", len(new), exc)
        logger.info("Added %s papers from outside the corpus", len(new))
    return list(usable)


# ── Graph channel ──────────────────────────────────────────────────────────

def graph_recall(db: Session, em, positive_ids: List[str], negative_ids: List[str],
                 limit: int = GRAPH_RECALL_LIMIT) -> List[str]:
    """arXiv ids Semantic Scholar recommends from the liked papers, in its order."""
    positives = list(dict.fromkeys(i for i in positive_ids if i))[:GRAPH_MAX_POSITIVES]
    if not positives:
        return []
    papers = semantic_scholar.recommend(positives, negative_ids, limit=limit) or []
    return ingest(db, em, [OutsidePaper.from_s2(p) for p in papers])


# ── Trending channel ───────────────────────────────────────────────────────

def trending_records() -> List[Dict]:
    """Hugging Face's trending papers (cached for six hours)."""
    key = f"{datetime.now(timezone.utc):%Y-%m-%d}"
    cached = _trending_cache.get(key)
    if cached is not None:
        return cached
    try:
        response = httpx.get(HF_TRENDING_URL, params={"sort": "trending", "limit": TRENDING_FETCH},
                             headers={"User-Agent": "ResearchMate/1.0"}, timeout=10)
        response.raise_for_status()
        items = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Hugging Face trending fetch failed: %s", exc)
        return []
    records = [{"arxiv_id": item["id"], "title": (item.get("title") or "").strip(),
                "abstract": (item.get("summary") or "").strip(),
                "authors": [a.get("name", "") for a in item.get("authors") or [] if a.get("name")],
                "published_at": item.get("publishedAt") or "", "upvotes": int(item.get("upvotes") or 0)}
               for item in items if isinstance(item, dict) and item.get("id")]
    _trending_cache.set(key, records)
    return records


def _iso_date(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def trending_recall(db: Session, em, profile_vec: np.ndarray,
                    min_similarity: float = TRENDING_MIN_SIMILARITY) -> Dict[str, int]:
    """{arXiv id: upvotes} for trending papers close enough to the profile, most upvoted first."""
    records = [r for r in trending_records() if r["title"] and r["abstract"]]
    vectors = paper_vectors(em, [r["arxiv_id"] for r in records])
    unseen = [r for r in records if r["arxiv_id"] not in vectors]
    if unseen:
        fresh = em.embed_papers([r["title"] for r in unseen], [r["abstract"] for r in unseen])
        vectors.update({r["arxiv_id"]: normalise(np.asarray(v, dtype=float)) for r, v in zip(unseen, fresh)})
    close = sorted((r for r in records if float(vectors[r["arxiv_id"]] @ profile_vec) >= min_similarity),
                   key=lambda r: r["upvotes"], reverse=True)
    present = set(ingest(db, em, [OutsidePaper(r["arxiv_id"], r["title"], r["abstract"], r["authors"],
                                               _iso_date(r["published_at"])) for r in close]))
    return {r["arxiv_id"]: r["upvotes"] for r in close if r["arxiv_id"] in present}


# ── Merge ──────────────────────────────────────────────────────────────────

def merge(ranked: Dict[str, List[str]], quotas: Dict[str, int] = CHANNEL_QUOTAS) -> List[str]:
    """
    One shortlist from per-channel lists (each already in ranking order).

    Every channel first gets its quota, so a strong content channel cannot crowd the graph
    and trending papers out before they are judged. Slots a channel leaves empty are then
    filled from the others, best-ranked first, in quota order.
    """
    total = sum(quotas.values())
    picked: List[str] = []
    for channel, quota in quotas.items():
        fresh = [a for a in ranked.get(channel, []) if a not in picked]
        picked.extend(fresh[:quota])
    for channel in quotas:
        for arxiv_id in ranked.get(channel, []):
            if len(picked) >= total:
                return picked
            if arxiv_id not in picked:
                picked.append(arxiv_id)
    return picked
