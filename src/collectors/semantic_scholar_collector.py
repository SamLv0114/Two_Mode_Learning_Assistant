"""
Semantic Scholar API collector.

Uses pre-computed embeddings on Semantic Scholar's side — no local embedding at query time.

Two modes:
  - search(query)     : cold start — keyword search over 200M+ papers
  - recommend(ids)    : with history — given saved ArXiv IDs, returns similar papers
"""
import logging
import requests
from datetime import datetime, timezone
from typing import List, Optional

from src.collectors.arxiv_collector import PaperData
from src.utils.config import settings

logger = logging.getLogger(__name__)

_BASE = "https://api.semanticscholar.org"
_FIELDS = "paperId,externalIds,title,abstract,year,citationCount,authors,publicationDate,openAccessPdf"
_TIMEOUT = 10


def _headers() -> dict:
    h = {"Accept": "application/json"}
    if settings.SEMANTIC_SCHOLAR_API_KEY:
        h["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY
    return h


def _to_paper_data(item: dict) -> Optional[PaperData]:
    """Convert a Semantic Scholar paper dict to PaperData. Returns None if no ArXiv ID."""
    external = item.get("externalIds") or {}
    arxiv_id = external.get("ArXiv")
    if not arxiv_id:
        return None

    pub_date = None
    if item.get("publicationDate"):
        try:
            pub_date = datetime.strptime(item["publicationDate"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            pass
    if pub_date is None and item.get("year"):
        pub_date = datetime(item["year"], 1, 1, tzinfo=timezone.utc)

    authors = [a.get("name", "") for a in (item.get("authors") or [])]

    return PaperData(
        arxiv_id=arxiv_id,
        title=(item.get("title") or "").strip(),
        abstract=(item.get("abstract") or "").strip(),
        authors=authors,
        categories=[],
        published_date=pub_date,
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        citation_count=item.get("citationCount"),
    )


class SemanticScholarCollector:

    def search(self, query: str, limit: int = 20) -> List[PaperData]:
        """
        Cold-start: keyword search. Semantic Scholar ranks by relevance.
        Returns only papers that have an ArXiv ID.
        """
        try:
            resp = requests.get(
                f"{_BASE}/graph/v1/paper/search",
                params={"query": query, "fields": _FIELDS, "limit": min(limit * 2, 100)},
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            items = resp.json().get("data", [])
            papers = [p for item in items if (p := _to_paper_data(item)) is not None]
            logger.info(f"S2 search '{query[:50]}': {len(papers)} ArXiv papers from {len(items)} results")
            return papers[:limit]
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []

    def recommend(self, positive_arxiv_ids: List[str], limit: int = 20) -> List[PaperData]:
        """
        With history: given the user's saved ArXiv IDs, returns similar papers.
        Falls back to search if conversion to S2 IDs fails.
        """
        s2_ids = self._arxiv_ids_to_s2_ids(positive_arxiv_ids[:10])
        if not s2_ids:
            logger.warning("Could not resolve any ArXiv IDs to S2 IDs — falling back to search")
            return []

        try:
            resp = requests.post(
                f"{_BASE}/recommendations/v1/papers",
                params={"fields": _FIELDS, "limit": min(limit * 2, 500)},
                json={"positivePaperIds": s2_ids, "negativePaperIds": []},
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            items = resp.json().get("recommendedPapers", [])
            papers = [p for item in items if (p := _to_paper_data(item)) is not None]
            logger.info(f"S2 recommend: {len(papers)} ArXiv papers from {len(items)} results")
            return papers[:limit]
        except Exception as e:
            logger.error(f"Semantic Scholar recommendations failed: {e}")
            return []

    def _arxiv_ids_to_s2_ids(self, arxiv_ids: List[str]) -> List[str]:
        """Batch-convert ArXiv IDs to Semantic Scholar paperIds (one API call)."""
        if not arxiv_ids:
            return []
        try:
            resp = requests.post(
                f"{_BASE}/graph/v1/paper/batch",
                params={"fields": "paperId"},
                json={"ids": [f"ArXiv:{aid}" for aid in arxiv_ids]},
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            results = resp.json()
            s2_ids = [r["paperId"] for r in results if r and r.get("paperId")]
            logger.info(f"Resolved {len(s2_ids)}/{len(arxiv_ids)} ArXiv IDs to S2 IDs")
            return s2_ids
        except Exception as e:
            logger.error(f"S2 batch ID conversion failed: {e}")
            return []
