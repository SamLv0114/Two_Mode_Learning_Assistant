"""
Semantic Scholar client for the recommendation and reading-path features.

Semantic Scholar's recommendations are built from the citation graph and from how
researchers read papers together, which is the collaborative signal this app cannot
collect from its own small user base.

Calls never raise: each returns None when Semantic Scholar gave no answer (network
error, rate limit after retries), so callers can tell "no answer" from "no papers".
Anonymous use shares one global rate limit, so calls are spaced out, a 429 is retried
with backoff, and answers are cached. Set SEMANTIC_SCHOLAR_API_KEY for a dedicated limit.
"""
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from src.utils.config import settings
from src.utils.redis_client import JsonCache

logger = logging.getLogger(__name__)

GRAPH_URL = "https://api.semanticscholar.org/graph/v1"
RECOMMENDATIONS_URL = "https://api.semanticscholar.org/recommendations/v1/papers"
FIELDS = "paperId,title,abstract,year,externalIds,authors,publicationDate,citationCount"
MIN_INTERVAL_SECONDS = 1.1      # one request per second is the documented limit
RETRIES = 2                     # after a 429 or a 5xx, with 2 s then 4 s waits
TIMEOUT_SECONDS = 20
_ARXIV_VERSION = re.compile(r"v\d+$")

_cache = JsonCache("s2", ttl_seconds=24 * 3600)
_lock = threading.Lock()
_last_call = 0.0


@dataclass
class S2Paper:
    paper_id: str
    title: str
    abstract: str = ""
    year: Optional[int] = None
    arxiv_id: str = ""            # without version suffix
    publication_date: str = ""
    authors: List[str] = field(default_factory=list)
    citation_count: int = 0

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}" if self.arxiv_id else \
            f"https://www.semanticscholar.org/paper/{self.paper_id}"

    @property
    def published(self) -> Optional[datetime]:
        """The publication date, or 1 January of the year when only the year is known."""
        try:
            return datetime.fromisoformat(self.publication_date)
        except ValueError:
            return datetime(self.year, 1, 1) if self.year else None

    @classmethod
    def from_api(cls, item: Dict[str, Any]) -> Optional["S2Paper"]:
        if not item or not item.get("paperId") or not item.get("title"):
            return None
        external = item.get("externalIds") or {}
        return cls(paper_id=item["paperId"], title=item["title"].strip(), abstract=(item.get("abstract") or "").strip(),
                   year=item.get("year"), arxiv_id=_ARXIV_VERSION.sub("", external.get("ArXiv") or ""),
                   publication_date=item.get("publicationDate") or "",
                   authors=[a.get("name", "") for a in item.get("authors") or [] if a.get("name")],
                   citation_count=int(item.get("citationCount") or 0))


def paper_ref(identifier: str) -> str:
    """The id form Semantic Scholar accepts: arXiv ids as ArXiv:<id> (no version), S2 ids unchanged."""
    identifier = identifier.strip()
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", identifier):
        return f"ArXiv:{_ARXIV_VERSION.sub('', identifier)}"
    return identifier


def _request(method: str, url: str, params: Dict[str, Any], body: Optional[Dict] = None) -> Optional[Any]:
    """The JSON answer (cached), or None after the retries."""
    key = hashlib.sha256(json.dumps([method, url, params, body], sort_keys=True).encode()).hexdigest()
    cached = _cache.get(key)
    if cached is not None:
        return cached
    data = _fetch(method, url, params, body)
    if data is not None:
        _cache.set(key, data)
    return data


def _fetch(method: str, url: str, params: Dict[str, Any], body: Optional[Dict]) -> Optional[Any]:
    global _last_call
    headers = {"x-api-key": settings.SEMANTIC_SCHOLAR_API_KEY} if settings.SEMANTIC_SCHOLAR_API_KEY else {}
    for attempt in range(RETRIES + 1):
        with _lock:  # space requests out across every thread in the process
            wait = MIN_INTERVAL_SECONDS - (time.monotonic() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.monotonic()
        try:
            response = httpx.request(method, url, params=params, json=body, headers=headers, timeout=TIMEOUT_SECONDS)
        except httpx.HTTPError as exc:
            logger.warning("Semantic Scholar request failed: %s", exc)
            return None
        if response.status_code == 200:
            return response.json()
        if response.status_code in (429, 500, 502, 503, 504) and attempt < RETRIES:
            time.sleep(2 * (attempt + 1))
            continue
        logger.warning("Semantic Scholar %s %s -> %s", method, url.rsplit("/", 1)[-1], response.status_code)
        return None
    return None


def _papers(items: List[Dict[str, Any]]) -> List[S2Paper]:
    return [p for p in (S2Paper.from_api(item) for item in items) if p]


def recommend(positive_ids: List[str], negative_ids: Optional[List[str]] = None,
              limit: int = 60) -> Optional[List[S2Paper]]:
    """Papers Semantic Scholar recommends for a set of liked (and disliked) papers; recent work first."""
    positives = [paper_ref(i) for i in positive_ids if i][:20]
    if not positives:
        return []
    data = _request("POST", RECOMMENDATIONS_URL, {"fields": FIELDS, "limit": min(limit, 100)},
                    {"positivePaperIds": positives, "negativePaperIds": [paper_ref(i) for i in negative_ids or []][:20]})
    return None if data is None else _papers(data.get("recommendedPapers", []))


def search(query: str, limit: int = 8) -> Optional[List[S2Paper]]:
    if not query.strip():
        return []
    data = _request("GET", f"{GRAPH_URL}/paper/search", {"query": query[:300], "limit": limit, "fields": FIELDS})
    return None if data is None else _papers(data.get("data", []))


def references(identifier: str, limit: int = 100) -> Optional[List[S2Paper]]:
    """Papers this paper cites (used to find the foundations of a topic)."""
    data = _request("GET", f"{GRAPH_URL}/paper/{paper_ref(identifier)}/references", {"fields": FIELDS, "limit": limit})
    return None if data is None else _papers([item.get("citedPaper") or {} for item in data.get("data", [])])


def as_dict(paper: S2Paper) -> Dict[str, Any]:
    return {**asdict(paper), "url": paper.url}
