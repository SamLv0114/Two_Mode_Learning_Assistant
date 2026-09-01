"""
WebArticleAgent: discovers recent CS/AI articles using Tavily web search.

Replaces static scrapers (HN, Dev.to, Medium) with dynamic, interest-driven
discovery. Each feed generation runs up to MAX_QUERIES targeted searches based
on the user's expanded interests, deduplicates results, and returns structured
dicts ready to upsert into the Article table.

Called from DailyFeedPipeline._discover_articles() instead of the disabled
_collect_articles().
"""
import hashlib
import logging
from typing import List, Dict
from urllib.parse import urlparse

from src.utils.config import settings

logger = logging.getLogger(__name__)

_PREFERRED_DOMAINS = [
    "huggingface.co",
    "openai.com",
    "deepmind.google",
    "anthropic.com",
    "pytorch.org",
    "towardsdatascience.com",
    "sebastianraschka.com",
    "lilianweng.github.io",
    "karpathy.github.io",
    "ai.googleblog.com",
    "bair.berkeley.edu",
    "distill.pub",
    "paperswithcode.com",
    "newsletter.theaiedge.io",
    "thegradient.pub",
]

_EXCLUDE_DOMAINS = [
    "arxiv.org",        # papers handled separately by nightly indexer
    "youtube.com",
    "twitter.com",
    "x.com",
    "reddit.com",
    "linkedin.com",
]


class WebArticleAgent:
    """
    Discovers recent CS/AI community content using Tavily web search.

    Strategy:
      - Builds N queries from expanded user interests (alternating tutorial
        vs. news framing)
      - Prefers known high-quality CS/ML blogs via include_domains
      - Deduplicates by URL across all queries
      - Returns at most max_articles dicts

    Returns [] when TAVILY_API_KEY is unset or tavily-python is not installed.
    """

    MAX_QUERIES = 4
    RESULTS_PER_QUERY = 4

    def discover(self, interests: List[str], max_articles: int = 10) -> List[Dict]:
        """
        Search for recent CS articles matching user interests.

        Args:
            interests: Expanded interest strings, e.g. ["machine learning", "NLP"].
            max_articles: Cap on total returned articles.

        Returns:
            List of dicts with keys:
              title, url, source, source_id, content, author, published_date, upvotes
        """
        if not settings.TAVILY_API_KEY:
            logger.info("TAVILY_API_KEY not configured — WebArticleAgent skipping")
            return []

        queries = self._build_queries(interests)
        seen_urls: set = set()
        articles: List[Dict] = []

        for query in queries[: self.MAX_QUERIES]:
            results = self._search(query)
            for r in results:
                url = (r.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                articles.append({
                    "title": (r.get("title") or "Untitled")[:500],
                    "url": url,
                    "source": self._extract_domain(url),
                    "source_id": "web_" + hashlib.md5(url.encode()).hexdigest()[:12],
                    "content": (r.get("content") or r.get("snippet") or "")[:3000],
                    "author": "",
                    "published_date": None,
                    "upvotes": 0,
                })

        logger.info(
            f"WebArticleAgent: {len(queries[:self.MAX_QUERIES])} queries "
            f"→ {len(articles)} unique articles"
        )
        return articles[:max_articles]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_queries(self, interests: List[str]) -> List[str]:
        """
        Build search queries from user interests.
        Alternates between tutorial/guide framing and news/release framing
        for coverage diversity.
        """
        queries: List[str] = []
        for i, interest in enumerate(interests[: self.MAX_QUERIES]):
            if i % 2 == 0:
                queries.append(f"{interest} tutorial guide explained 2025")
            else:
                queries.append(f"latest {interest} research release blog 2025")
        if not queries:
            queries = ["machine learning AI deep learning blog post 2025"]
        return queries

    def _search(self, query: str) -> List[Dict]:
        """Execute a single Tavily search, returning raw result dicts."""
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            resp = client.search(
                query=query,
                search_depth="basic",
                max_results=self.RESULTS_PER_QUERY,
                include_domains=_PREFERRED_DOMAINS,
                exclude_domains=_EXCLUDE_DOMAINS,
            )
            return resp.get("results", [])
        except ImportError:
            logger.warning("tavily-python not installed — run: pip install tavily-python")
            return []
        except Exception as e:
            logger.warning(f"Tavily search failed for '{query}': {e}")
            return []

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            return urlparse(url).netloc.removeprefix("www.")
        except Exception:
            return "web"
