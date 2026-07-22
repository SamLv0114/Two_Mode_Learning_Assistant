"""
ArXiv RSS collector — fetches today's new submissions per category.
One HTTP request per category, no pagination, no time-slice bucketing needed.
ArXiv RSS updates daily at ~00:00 EST (05:00 UTC).
"""
import feedparser
import re
from datetime import datetime, timezone
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

RSS_BASE = "https://arxiv.org/rss/{category}"


@dataclass
class RSSPaperData:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    published_date: Optional[datetime]
    arxiv_url: str
    pdf_url: str
    citation_count: Optional[int] = None


class ArxivRSSCollector:
    """Fetches today's new ArXiv submissions via RSS."""

    def fetch(self, categories: List[str]) -> List[RSSPaperData]:
        """Fetch today's papers from the given ArXiv categories, deduplicated."""
        papers: dict[str, RSSPaperData] = {}

        for category in categories:
            url = RSS_BASE.format(category=category)
            try:
                feed = feedparser.parse(url)
                count_before = len(papers)
                for entry in feed.entries:
                    arxiv_id = self._extract_arxiv_id(entry)
                    if not arxiv_id or arxiv_id in papers:
                        continue
                    papers[arxiv_id] = RSSPaperData(
                        arxiv_id=arxiv_id,
                        title=entry.get("title", "").replace("\n", " ").strip(),
                        abstract=entry.get("summary", "").replace("\n", " ").strip(),
                        authors=self._extract_authors(entry),
                        categories=[category],
                        published_date=self._parse_date(entry),
                        arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
                        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                    )
                logger.info(f"ArXiv RSS {category}: {len(feed.entries)} entries, {len(papers) - count_before} new")
            except Exception as e:
                logger.error(f"Error fetching ArXiv RSS {category}: {e}")

        return list(papers.values())

    def _extract_arxiv_id(self, entry) -> Optional[str]:
        link = entry.get("link", "")
        if "/abs/" in link:
            arxiv_id = link.split("/abs/")[-1].strip()
            # Strip version suffix: "2401.12345v2" -> "2401.12345"
            arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
            return arxiv_id
        return None

    def _extract_authors(self, entry) -> List[str]:
        authors = entry.get("authors", [])
        if authors:
            return [a.get("name", "") for a in authors]
        author = entry.get("author", "")
        return [author] if author else []

    def _parse_date(self, entry) -> Optional[datetime]:
        if entry.get("published_parsed"):
            try:
                return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            except Exception:
                pass
        return None
