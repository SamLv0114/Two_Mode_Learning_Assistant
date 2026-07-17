"""
Dev.to article collector
"""
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ArticleData:
    """Article data structure"""
    source: str
    source_id: str
    title: str
    url: str
    content: str
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    upvotes: int = 0
    personalized_summary: Optional[str] = None
    relevance_score: Optional[float] = None


class DevToCollector:
    """Fetches articles from Dev.to"""
    
    def fetch(self, limit: int = 30, days: Optional[int] = None) -> List[ArticleData]:
        """Fetch articles from Dev.to"""
        articles = []
        cutoff_date = None
        if days:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        try:
            # Dev.to RSS feed
            feed_url = "https://dev.to/feed"
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:limit * 2]:  # Fetch more to account for date filtering
                # Parse published date
                published_date = None
                if entry.get("published_parsed"):
                    published_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                
                # Filter by date if specified
                if cutoff_date and published_date:
                    if published_date < cutoff_date:
                        continue
                
                # Use RSS summary directly — avoids scraping each article URL
                raw_summary = entry.get("summary", "") or entry.get("description", "")
                content = BeautifulSoup(raw_summary, "html.parser").get_text()[:5000] if raw_summary else entry.title

                article = ArticleData(
                    source="devto",
                    source_id=entry.get("id", entry.link),
                    title=entry.title,
                    url=entry.link,
                    content=content,
                    author=entry.get("author"),
                    published_date=published_date,
                    upvotes=0  # Dev.to doesn't provide upvotes in RSS
                )
                articles.append(article)
                
                if len(articles) >= limit:
                    break

        except Exception as e:
            logger.error(f"Error fetching Dev.to articles: {e}")

        return articles

