"""
Dev.to article collector
"""
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
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


class DevToCollector:
    """Fetches articles from Dev.to"""
    
    def fetch(self, limit: int = 20) -> List[ArticleData]:
        """Fetch articles from Dev.to"""
        articles = []
        
        try:
            # Dev.to RSS feed
            feed_url = "https://dev.to/feed"
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:limit]:
                content = self._extract_content(entry.link)
                
                article = ArticleData(
                    source="devto",
                    source_id=entry.get("id", entry.link),
                    title=entry.title,
                    url=entry.link,
                    content=content or entry.get("summary", ""),
                    author=entry.get("author"),
                    published_date=datetime(*entry.published_parsed[:6]) if entry.get("published_parsed") else None,
                    upvotes=0  # Dev.to doesn't provide upvotes in RSS
                )
                articles.append(article)
                
        except Exception as e:
            logger.error(f"Error fetching Dev.to articles: {e}")
        
        return articles
    
    def _extract_content(self, url: str) -> str:
        """Extract text content from a URL"""
        try:
            response = requests.get(url, timeout=10, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)
            
            return text[:5000]  # Limit content length
            
        except Exception as e:
            logger.debug(f"Could not extract content from {url}: {e}")
            return ""

