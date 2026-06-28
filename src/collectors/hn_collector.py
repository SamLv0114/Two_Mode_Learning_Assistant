"""
Hacker News article collector
"""
import requests
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


class HNCollector:
    """Fetches articles from Hacker News"""
    
    def fetch(self, limit: int = 30, days: Optional[int] = None) -> List[ArticleData]:
        """Fetch top articles from Hacker News"""
        articles = []
        cutoff_date = None
        if days:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        try:
            # Hacker News API
            top_stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            response = requests.get(top_stories_url, timeout=10)
            story_ids = response.json()[:limit * 2]  # Fetch more to account for date filtering
            
            for story_id in story_ids:
                story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                story_data = requests.get(story_url, timeout=10).json()
                
                if story_data and story_data.get("type") == "story" and story_data.get("url"):
                    # Parse published date
                    published_date = None
                    if story_data.get("time"):
                        published_date = datetime.fromtimestamp(story_data.get("time", 0), tz=timezone.utc)
                    
                    # Filter by date if specified
                    if cutoff_date and published_date:
                        if published_date < cutoff_date:
                            continue
                    
                    # Extract content from URL
                    content = self._extract_content(story_data.get("url", ""))
                    
                    article = ArticleData(
                        source="hackernews",
                        source_id=str(story_id),
                        title=story_data.get("title", ""),
                        url=story_data.get("url", ""),
                        content=content or story_data.get("title", ""),
                        author=story_data.get("by"),
                        published_date=published_date,
                        upvotes=story_data.get("score", 0)
                    )
                    articles.append(article)
                    
                    if len(articles) >= limit:
                        break
                    
        except Exception as e:
            logger.error(f"Error fetching Hacker News articles: {e}")
        
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

