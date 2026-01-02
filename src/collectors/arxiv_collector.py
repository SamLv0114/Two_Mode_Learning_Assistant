"""
ArXiv paper collection module
"""
import arxiv
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass
from src.utils.config import settings
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PaperData:
    """Paper data structure"""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_date: datetime
    arxiv_url: str
    pdf_url: str
    citation_count: int = 0
    personalized_summary: Optional[str] = None
    relevance_score: Optional[float] = None


class ArxivCollector:
    """Fetches papers from ArXiv"""
    
    def __init__(self):
        self.categories = settings.ARXIV_CATEGORIES
        self.max_results = settings.MAX_PAPERS_PER_DAY
    
    def fetch_recent_papers(
        self,
        days: int = 1,
        max_results: Optional[int] = None,
        categories: Optional[List[str]] = None,
        enrich_citations: bool = False,
    ) -> List[PaperData]:
        """
        Fetch recent papers from ArXiv
        """
        max_results = max_results or self.max_results
        papers = []
        
        # Build query for categories
        use_categories = categories if categories else self.categories
        category_query = " OR ".join([f"cat:{cat}" for cat in use_categories])
        
        # Date filter (ArXiv doesn't support date filtering directly, so we fetch and filter)
        search = arxiv.Search(
            query=category_query,
            max_results=max_results * 2,  # Fetch more to account for filtering
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            for result in search.results():
                # Filter by date
                if result.published.date() >= cutoff_date.date():
                    citation_count = 0
                    paper_id = result.entry_id.split('/')[-1]
                    if enrich_citations and settings.CITATION_ENRICHMENT_ENABLED:
                        citation_count = self._fetch_citation_count(paper_id)

                    paper = PaperData(
                        arxiv_id=paper_id,
                        title=result.title,
                        authors=[author.name for author in result.authors],
                        abstract=result.summary,
                        categories=result.categories,
                        published_date=result.published,
                        arxiv_url=result.entry_id,
                        pdf_url=result.pdf_url,
                        citation_count=citation_count  # Best-effort enrichment (0 if unavailable)
                    )
                    papers.append(paper)
                    
                    if len(papers) >= max_results:
                        break
                        
        except Exception as e:
            logger.error(f"Error fetching ArXiv papers: {e}")
        
        logger.info(f"Fetched {len(papers)} papers from ArXiv")
        return papers

    def _fetch_citation_count(self, arxiv_id: str) -> int:
        """
        Best-effort citation count from Semantic Scholar. Returns 0 on failure.
        """
        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=citationCount"
            headers = {}
            if settings.SEMANTIC_SCHOLAR_API_KEY:
                headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY
            resp = requests.get(url, headers=headers, timeout=settings.CITATION_API_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                return int(data.get("citationCount", 0) or 0)
            else:
                logger.debug(f"Citation fetch failed for {arxiv_id}: {resp.status_code}")
        except Exception as e:
            logger.debug(f"Citation fetch error for {arxiv_id}: {e}")
        return 0
    
    def fetch_by_query(self, query: str, max_results: int = 10) -> List[PaperData]:
        """
        Fetch papers by search query
        """
        papers = []
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        try:
            for result in search.results():
                paper = PaperData(
                    arxiv_id=result.entry_id.split('/')[-1],
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    categories=result.categories,
                    published_date=result.published,
                    arxiv_url=result.entry_id,
                    pdf_url=result.pdf_url,
                    citation_count=0
                )
                papers.append(paper)
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
        
        return papers

