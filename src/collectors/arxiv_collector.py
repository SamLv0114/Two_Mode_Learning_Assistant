"""
ArXiv paper collection module
"""
import arxiv
from arxiv import Client
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass
from src.utils.config import settings
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
    citation_count: Optional[int] = None
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
    ) -> List[PaperData]:
        """
        Fetch recent papers from ArXiv
        """
        max_results = max_results or self.max_results
        papers = []
        
        # Build query for categories
        use_categories = categories if categories else self.categories
        category_query = " OR ".join([f"cat:{cat}" for cat in use_categories])
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Try modern pagination via Client.results (preferred)
        try:
            client = Client(
                page_size=min(200, max_results * 2),
                delay_seconds=3,
            )
            search = arxiv.Search(
                query=category_query,
                max_results=max_results * 2,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            for result in client.results(search):
                if len(papers) >= max_results:
                    break
                if result.published.date() >= cutoff_date.date():
                    paper_id = result.entry_id.split('/')[-1]
                    paper = PaperData(
                        arxiv_id=paper_id,
                        title=result.title,
                        authors=[author.name for author in result.authors],
                        abstract=result.summary,
                        categories=result.categories,
                        published_date=result.published,
                        arxiv_url=result.entry_id,
                        pdf_url=result.pdf_url,
                    )
                    papers.append(paper)
        except Exception as e:
            logger.warning(f"Client pagination failed, falling back to single search: {e}")
            try:
                search = arxiv.Search(
                    query=category_query,
                    max_results=max_results * 2,  # overfetch then trim/filter
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )
                for result in search.results():
                    if len(papers) >= max_results:
                        break
                    if result.published.date() >= cutoff_date.date():
                        paper_id = result.entry_id.split('/')[-1]
                        paper = PaperData(
                            arxiv_id=paper_id,
                            title=result.title,
                            authors=[author.name for author in result.authors],
                            abstract=result.summary,
                            categories=result.categories,
                            published_date=result.published,
                            arxiv_url=result.entry_id,
                            pdf_url=result.pdf_url,
                        )
                        papers.append(paper)
            except Exception as e2:
                logger.error(f"Error fetching ArXiv papers: {e2}")
        
        logger.info(f"Fetched {len(papers)} papers from ArXiv")
        return papers
    
    def fetch_by_query(self, query: str, max_results: int = 10, categories: Optional[List[str]] = None) -> List[PaperData]:
        """
        Fetch papers by search query (optionally constrained by categories)
        """
        papers = []
        category_query = None
        if categories:
            category_query = " OR ".join([f"cat:{cat}" for cat in categories])
            query = f"({query}) AND ({category_query})"
        
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
                )
                papers.append(paper)
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
        
        return papers

