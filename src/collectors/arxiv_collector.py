"""
ArXiv paper collection module
"""
import arxiv
from arxiv import Client
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
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
    doi: Optional[str] = None
    journal_ref: Optional[str] = None


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
        
        # Try modern pagination via Client.results
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
                    # Extract DOI and journal_ref if available
                    doi = getattr(result, 'doi', None) or None
                    journal_ref = getattr(result, 'journal_ref', None) or None
                    paper = PaperData(
                        arxiv_id=paper_id,
                        title=result.title,
                        authors=[author.name for author in result.authors],
                        abstract=result.summary,
                        categories=result.categories,
                        published_date=result.published,
                        arxiv_url=result.entry_id,
                        pdf_url=result.pdf_url,
                        doi=doi,
                        journal_ref=journal_ref,
                    )
                    papers.append(paper)
        except Exception as e:
            logger.warning(f"Client pagination failed.")
        return papers
    
    def fetch_by_time_slices(
        self,
        days: int,
        candidates_per_slice: int = 300,
        num_slices: Optional[int] = None,
        categories: Optional[List[str]] = None,
    ) -> List[PaperData]:
        """
        Fetch papers by dividing time window into equal slices and sampling from each
        
        This ensures we get papers from across the entire time window, not just the most recent.
        Examples:
        - 1 year (365 days) → 12 monthly slices → ~30 days per slice
        - 1 month (30 days) → 4 weekly slices → ~7 days per slice
        - 1 week (7 days) → 7 daily slices → 1 day per slice
        
        Args:
            days: Total time window in days
            candidates_per_slice: Target number of candidates per slice (200-500)
            num_slices: Number of time slices (auto-calculated if None)
            categories: ArXiv categories to filter by
            
        Returns:
            List of PaperData objects
        """
        papers = []
        use_categories = categories if categories else self.categories
        category_query = " OR ".join([f"cat:{cat}" for cat in use_categories])
        
        # Auto-calculate number of slices if not provided
        if num_slices is None:
            if days >= 365:
                num_slices = 12  # Monthly slices for 1 year
            elif days >= 30:
                num_slices = 4  # Weekly slices for 1 month
            elif days >= 7:
                num_slices = 7  # Daily slices for 1 week
            else:
                num_slices = days  # Daily slices for < 7 days
        
        # Calculate slice duration
        slice_duration_days = days / num_slices
        now = datetime.now()
        
        logger.info(f"Dividing {days} days into {num_slices} slices (~{slice_duration_days:.1f} days per slice)")
        
        # Process each time slice (newest first)
        for slice_idx in range(num_slices):
            # Calculate slice boundaries
            slice_start = now - timedelta(days=(slice_idx + 1) * slice_duration_days)
            slice_end = now - timedelta(days=slice_idx * slice_duration_days)
            
            # For the first slice (most recent), cap at now
            if slice_idx == 0:
                slice_end = now
            
            logger.info(f"Fetching time slice {slice_idx + 1}/{num_slices}: {slice_start.strftime('%Y-%m-%d')} to {slice_end.strftime('%Y-%m-%d')} (target: {candidates_per_slice} candidates)")
            
            # Build date-range query so arXiv API returns papers from THIS slice
            # arXiv query syntax: submittedDate:[YYYYMMDDHHMI TO YYYYMMDDHHMI]
            date_start_str = slice_start.strftime("%Y%m%d0000")
            date_end_str = slice_end.strftime("%Y%m%d2359")
            date_range_query = f"submittedDate:[{date_start_str} TO {date_end_str}]"
            slice_query = f"({category_query}) AND {date_range_query}"

            slice_papers = []
            try:
                client = Client(
                    page_size=min(200, candidates_per_slice * 2),
                    delay_seconds=3,
                )
                search = arxiv.Search(
                    query=slice_query,
                    max_results=candidates_per_slice * 2,  # Overfetch then trim
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )

                for result in client.results(search):
                    if len(slice_papers) >= candidates_per_slice:
                        break

                    paper_id = result.entry_id.split('/')[-1]
                    # Extract DOI and journal_ref if available
                    doi = getattr(result, 'doi', None) or None
                    journal_ref = getattr(result, 'journal_ref', None) or None
                    paper = PaperData(
                        arxiv_id=paper_id,
                        title=result.title,
                        authors=[author.name for author in result.authors],
                        abstract=result.summary,
                        categories=result.categories,
                        published_date=result.published,
                        arxiv_url=result.entry_id,
                        pdf_url=result.pdf_url,
                        doi=doi,
                        journal_ref=journal_ref,
                    )
                    slice_papers.append(paper)
            except Exception as e:
                logger.warning(f"Error fetching time slice {slice_idx + 1}: {e}")
            
            papers.extend(slice_papers)
            logger.info(f"Collected {len(slice_papers)} papers for slice {slice_idx + 1}/{num_slices}")
        
        logger.info(f"Total collected from {num_slices} time slices: {len(papers)} papers")
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
                paper_id = result.entry_id.split('/')[-1]
                # Extract DOI and journal_ref if available
                doi = getattr(result, 'doi', None) or None
                journal_ref = getattr(result, 'journal_ref', None) or None
                paper = PaperData(
                    arxiv_id=paper_id,
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    categories=result.categories,
                    published_date=result.published,
                    arxiv_url=result.entry_id,
                    pdf_url=result.pdf_url,
                    doi=doi,
                    journal_ref=journal_ref,
                )
                papers.append(paper)
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
        
        return papers

