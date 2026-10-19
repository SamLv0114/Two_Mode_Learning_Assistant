"""Paper record passed between collectors, the feed pipeline and import jobs."""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


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
    reasons: Optional[Dict[str, Any]] = None  # why it is in this user's feed (feed/judge.py)

