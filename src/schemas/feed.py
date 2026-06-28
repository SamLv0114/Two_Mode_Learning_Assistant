"""
Feed-related Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class FeedRequest(BaseModel):
    """Schema for requesting a daily feed"""
    time_window_days: int = Field(default=7, ge=1, le=365)
    focus_areas: Optional[List[str]] = None
    custom_interests: Optional[List[str]] = None
    use_ml: bool = True  # Whether to use ML ranking if available


class PaperResponse(BaseModel):
    """Schema for paper in feed response"""
    id: int
    rank: int
    arxiv_id: str
    title: str
    authors: Optional[str] = None
    abstract: Optional[str] = None
    categories: Optional[str] = None
    published_date: Optional[datetime] = None
    arxiv_url: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: int = 0
    relevance_score: float = 0.0
    impact_score: Optional[float] = None
    summary: Optional[str] = None

    class Config:
        from_attributes = True


class ArticleResponse(BaseModel):
    """Schema for article in feed response"""
    id: int
    rank: int
    source: str
    title: str
    url: str
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    upvotes: int = 0
    relevance_score: float = 0.0
    summary: Optional[str] = None

    class Config:
        from_attributes = True


class FeedResponse(BaseModel):
    """Schema for daily feed response"""
    papers: List[PaperResponse] = []
    articles: List[ArticleResponse] = []
    generated_at: datetime
    time_window_days: int
    focus_areas: List[str] = []
    used_ml_ranking: bool = False
    total_papers_considered: int = 0
    total_articles_considered: int = 0


class FeedHistoryItem(BaseModel):
    """Schema for a single feed history entry"""
    id: int
    generated_at: datetime
    time_window_days: int
    paper_count: int
    article_count: int


class FeedHistoryResponse(BaseModel):
    """Schema for paginated feed history"""
    items: List[FeedHistoryItem]
    total: int
    page: int
    per_page: int
    has_more: bool
