"""
Feed-related Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List
from datetime import datetime


class FeedRequest(BaseModel):
    """Schema for requesting a daily feed"""
    time_window_days: int = Field(default=7, ge=1, le=365)
    focus_areas: Optional[List[str]] = None
    custom_interests: Optional[List[str]] = None
    use_ml: bool = True  # Whether to use ML ranking if available
    mode: str = Field(default="recommended", pattern="^(recommended|latest)$")
    # Generation is idempotent per day. Set this to explicitly request a
    # different batch, which consumes the novelty window.
    force_refresh: bool = False


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
    # Why this paper is in the feed: the model's one-line reason, its judgement, and the
    # recall channels that found it (src/pipelines/feed/judge.py). None for older feeds.
    reasons: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True
