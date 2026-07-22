"""
Daily feed endpoints
"""
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from src.database.models import User, Paper, Article, UserInteraction
from src.api.deps import get_db_session, get_current_user, get_embedding_manager
from src.schemas.feed import (
    FeedRequest,
    FeedResponse,
    PaperResponse,
    ArticleResponse
)
from src.utils.config import settings
from src.models.embeddings import EmbeddingManager

router = APIRouter(prefix="/feed", tags=["Daily Feed"])
logger = logging.getLogger(__name__)

# ── Job status store (Redis-backed, in-memory fallback) ───────────────────────
_job_store: dict = {}
_JOB_TTL = 3600  # 1 hour


def _get_redis():
    if not settings.REDIS_URL:
        return None
    try:
        import redis
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


def _set_status(job_id: str, data: dict) -> None:
    rc = _get_redis()
    if rc:
        try:
            rc.setex(f"feed_job:{job_id}", _JOB_TTL, json.dumps(data))
            return
        except Exception:
            pass
    _job_store[job_id] = data


def _get_status(job_id: str) -> dict:
    rc = _get_redis()
    if rc:
        try:
            raw = rc.get(f"feed_job:{job_id}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    return _job_store.get(job_id, {"status": "not_found"})


# ── Background pipeline runner ────────────────────────────────────────────────

def _run_pipeline_background(
    job_id: str,
    user_id: int,
    time_window_days: int,
    focus_areas: List[str],
    user_interests: List[str],
    mode: str = "recommended",
) -> None:
    """Runs the full feed pipeline in a background thread with its own DB session."""
    from src.database.models import SessionLocal, User as UserModel
    from src.api.deps import get_embedding_manager

    try:
        _set_status(job_id, {"status": "collecting", "message": "Collecting papers and articles..."})

        db = SessionLocal()
        try:
            embedding_manager = get_embedding_manager()
            from src.pipelines.daily_feed import DailyFeedPipeline

            _set_status(job_id, {"status": "ranking", "message": "Ranking and summarizing content..."})

            pipeline = DailyFeedPipeline(
                user_id=user_id,
                db_session=db,
                embedding_manager=embedding_manager,
            )
            result = pipeline.run(
                time_window_days=time_window_days,
                focus_areas=focus_areas,
                user_interests=user_interests,
                mode=mode,
            )

            _set_status(job_id, {
                "status": "done",
                "message": "Feed ready",
                "papers_count": len(result.get("papers", [])),
                "articles_count": len(result.get("articles", [])),
                "used_ml_ranking": result.get("used_ml_ranking", False),
            })
            logger.info(f"Feed job {job_id} completed for user {user_id}")
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Feed job {job_id} failed: {e}")
        _set_status(job_id, {"status": "error", "message": str(e)})


@router.post("/generate")
async def generate_feed(
    request: FeedRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager)
):
    """
    Start personalized feed generation in the background.

    Returns a job_id immediately. Poll GET /feed/status/{job_id} to track
    progress, then fetch results from GET /feed/papers and GET /feed/articles.

    - **time_window_days**: How far back to look for content (1-365)
    - **focus_areas**: Optional list of focus areas to prioritize
    - **custom_interests**: Optional additional interest keywords
    """
    user_interests = current_user.get_interests_list()
    if request.custom_interests:
        user_interests = user_interests + request.custom_interests
    focus_areas = request.focus_areas or current_user.get_focus_areas_list()

    job_id = str(uuid.uuid4())
    _set_status(job_id, {"status": "generating", "message": "Starting feed generation..."})

    background_tasks.add_task(
        _run_pipeline_background,
        job_id=job_id,
        user_id=current_user.id,
        time_window_days=request.time_window_days,
        focus_areas=focus_areas or user_interests,
        user_interests=user_interests,
        mode=request.mode,
    )

    return {"job_id": job_id, "status": "generating", "message": "Feed generation started"}


@router.get("/status/{job_id}")
async def feed_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Poll the status of a feed generation job.

    Status values:
    - **generating** — pipeline is starting up
    - **collecting** — fetching papers and articles
    - **ranking**    — ML ranking and LLM summarization in progress
    - **done**       — complete, fetch results from /feed/papers
    - **error**      — pipeline failed (message contains reason)
    - **not_found**  — job_id unknown or expired (TTL: 1 hour)
    """
    return _get_status(job_id)


@router.get("/papers", response_model=List[PaperResponse])
async def get_recommended_papers(
    limit: int = 10,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get previously recommended papers

    Returns papers that were marked as recommended.
    """
    papers = db.query(Paper).filter(
        Paper.recommended == True
    ).order_by(
        Paper.recommended_date.desc()
    ).offset(offset).limit(limit).all()

    result = []
    for i, paper in enumerate(papers, offset + 1):
        result.append(PaperResponse(
            id=paper.id,
            rank=i,
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            categories=paper.categories,
            published_date=paper.published_date,
            arxiv_url=paper.arxiv_url,
            pdf_url=paper.pdf_url,
            citation_count=paper.citation_count,
            relevance_score=paper.relevance_score,
            impact_score=paper.heuristic_impact_score,
            summary=paper.personalized_summary
        ))

    return result


@router.get("/articles", response_model=List[ArticleResponse])
async def get_recommended_articles(
    limit: int = 10,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get previously recommended articles

    Returns articles that were marked as recommended.
    """
    articles = db.query(Article).filter(
        Article.recommended == True
    ).order_by(
        Article.recommended_date.desc()
    ).offset(offset).limit(limit).all()

    result = []
    for i, article in enumerate(articles, offset + 1):
        result.append(ArticleResponse(
            id=article.id,
            rank=i,
            source=article.source,
            title=article.title,
            url=article.url,
            author=article.author,
            published_date=article.published_date,
            upvotes=article.upvotes,
            relevance_score=article.relevance_score,
            summary=article.personalized_summary
        ))

    return result


@router.get("/saved", response_model=dict)
async def get_saved_items(
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get items the user has saved
    """
    # Get saved interactions
    saved_interactions = db.query(UserInteraction).filter(
        UserInteraction.user_id == current_user.id,
        UserInteraction.interaction_type == "saved"
    ).order_by(UserInteraction.timestamp.desc()).offset(offset).limit(limit).all()

    papers = []
    articles = []

    for interaction in saved_interactions:
        if interaction.item_type == "paper":
            paper = db.query(Paper).filter(Paper.id == interaction.item_id).first()
            if paper:
                papers.append(PaperResponse(
                    id=paper.id,
                    rank=len(papers) + 1,
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    authors=paper.authors,
                    abstract=paper.abstract,
                    categories=paper.categories,
                    published_date=paper.published_date,
                    arxiv_url=paper.arxiv_url,
                    pdf_url=paper.pdf_url,
                    citation_count=paper.citation_count,
                    relevance_score=paper.relevance_score,
                    impact_score=paper.heuristic_impact_score,
                    summary=paper.personalized_summary
                ))
        else:
            article = db.query(Article).filter(Article.id == interaction.item_id).first()
            if article:
                articles.append(ArticleResponse(
                    id=article.id,
                    rank=len(articles) + 1,
                    source=article.source,
                    title=article.title,
                    url=article.url,
                    author=article.author,
                    published_date=article.published_date,
                    upvotes=article.upvotes,
                    relevance_score=article.relevance_score,
                    summary=article.personalized_summary
                ))

    return {
        "papers": papers,
        "articles": articles,
        "total_saved": len(saved_interactions)
    }
