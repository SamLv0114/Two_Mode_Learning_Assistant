"""
Daily feed endpoints
"""
import json
import uuid
import logging
import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from src.database.models import User, Paper, UserInteraction, UserPaperRecommendation, count_labeled_paper_interactions
from src.api.deps import get_db_session, get_current_user, get_embedding_manager
from src.schemas.feed import (
    FeedRequest,
    PaperResponse,
)
from src.utils.config import settings
from src.models.embeddings import EmbeddingManager
from src.pipelines.daily_feed import DailyFeedPipeline, _describe_novelty

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


def _status_key(user_id: int, job_id: str) -> str:
    return f"feed_job:{user_id}:{job_id}"


def _set_status(user_id: int, job_id: str, data: dict) -> None:
    key = _status_key(user_id, job_id)
    rc = _get_redis()
    if rc:
        try:
            rc.setex(key, _JOB_TTL, json.dumps(data))
            return
        except Exception:
            pass
    _job_store[key] = (time.monotonic() + _JOB_TTL, data)
    if len(_job_store) > 1000:
        now = time.monotonic()
        for stale_key, (expires, _) in list(_job_store.items()):
            if expires <= now:
                _job_store.pop(stale_key, None)


def _get_status(user_id: int, job_id: str) -> dict:
    key = _status_key(user_id, job_id)
    rc = _get_redis()
    if rc:
        try:
            raw = rc.get(key)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    entry = _job_store.get(key)
    if entry and entry[0] > time.monotonic():
        return entry[1]
    _job_store.pop(key, None)
    return {"status": "not_found"}


# ── Background pipeline runner ────────────────────────────────────────────────

def _run_pipeline_background(
    job_id: str,
    user_id: int,
    time_window_days: int,
    focus_areas: List[str],
    user_interests: List[str],
    mode: str = "recommended",
    force_refresh: bool = False,
    use_ml: bool = True,
) -> None:
    """Runs the full feed pipeline in a background thread with its own DB session."""
    from src.database.models import SessionLocal
    from src.api.deps import get_embedding_manager

    try:
        _set_status(user_id, job_id, {"status": "collecting", "message": "Collecting papers..."})

        db = SessionLocal()
        try:
            embedding_manager = get_embedding_manager()

            _set_status(user_id, job_id, {"status": "ranking", "message": "Ranking and summarizing content..."})

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
                force_refresh=force_refresh,
                use_ml=use_ml,
            )

            _set_status(user_id, job_id, {
                "status": "done",
                "message": "Showing today's feed" if result.get("reused") else "Feed ready",
                "reused": bool(result.get("reused")),
                "papers_count": len(result.get("papers", [])),
                "articles_count": len(result.get("articles", [])),
                "used_ml_ranking": result.get("used_ml_ranking", False),
                "ranking_source": result.get("ranking_source", "heuristic"),
                "ranking_fallback_reason": result.get("ranking_fallback_reason"),
            })
            logger.info(f"Feed job {job_id} completed for user {user_id}")
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Feed job {job_id} failed: {e}")
        _set_status(user_id, job_id, {"status": "error", "message": "Feed generation failed. Please retry."})


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
    progress, then fetch results from GET /feed/papers.

    Generation is idempotent per day: calling this again returns the feed already
    built today instead of replacing it. Pass force_refresh to deliberately draw a
    different batch, which consumes the novelty window.

    - **time_window_days**: How far back to look for content (1-365)
    - **focus_areas**: Optional list of focus areas to prioritize
    - **custom_interests**: Optional additional interest keywords
    - **force_refresh**: Draw a new batch instead of reusing today's
    """
    user_interests = current_user.get_interests_list()
    if request.custom_interests:
        user_interests = user_interests + request.custom_interests
    focus_areas = request.focus_areas or current_user.get_focus_areas_list()

    job_id = str(uuid.uuid4())
    _set_status(current_user.id, job_id, {"status": "generating", "message": "Starting feed generation..."})

    background_tasks.add_task(
        _run_pipeline_background,
        job_id=job_id,
        user_id=current_user.id,
        time_window_days=request.time_window_days,
        focus_areas=focus_areas or user_interests,
        user_interests=user_interests,
        mode=request.mode,
        force_refresh=request.force_refresh,
        use_ml=request.use_ml,
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
    - **collecting** — fetching papers
    - **ranking**    — ML ranking and LLM summarization in progress
    - **done**       — complete, fetch results from /feed/papers
    - **error**      — pipeline failed (generic message; details in server logs)
    - **not_found**  — job_id unknown or expired (TTL: 1 hour)
    """
    return _get_status(current_user.id, job_id)


@router.get("/coldstart")
async def get_coldstart_seeds(
    count: int = 12,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
):
    """
    Diverse, high-impact papers for a new user to react to during onboarding.

    Rating a few of these through POST /interactions bootstraps the profile far
    faster than waiting for organic interactions: the same table feeds the LLM
    profile query, the novelty dial, both ranking penalties, and eventually
    LightGBM. `interactions_recorded` lets the client decide whether onboarding
    is still warranted.
    """
    recorded = count_labeled_paper_interactions(db, current_user.id)

    pipeline = DailyFeedPipeline(
        user_id=current_user.id,
        db_session=db,
        embedding_manager=embedding_manager,
    )
    seeds = pipeline.get_coldstart_seeds(n=max(1, min(count, 30)))

    return {
        "seeds": seeds,
        "count": len(seeds),
        "interactions_recorded": recorded,
        "onboarding_recommended": recorded < 5 and len(seeds) > 0,
    }


@router.get("/papers", response_model=List[PaperResponse])
async def get_recommended_papers(
    limit: int = settings.TOP_PAPERS_COUNT,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Get this user's most recently generated paper feed, ordered by rank."""
    rows = (
        db.query(Paper, UserPaperRecommendation)
        .join(UserPaperRecommendation, Paper.id == UserPaperRecommendation.paper_id)
        .filter(UserPaperRecommendation.user_id == current_user.id)
        .order_by(UserPaperRecommendation.rank.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return [
        PaperResponse(
            id=paper.id,
            rank=rec.rank or (offset + i),
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            categories=paper.categories,
            published_date=paper.published_date,
            arxiv_url=paper.arxiv_url,
            pdf_url=paper.pdf_url,
            citation_count=paper.citation_count,
            relevance_score=rec.relevance_score,
            impact_score=paper.heuristic_impact_score,
            summary=rec.personalized_summary,
        )
        for i, (paper, rec) in enumerate(rows, offset + 1)
    ]


def _materialize_refined_papers(db: Session, items: list[dict]) -> list[PaperResponse]:
    """Restore full paper fields after the optimizer swaps lightweight entries."""
    paper_by_arxiv = {
        paper.arxiv_id: paper for paper in db.query(Paper).filter(
            Paper.arxiv_id.in_([item["arxiv_id"] for item in items])
        ).all()
    }
    refined_papers = []
    for rank, item in enumerate(items, 1):
        paper = paper_by_arxiv.get(item["arxiv_id"])
        if paper is None:
            raise HTTPException(status_code=500, detail="Refined paper is no longer available")
        refined_papers.append(PaperResponse(
            id=paper.id,
            rank=rank,
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            categories=paper.categories,
            published_date=paper.published_date,
            arxiv_url=paper.arxiv_url,
            pdf_url=paper.pdf_url,
            citation_count=paper.citation_count or 0,
            relevance_score=float(item.get("relevance_score") or 0.0),
            impact_score=paper.heuristic_impact_score,
            summary=item.get("summary"),
        ))
    return refined_papers


@router.post("/refine")
async def refine_feed(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
):
    """
    Evaluator-Optimizer refinement pass on the current feed.

    An LLM scores the feed and names the weak entries; those get replaced with
    fresh candidates retrieved from the corpus, and the revised list is re-scored.
    Up to 3 rounds, stopping early once the score clears 0.70 or there is nothing
    left to swap in.

    Reads feed_ctx:{user_id} from Redis (written by the last /generate run) for the
    reader profile, and returns the best-scoring version along with its score.

    Returns 404 only when there is no feed to refine.
    """
    from src.database.models import Paper, UserPaperRecommendation

    # The feed itself is the hard requirement — without papers there is nothing to
    # score. Load it first so a missing context never masks a present feed.
    rows = (
        db.query(Paper, UserPaperRecommendation)
        .join(UserPaperRecommendation, Paper.id == UserPaperRecommendation.paper_id)
        .filter(UserPaperRecommendation.user_id == current_user.id)
        .order_by(UserPaperRecommendation.rank.asc())
        # Refine operates on the whole feed, so this tracks the feed size rather
        # than a literal that happens to exceed it today.
        .limit(settings.TOP_PAPERS_COUNT)
        .all()
    )

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No feed to refine. Generate a feed first.",
        )

    # The reading profile is a nicety: it becomes one line of the evaluator prompt.
    # feed_ctx holds it but expires after 2h, so a feed generated this morning
    # would otherwise stop being refinable by the afternoon even though it is still
    # on screen. Recompute it when the cache has gone rather than refusing to run.
    DEFAULT_PROFILE = "balances established literature and recent work"
    user_mode = None

    rc = _get_redis()
    if rc:
        try:
            ctx_raw = rc.get(f"feed_ctx:{current_user.id}")
            if ctx_raw:
                user_mode = json.loads(ctx_raw).get("reading_profile")
        except Exception:
            pass

    if not user_mode:
        try:
            probe = DailyFeedPipeline(
                user_id=current_user.id,
                db_session=db,
                embedding_manager=embedding_manager,
            )
            user_mode = _describe_novelty(probe._infer_novelty_preference())
            logger.info(f"feed_ctx missing — recomputed reading profile: {user_mode}")
        except Exception as e:
            logger.debug(f"Could not recompute reading profile ({e}) — using default")
            user_mode = DEFAULT_PROFILE

    user_interests = current_user.get_interests_list()

    papers_input = [
        {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "relevance_score": rec.relevance_score or 0.0,
            "citation_count": paper.citation_count,
            "url": paper.arxiv_url,
            "summary": rec.personalized_summary,
            "rank": rec.rank,
        }
        for paper, rec in rows
    ]

    # Replacements are retrieved from the corpus, not from the user's own
    # recommendation rows — those rows are the feed being refined, so searching
    # them for alternatives always came back empty.
    pipeline = DailyFeedPipeline(
        user_id=current_user.id,
        db_session=db,
        embedding_manager=embedding_manager,
    )
    focus_areas = current_user.get_focus_areas_list() or user_interests

    def _provide_candidates(exclude_ids: set, limit: int):
        return pipeline.get_alternate_candidates(
            exclude_arxiv_ids=exclude_ids,
            selected_interests=focus_areas,
            limit=limit,
        )

    from src.agents.iterative_refinement_agent import IterativeRefinementAgent
    agent = IterativeRefinementAgent()
    result = agent.refine(
        papers=papers_input,
        user_mode=user_mode,
        user_interests=user_interests,
        user_id=current_user.id,
        db_session=db,
        candidate_provider=_provide_candidates,
    )

    # The optimizer works with lightweight dictionaries. Restore the same paper
    # schema as GET /feed/papers so cards remain usable after a replacement.
    refined_papers = _materialize_refined_papers(db, result["papers"])

    return {
        "papers": refined_papers,
        "score": result["score"],
        "rounds": result["rounds"],
        "passed": result["passed"],
        "message": (
            f"Feed refined in {result['rounds']} round(s). "
            f"Quality score: {result['score']:.2f} "
            f"({'PASS' if result['passed'] else 'best effort'})."
        ),
    }


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
    total_saved = db.query(UserInteraction).filter(
        UserInteraction.user_id == current_user.id,
        UserInteraction.interaction_type == "saved"
    ).count()

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
                rec = db.query(UserPaperRecommendation).filter(
                    UserPaperRecommendation.user_id == current_user.id,
                    UserPaperRecommendation.paper_id == paper.id,
                ).first()
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
                    relevance_score=rec.relevance_score if rec else 0.0,
                    impact_score=paper.heuristic_impact_score,
                    summary=rec.personalized_summary if rec else None,
                ))

    return {
        "papers": papers,
        "articles": articles,
        "total_saved": total_saved
    }
