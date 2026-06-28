"""
Daily feed endpoints
"""
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


@router.post("/generate", response_model=FeedResponse)
async def generate_feed(
    request: FeedRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager)
):
    """
    Generate a personalized daily feed of papers and articles

    - **time_window_days**: How far back to look for content (1-365)
    - **focus_areas**: Optional list of focus areas to prioritize
    - **custom_interests**: Optional additional interest keywords
    - **use_ml**: Whether to use ML ranking (if model is trained)
    """
    try:
        # Get user interests
        user_interests = current_user.get_interests_list()
        if request.custom_interests:
            user_interests = user_interests + request.custom_interests

        focus_areas = request.focus_areas or current_user.get_focus_areas_list()

        # Import pipeline
        from src.pipelines.daily_feed import DailyFeedPipeline

        # Create per-user pipeline
        pipeline = DailyFeedPipeline(
            user_id=current_user.id,
            db_session=db,
            embedding_manager=embedding_manager,
        )

        # Run pipeline
        result = pipeline.run(
            time_window_days=request.time_window_days,
            focus_areas=focus_areas or user_interests,
            user_interests=user_interests,
        )

        # Format response
        papers = []
        for i, paper_data in enumerate(result.get("papers", []), 1):
            # Handle None values - .get() returns None when key exists with None value
            impact = paper_data.get("impact_score")
            if isinstance(impact, str):
                try:
                    impact = float(impact)
                except (ValueError, TypeError):
                    impact = None

            papers.append(PaperResponse(
                id=paper_data.get("db_id") or 0,
                rank=i,
                arxiv_id=paper_data.get("arxiv_id") or "",
                title=paper_data.get("title") or "",
                authors=paper_data.get("authors"),
                abstract=paper_data.get("abstract"),
                categories=paper_data.get("categories"),
                published_date=paper_data.get("published_date"),
                arxiv_url=paper_data.get("url"),
                pdf_url=paper_data.get("pdf_url"),
                citation_count=paper_data.get("citation_count") or 0,
                relevance_score=paper_data.get("relevance_score") or 0.0,
                impact_score=impact,
                summary=paper_data.get("summary")
            ))

        articles = []
        for i, article_data in enumerate(result.get("articles", []), 1):
            articles.append(ArticleResponse(
                id=article_data.get("db_id") or 0,
                rank=i,
                source=article_data.get("source") or "",
                title=article_data.get("title") or "",
                url=article_data.get("url") or "",
                author=article_data.get("author"),
                published_date=article_data.get("published_date"),
                upvotes=article_data.get("upvotes") or 0,
                relevance_score=article_data.get("relevance_score") or 0.0,
                summary=article_data.get("summary")
            ))

        return FeedResponse(
            papers=papers,
            articles=articles,
            generated_at=datetime.now(timezone.utc),
            time_window_days=request.time_window_days,
            focus_areas=focus_areas or [],
            used_ml_ranking=result.get("used_ml_ranking", False),
            total_papers_considered=result.get("total_papers_considered", 0),
            total_articles_considered=result.get("total_articles_considered", 0)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate feed: {str(e)}"
        )


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
