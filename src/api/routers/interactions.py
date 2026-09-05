"""
User interaction endpoints
"""
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database.models import User, UserInteraction, UserModelState, Paper, Article
from src.api.deps import get_db_session, get_current_user

logger = logging.getLogger(__name__)
from src.schemas.interaction import (
    InteractionCreate,
    InteractionResponse,
    InteractionStats,
    InteractionList,
    ModelStatus,
    RetrainResponse
)
from src.utils.config import settings

router = APIRouter(prefix="/interactions", tags=["Interactions"])


# ── V4: Dismiss Reflexion ─────────────────────────────────────────────────────

def _extract_dismiss_reason_and_store(user_id: int, paper_title: str, paper_abstract: str) -> None:
    """
    Background task: use gpt-4o-mini to infer why a user dismissed a paper,
    then store the result as a negative fact in UserFactMemory.

    This implements the 'Dismiss Reflexion' pattern — the system learns not just
    that the user dismissed something, but *why*, and uses that signal to avoid
    similar content in future feeds.
    """
    from openai import OpenAI
    from src.agents.memory import UserFactMemory
    from src.utils.config import settings

    if not settings.OPENAI_API_KEY:
        return

    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        prompt = (
            f"A researcher dismissed this paper from their feed. "
            f"In 5-10 words, why might they have dismissed it?\n\n"
            f"Title: {paper_title}\n"
            f"Abstract: {(paper_abstract or '')[:300]}\n\n"
            f"Return only the reason phrase, e.g. 'too theoretical, no practical applications'"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            temperature=0.3,
        )
        reason = resp.choices[0].message.content.strip().strip('"').strip("'")
        if reason:
            # Get Redis client
            redis_client = None
            try:
                import redis
                if settings.REDIS_URL:
                    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                    redis_client.ping()
            except Exception:
                redis_client = None

            UserFactMemory().add_negative_fact(user_id, reason, redis_client)
            logger.info(f"Dismiss reflexion stored for user {user_id}: {reason[:60]}")
    except Exception as e:
        logger.debug(f"Dismiss reflexion failed: {e}")


@router.post("", response_model=InteractionResponse, status_code=status.HTTP_201_CREATED)
async def create_interaction(
    interaction: InteractionCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Record a user interaction with a paper or article

    - **item_type**: "paper" or "article"
    - **item_id**: Database ID of the item
    - **interaction_type**: "viewed", "saved", or "dismissed"

    If an interaction already exists for this item, it will be updated.
    """
    # Verify the item exists
    if interaction.item_type == "paper":
        item = db.query(Paper).filter(Paper.id == interaction.item_id).first()
    else:
        item = db.query(Article).filter(Article.id == interaction.item_id).first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{interaction.item_type.capitalize()} not found"
        )

    # Check for existing interaction
    existing = db.query(UserInteraction).filter(
        UserInteraction.user_id == current_user.id,
        UserInteraction.item_type == interaction.item_type,
        UserInteraction.item_id == interaction.item_id
    ).first()

    if existing:
        existing.interaction_type = interaction.interaction_type
        existing.timestamp = datetime.now(timezone.utc)
        db.commit()
        db.refresh(existing)
        result = InteractionResponse.model_validate(existing)
    else:
        new_interaction = UserInteraction(
            user_id=current_user.id,
            item_type=interaction.item_type,
            item_id=interaction.item_id,
            interaction_type=interaction.interaction_type
        )
        db.add(new_interaction)
        db.commit()
        db.refresh(new_interaction)
        result = InteractionResponse.model_validate(new_interaction)

    # V4: Dismiss Reflexion — infer why user dismissed a paper and store as negative fact
    if interaction.interaction_type == "dismissed" and interaction.item_type == "paper":
        background_tasks.add_task(
            _extract_dismiss_reason_and_store,
            user_id=current_user.id,
            paper_title=item.title,
            paper_abstract=getattr(item, "abstract", "") or "",
        )

    return result


@router.get("", response_model=InteractionList)
async def list_interactions(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    item_type: str = Query(default=None, pattern="^(paper|article)$"),
    interaction_type: str = Query(default=None, pattern="^(viewed|saved|dismissed)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    List user interactions with pagination and optional filtering
    """
    query = db.query(UserInteraction).filter(
        UserInteraction.user_id == current_user.id
    )

    if item_type:
        query = query.filter(UserInteraction.item_type == item_type)
    if interaction_type:
        query = query.filter(UserInteraction.interaction_type == interaction_type)

    # Get total count
    total = query.count()

    # Apply pagination
    offset = (page - 1) * per_page
    interactions = query.order_by(UserInteraction.timestamp.desc()).offset(offset).limit(per_page).all()

    return InteractionList(
        items=[InteractionResponse.model_validate(i) for i in interactions],
        total=total,
        page=page,
        per_page=per_page,
        has_more=(offset + len(interactions)) < total
    )


@router.get("/stats", response_model=InteractionStats)
async def get_interaction_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get statistics about user interactions
    """
    # Count by interaction type
    stats = db.query(
        UserInteraction.interaction_type,
        func.count(UserInteraction.id)
    ).filter(
        UserInteraction.user_id == current_user.id
    ).group_by(
        UserInteraction.interaction_type
    ).all()

    stats_dict = {stat[0]: stat[1] for stat in stats}
    total = sum(stats_dict.values())

    min_required = settings.MIN_INTERACTIONS_FOR_TRAINING
    ready_for_training = total >= min_required

    return InteractionStats(
        total=total,
        saved=stats_dict.get("saved", 0),
        viewed=stats_dict.get("viewed", 0),
        dismissed=stats_dict.get("dismissed", 0),
        ready_for_training=ready_for_training,
        interactions_until_training=max(0, min_required - total)
    )


@router.delete("/{interaction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_interaction(
    interaction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Delete a specific interaction
    """
    interaction = db.query(UserInteraction).filter(
        UserInteraction.id == interaction_id,
        UserInteraction.user_id == current_user.id
    ).first()

    if not interaction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interaction not found"
        )

    db.delete(interaction)
    db.commit()
    return None


@router.get("/model/status", response_model=ModelStatus)
async def get_model_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get the status of the user's personalized ML model
    """
    # Get interaction count
    interaction_count = db.query(UserInteraction).filter(
        UserInteraction.user_id == current_user.id
    ).count()

    # Get model state
    model_state = db.query(UserModelState).filter(
        UserModelState.user_id == current_user.id
    ).first()

    if model_state:
        return ModelStatus(
            is_trained=model_state.is_trained,
            last_trained_at=model_state.last_trained_at,
            interaction_count=interaction_count,
            min_interactions_required=settings.MIN_INTERACTIONS_FOR_TRAINING,
            train_ndcg=model_state.train_ndcg,
            train_mrr=model_state.train_mrr,
            val_ndcg=model_state.val_ndcg,
            val_mrr=model_state.val_mrr
        )
    else:
        return ModelStatus(
            is_trained=False,
            interaction_count=interaction_count,
            min_interactions_required=settings.MIN_INTERACTIONS_FOR_TRAINING
        )


@router.post("/model/retrain", response_model=RetrainResponse)
async def retrain_model(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Trigger retraining of the user's personalized ML model

    Requires at least 50 interactions.
    """
    # Check interaction count
    interaction_count = db.query(UserInteraction).filter(
        UserInteraction.user_id == current_user.id
    ).count()

    min_required = settings.MIN_INTERACTIONS_FOR_TRAINING
    if interaction_count < min_required:
        return RetrainResponse(
            success=False,
            message=f"Not enough interactions. Need {min_required - interaction_count} more."
        )

    try:
        # Import here to avoid circular dependencies
        from src.models.user_recommender import UserRecommender
        from src.models.user_trainer import UserModelTrainer
        from src.api.deps import get_embedding_manager

        embedding_manager = get_embedding_manager()
        trainer = UserModelTrainer(current_user.id, db, embedding_manager)
        recommender = UserRecommender(current_user.id, db)

        success = trainer.retrain_model(recommender, min_interactions=min_required)

        if success:
            # Get updated metrics
            model_state = db.query(UserModelState).filter(
                UserModelState.user_id == current_user.id
            ).first()

            metrics = None
            if model_state:
                metrics = {
                    "train_ndcg": model_state.train_ndcg,
                    "train_mrr": model_state.train_mrr,
                    "val_ndcg": model_state.val_ndcg,
                    "val_mrr": model_state.val_mrr
                }

            return RetrainResponse(
                success=True,
                message=f"Model retrained successfully with {interaction_count} interactions",
                metrics=metrics
            )
        else:
            return RetrainResponse(
                success=False,
                message="Training failed. Check server logs for details."
            )

    except Exception as e:
        return RetrainResponse(
            success=False,
            message=f"Training error: {str(e)}"
        )
