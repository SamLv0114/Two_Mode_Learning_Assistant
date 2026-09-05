"""
What's Hot endpoint — community trending content (not personalised).

GET /whats_hot
  Returns HuggingFace trending papers + GitHub trending ML repos,
  each with a 2-3 sentence digest and source citation.
  Cached 6h in Redis (shared across all users).

POST /whats_hot/refresh
  Admin/manual cache bust — re-fetches immediately.
"""
import logging
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db_session
from src.database.models import User, UserInteraction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whats_hot", tags=["What's Hot"])


def _get_redis():
    from src.utils.config import settings
    if not settings.REDIS_URL:
        return None
    try:
        import redis
        rc = redis.from_url(settings.REDIS_URL, decode_responses=True)
        rc.ping()
        return rc
    except Exception:
        return None


def _infer_last_visit_days(user_id: int, db: Session) -> int:
    """Estimate days since user last visited from most recent interaction."""
    latest = (
        db.query(UserInteraction.timestamp)
        .filter(UserInteraction.user_id == user_id)
        .order_by(UserInteraction.timestamp.desc())
        .first()
    )
    if not latest or not latest[0]:
        return 7  # default → weekly window
    ts = latest[0]
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - ts).days)


@router.get("")
async def get_whats_hot(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    Return What's Hot: HuggingFace trending papers + GitHub trending ML repos.

    - HuggingFace: community-voted ML paper hotness (digest included)
    - GitHub: trending ML repositories this week (or day if visited recently)
    - Cached 6h in Redis, shared across all users
    """
    rc = _get_redis()
    last_visit_days = _infer_last_visit_days(current_user.id, db)

    from src.agents.hot_news_collector import HotNewsCollector
    result = HotNewsCollector().collect(
        user_last_visit_days=last_visit_days,
        redis_client=rc,
    )
    return result


@router.post("/refresh")
async def refresh_whats_hot(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    Manually bust the What's Hot cache and re-fetch immediately.
    Useful when the user clicks the refresh button in the UI.
    """
    rc = _get_redis()

    # Clear today's cache entry
    if rc:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            rc.delete(f"whats_hot:{today}")
        except Exception:
            pass

    last_visit_days = _infer_last_visit_days(current_user.id, db)
    from src.agents.hot_news_collector import HotNewsCollector
    result = HotNewsCollector().collect(
        user_last_visit_days=last_visit_days,
        redis_client=rc,
    )
    return result
