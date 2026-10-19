"""
Unauthenticated facts for the landing page: corpus size and whether the demo is open.

Read from the database rather than written into the page, so the numbers a visitor
sees are the real ones. Cached briefly in-process; the counts move once a night.
"""
import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.demo import get_demo_user
from src.api.deps import get_db_session
from src.database.models import Paper

router = APIRouter(prefix="/public", tags=["Public"])

CACHE_SECONDS = 600
_cache: dict = {"at": 0.0, "value": None}


@router.get("/stats")
def public_stats(db: Session = Depends(get_db_session)):
    if _cache["value"] is None or time.monotonic() - _cache["at"] > CACHE_SECONDS:
        _cache["value"] = {
            "papers_indexed": db.query(Paper).count(),
            "demo_available": get_demo_user(db) is not None,
        }
        _cache["at"] = time.monotonic()
    return _cache["value"]
