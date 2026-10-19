"""
The shared demo account behind the landing page's "Try the demo".

Every visitor signs in as the same user, so two kinds of endpoint need a guard:

- forbid_demo: actions that would change the account for everyone after them
  (profile, password, deleting the account, uploads) or bust global caches.
- demo_quota: actions that start model work. Each one counts against a per-visitor
  and a whole-demo daily allowance, so the demo's cost has a fixed ceiling no
  matter how it is used.

Both are no-ops for every other user, and for everyone when DEMO_ENABLED is off.
"""
import logging
import threading
from datetime import datetime, timezone
from typing import Dict

from fastapi import Depends, HTTPException, Request, status

from src.api.deps import get_current_user
from src.api.middleware import client_ip
from src.database.models import User
from src.utils.config import settings
from src.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

FORBIDDEN_DETAIL = "Not available in the shared demo account. Create a free account to do this."
EXHAUSTED_DETAIL = ("The demo's AI actions for today are used up. "
                    "Create a free account to keep going — it takes a minute.")

_memory_counts: Dict[str, int] = {}
_memory_lock = threading.Lock()


def is_demo_user(user) -> bool:
    return bool(settings.DEMO_ENABLED and getattr(user, "email", None) == settings.DEMO_USER_EMAIL)


def get_demo_user(db) -> User | None:
    if not settings.DEMO_ENABLED:
        return None
    return db.query(User).filter(User.email == settings.DEMO_USER_EMAIL, User.is_active.is_(True)).first()


def forbid_demo(current_user: User = Depends(get_current_user)) -> User:
    if is_demo_user(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=FORBIDDEN_DETAIL)
    return current_user


def _increment(key: str) -> int:
    """Today's count for key, after adding one. Redis when available, else this process."""
    client = get_redis()
    if client is not None:
        try:
            count = int(client.incr(key))
            if count == 1:
                client.expire(key, 2 * 86400)
            return count
        except Exception as exc:
            logger.warning("Demo quota counter unavailable, counting in memory: %s", exc)
    with _memory_lock:
        if len(_memory_counts) > 10_000:  # yesterday's keys; today's are recreated on demand
            _memory_counts.clear()
        _memory_counts[key] = _memory_counts.get(key, 0) + 1
        return _memory_counts[key]


def charge_demo(request: Request, user) -> None:
    """Count one model-backed action against the demo's allowances; 429 once they are spent."""
    if not is_demo_user(user):
        return
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # The visitor allowance is checked first, so one visitor past theirs does not use up everyone's.
    if _increment(f"demo_quota:{day}:visitor:{client_ip(request)}") > settings.DEMO_ACTIONS_PER_VISITOR_PER_DAY:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=EXHAUSTED_DETAIL)
    if _increment(f"demo_quota:{day}:total") > settings.DEMO_ACTIONS_TOTAL_PER_DAY:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=EXHAUSTED_DETAIL)


def demo_quota(request: Request, current_user: User = Depends(get_current_user)) -> User:
    """Dependency form of charge_demo for endpoints whose every call starts model work."""
    charge_demo(request, current_user)
    return current_user
