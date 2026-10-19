"""
One Redis client per process.

redis-py clients are thread-safe and pool their connections, so every caller
shares this one instead of opening a new pool per request. get_redis() still
pings before handing the client out, so callers keep the contract they were
written against: a working client, or None when Redis is unset or down (every
Redis use in this app degrades gracefully without it).
"""
import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

from src.utils.config import settings

logger = logging.getLogger(__name__)

# After a failed connection, report Redis as down for this long instead of
# making every request wait out its own connect timeout.
RETRY_AFTER_SECONDS = 30

_lock = threading.Lock()
_client = None
_down_since = 0.0


def get_redis():
    """The shared client (decode_responses=True), or None when Redis is unset or unreachable."""
    global _client, _down_since
    if not settings.REDIS_URL:
        return None
    if _down_since and time.monotonic() - _down_since < RETRY_AFTER_SECONDS:
        return None
    try:
        with _lock:
            if _client is None:
                import redis
                _client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=5,
                )
        _client.ping()
    except Exception as exc:
        if not _down_since:
            logger.warning("Redis unavailable, continuing without it: %s", exc)
        _down_since = time.monotonic()
        return None
    if _down_since:
        logger.info("Redis reachable again")
        _down_since = 0.0
    return _client


class JsonCache:
    """
    JSON values that expire: in Redis when it is up, else in this process.

    For data that is cheap to lose and fine to recompute (API responses, job progress).
    Without Redis each worker process keeps its own copy, so this is never the place for
    anything another process has to see.
    """

    def __init__(self, prefix: str, ttl_seconds: int, max_local: int = 2000):
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self.max_local = max_local
        self.local: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        full_key = f"{self.prefix}:{key}"
        client = get_redis()
        if client is not None:
            try:
                raw = client.get(full_key)
                return json.loads(raw) if raw else None
            except Exception as exc:
                logger.warning("Cache read failed for %s, using this process: %s", self.prefix, exc)
        entry = self.local.get(full_key)
        if entry and entry[0] > time.monotonic():
            return entry[1]
        self.local.pop(full_key, None)
        return None

    def set(self, key: str, value: Any) -> None:
        full_key = f"{self.prefix}:{key}"
        client = get_redis()
        if client is not None:
            try:
                client.setex(full_key, self.ttl_seconds, json.dumps(value))
                return
            except Exception as exc:
                logger.warning("Cache write failed for %s, using this process: %s", self.prefix, exc)
        if len(self.local) >= self.max_local:
            now = time.monotonic()
            self.local = {k: v for k, v in self.local.items() if v[0] > now}
            if len(self.local) >= self.max_local:
                self.local.clear()
        self.local[full_key] = (time.monotonic() + self.ttl_seconds, value)
