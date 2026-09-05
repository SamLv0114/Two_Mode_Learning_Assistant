"""
Rate limiting middleware.

settings.RATE_LIMIT_PER_MINUTE existed as configuration long before any code
enforced it, which is worse than having no setting at all — it reads like the
API is protected when it is not. This module makes the setting real.

Backends, chosen at request time:
  - Redis  — shared across workers, survives restarts. Used when REDIS_URL is
             configured and reachable.
  - Memory — per-process fallback. Correct for a single worker; with N workers
             the effective limit is up to N x the configured value. Still bounds
             a runaway client, so it is preferred over failing open entirely.

Fixed-window counters (not sliding) — a client can burst up to 2x the limit
across a window boundary. That is the standard trade-off for this approach and
is acceptable here; the goal is bounding abuse, not precise metering.
"""
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.config import settings

logger = logging.getLogger(__name__)

# Paths that must stay reachable for monitoring and docs.
EXEMPT_PREFIXES = ("/health", "/metrics", "/docs", "/redoc", "/openapi.json")

WINDOW_SECONDS = 60


class _MemoryWindow:
    """Per-process fixed-window counter. Prunes lazily to bound memory."""

    def __init__(self) -> None:
        self._hits: Dict[str, List[float]] = defaultdict(list)
        self._last_prune = time.time()

    def incr(self, key: str, now: float) -> int:
        cutoff = now - WINDOW_SECONDS

        # Periodic sweep so keys for departed clients do not accumulate.
        if now - self._last_prune > WINDOW_SECONDS:
            for k in list(self._hits):
                kept = [t for t in self._hits[k] if t > cutoff]
                if kept:
                    self._hits[k] = kept
                else:
                    del self._hits[k]
            self._last_prune = now

        hits = [t for t in self._hits[key] if t > cutoff]
        hits.append(now)
        self._hits[key] = hits
        return len(hits)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Bounds requests per client per minute, keyed by client IP."""

    def __init__(self, app, limit_per_minute: Optional[int] = None) -> None:
        super().__init__(app)
        self.limit = limit_per_minute or settings.RATE_LIMIT_PER_MINUTE
        self._memory = _MemoryWindow()
        self._redis = None
        self._redis_checked = False

    # ── Backend selection ────────────────────────────────────────────────────

    def _get_redis(self):
        """Resolve a Redis client once; fall back to memory permanently on failure."""
        if self._redis_checked:
            return self._redis
        self._redis_checked = True
        if not settings.REDIS_URL:
            logger.info("Rate limiting: REDIS_URL unset — using in-process counters")
            return None
        try:
            import redis
            client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            client.ping()
            self._redis = client
            logger.info("Rate limiting: using Redis backend")
        except Exception as e:
            logger.warning(f"Rate limiting: Redis unavailable ({e}) — using in-process counters")
        return self._redis

    @staticmethod
    def _client_key(request: Request) -> str:
        """
        Identify the caller. X-Forwarded-For wins when present so a reverse proxy
        does not collapse every client onto one bucket; its first hop is the client.
        """
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _count(self, key: str) -> int:
        """Return this client's hit count in the current window."""
        now = time.time()
        rc = self._get_redis()
        if rc is not None:
            try:
                bucket = int(now // WINDOW_SECONDS)
                rkey = f"ratelimit:{key}:{bucket}"
                pipe = rc.pipeline()
                pipe.incr(rkey)
                pipe.expire(rkey, WINDOW_SECONDS * 2)
                count, _ = pipe.execute()
                return int(count)
            except Exception as e:
                # Never let a limiter outage take down the API.
                logger.warning(f"Rate limiting: Redis op failed ({e}) — falling back to memory")
                self._redis = None
        return self._memory.incr(key, now)

    # ── Dispatch ─────────────────────────────────────────────────────────────

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith(EXEMPT_PREFIXES):
            return await call_next(request)

        key = self._client_key(request)
        count = self._count(key)
        remaining = max(0, self.limit - count)

        if count > self.limit:
            logger.warning(f"Rate limit exceeded for {key}: {count}/{self.limit}")
            retry_after = WINDOW_SECONDS - int(time.time()) % WINDOW_SECONDS
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: {self.limit} requests per minute. "
                        f"Retry in {retry_after}s."
                    )
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
