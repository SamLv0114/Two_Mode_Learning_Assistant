"""
Redis-backed conversation memory with transparent in-memory fallback.
"""
import json
import logging
import uuid
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

MAX_HISTORY_LENGTH = 20
SESSION_TTL_SECONDS = 86400  # 24 hours


class ConversationMemory:
    """
    Stores per-session conversation history.
    Uses Redis when available; falls back to a class-level dict silently.
    """

    _fallback: Dict[str, list] = {}

    def __init__(self, redis_client=None, user_id: int = 0):
        self.redis = redis_client
        self.user_id = user_id

    def _key(self, session_id: str) -> str:
        return f"chat:{self.user_id}:{session_id}"

    def add_message(self, session_id: str, role: str, content: str) -> None:
        message = {"role": role, "content": content}
        if self.redis:
            try:
                key = self._key(session_id)
                raw = self.redis.get(key)
                history = json.loads(raw) if raw else []
                history.append(message)
                if len(history) > MAX_HISTORY_LENGTH:
                    history = history[-MAX_HISTORY_LENGTH:]
                self.redis.setex(key, SESSION_TTL_SECONDS, json.dumps(history))
                return
            except Exception as e:
                logger.warning(f"Redis write failed, falling back to memory: {e}")

        key = self._key(session_id)
        self._fallback.setdefault(key, []).append(message)
        if len(self._fallback[key]) > MAX_HISTORY_LENGTH:
            self._fallback[key] = self._fallback[key][-MAX_HISTORY_LENGTH:]

    def get_history(self, session_id: str, max_messages: int = 10) -> List[Dict]:
        if self.redis:
            try:
                raw = self.redis.get(self._key(session_id))
                if raw:
                    return json.loads(raw)[-max_messages:]
                return []
            except Exception as e:
                logger.warning(f"Redis read failed, falling back to memory: {e}")

        history = self._fallback.get(self._key(session_id), [])
        return history[-max_messages:]

    def clear_session(self, session_id: str) -> None:
        if self.redis:
            try:
                self.redis.delete(self._key(session_id))
                return
            except Exception:
                pass
        self._fallback.pop(self._key(session_id), None)

    @staticmethod
    def new_session_id() -> str:
        return str(uuid.uuid4())
