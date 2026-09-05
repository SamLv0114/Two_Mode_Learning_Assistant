"""
Conversation and long-term user-fact memory.

ConversationMemory  — per-session message history (Redis + in-memory fallback)
UserFactMemory      — persistent user facts extracted from conversations (Redis)
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional

from openai import OpenAI

from src.utils.config import settings

logger = logging.getLogger(__name__)

MAX_HISTORY_LENGTH = 20
SESSION_TTL_SECONDS = 86400       # 24 hours
FACTS_TTL_SECONDS = 86400 * 30   # 30 days
MAX_FACTS = 25                    # rolling cap per user


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


class UserFactMemory:
    """
    Persistent long-term user facts extracted after each conversation turn.

    Facts are short strings like:
      "user is studying contrastive self-supervised learning"
      "user found the attention mechanism paper interesting"
      "user's goal: write a survey on RLHF by end of semester"

    Stored in Redis under key  user_facts:{user_id}  as a JSON list (TTL 30d).
    Inject get_context_string() into an agent's system prompt each session.
    Uses gpt-4o-mini for lightweight extraction.
    """

    _fallback: Dict[int, list] = {}

    # V4: temporal decay constant (Zep/Graphiti-inspired)
    _DECAY_BASE = 0.95  # per 30-day period

    _EXTRACT_PROMPT = """\
You are a memory extraction agent for an ML research assistant.

Given the following conversation exchange, extract any facts about the user that
would help personalise future responses. Focus on:
- Topics or papers they are actively studying
- Concepts they found interesting or confusing
- Their research goals or deadlines
- Their self-described expertise level
- Preferences for explanation style

User message: {user_msg}
Assistant response: {assistant_msg}

Return a JSON array of concise fact strings (max 3 items).
Return [] if this exchange reveals nothing new and useful about the user.
Each fact must start with "user" (e.g. "user is studying LoRA fine-tuning").
No markdown fences. Only the JSON array."""

    def __init__(self):
        self._client: Optional[OpenAI] = None
        if settings.OPENAI_API_KEY:
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # ── Storage helpers ───────────────────────────────────────────────────────

    def _key(self, user_id: int) -> str:
        return f"user_facts:{user_id}"

    def _load(self, user_id: int, redis_client) -> List[Dict]:
        """Load facts as List[Dict]; handles backward-compat with old List[str] format."""
        raw_list = []
        if redis_client:
            try:
                raw = redis_client.get(self._key(user_id))
                raw_list = json.loads(raw) if raw else []
            except Exception:
                pass
        else:
            raw_list = list(self._fallback.get(user_id, []))

        # Normalize old plain-string entries to dict format
        normalized = []
        for item in raw_list:
            if isinstance(item, str):
                normalized.append({"fact": item, "created_at": None, "negative": False})
            elif isinstance(item, dict):
                normalized.append(item)
        return normalized

    def _save(self, user_id: int, facts: List[Dict], redis_client) -> None:
        if redis_client:
            try:
                redis_client.setex(self._key(user_id), FACTS_TTL_SECONDS, json.dumps(facts))
                return
            except Exception:
                pass
        self._fallback[user_id] = facts

    def _decay_weight(self, fact: Dict) -> float:
        """Compute temporal decay weight: 0.95^(days_since_creation / 30)."""
        created_at = fact.get("created_at")
        if not created_at:
            return 0.0  # Unknown age — lowest priority
        try:
            created = datetime.fromisoformat(created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            days = max(0, (datetime.now(timezone.utc) - created).days)
            return self._DECAY_BASE ** (days / 30)
        except Exception:
            return 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_and_store(
        self,
        user_id: int,
        user_message: str,
        assistant_response: str,
        redis_client=None,
    ) -> List[str]:
        """
        Extract facts from one conversation exchange and merge into storage.
        Returns newly extracted facts (may be empty).
        Call as a background task after each chat turn.
        """
        if not self._client:
            return []

        prompt = self._EXTRACT_PROMPT.format(
            user_msg=user_message[:800],
            assistant_msg=assistant_response[:800],
        )
        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            new_facts: List[str] = json.loads(text)
            if not isinstance(new_facts, list):
                return []
            new_facts = [f for f in new_facts if isinstance(f, str) and f.startswith("user")][:3]
        except Exception as e:
            logger.debug(f"UserFactMemory extraction failed: {e}")
            return []

        if not new_facts:
            return []

        # V4: wrap new facts as dicts with creation timestamp
        now_str = datetime.now(timezone.utc).isoformat()
        new_fact_dicts = [{"fact": f, "created_at": now_str, "negative": False} for f in new_facts]

        existing = self._load(user_id, redis_client)
        merged = list(existing)
        for fd in new_fact_dicts:
            fact_text = fd["fact"]
            if not any(fact_text.lower() in e["fact"].lower() or e["fact"].lower() in fact_text.lower() for e in merged):
                merged.append(fd)

        # V4: trim by decay weight (drop lowest-weight / oldest facts first)
        if len(merged) > MAX_FACTS:
            merged = sorted(merged, key=lambda f: self._decay_weight(f), reverse=True)[:MAX_FACTS]

        self._save(user_id, merged, redis_client)
        logger.debug(f"UserFactMemory: stored {len(new_facts)} new facts for user {user_id}")
        return new_facts

    def add_negative_fact(self, user_id: int, reason: str, redis_client=None) -> None:
        """
        Store a negative preference extracted from a dismiss action.
        Called by the interactions router when a paper is dismissed.
        Reason should be a short phrase like 'too theoretical, no code examples'.
        """
        fact_text = f"user disliked: {reason}"
        now_str = datetime.now(timezone.utc).isoformat()
        fact_dict = {"fact": fact_text, "created_at": now_str, "negative": True}

        existing = self._load(user_id, redis_client)
        # Avoid near-duplicate negative facts
        if not any(reason.lower() in e["fact"].lower() for e in existing if e.get("negative")):
            existing.append(fact_dict)
            if len(existing) > MAX_FACTS:
                existing = sorted(existing, key=lambda f: self._decay_weight(f), reverse=True)[:MAX_FACTS]
            self._save(user_id, existing, redis_client)
            logger.debug(f"UserFactMemory: stored negative fact for user {user_id}: {reason[:60]}")

    def get_context_string(self, user_id: int, redis_client=None) -> str:
        """Return a context block to prepend to an agent's system prompt.

        V4: Facts are sorted by temporal decay weight (most recent first),
        top 15 injected. Negative facts are labeled to guide avoidance.
        """
        facts = self._load(user_id, redis_client)
        if not facts:
            return ""

        # Sort by decay weight — newest facts get highest weight
        weighted = sorted(facts, key=lambda f: self._decay_weight(f), reverse=True)
        top = weighted[:15]

        lines = "\n".join(f"- {f['fact']}" for f in top)
        return f"\n\n## Personalisation context (from prior sessions)\n{lines}\n"

    def clear(self, user_id: int, redis_client=None) -> None:
        if redis_client:
            try:
                redis_client.delete(self._key(user_id))
            except Exception:
                pass
        self._fallback.pop(user_id, None)
