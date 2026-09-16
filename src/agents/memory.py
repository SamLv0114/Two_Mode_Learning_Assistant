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
You are a memory extraction agent for a research assistant.

Given the following conversation exchange, extract any facts about the user that
would help personalise future responses. Focus on:
- Topics or papers they are actively studying
- Concepts they found interesting or confusing
- Their research goals or deadlines
- Their self-described expertise level
- Preferences for explanation style

User message: {user_msg}
Assistant response: {assistant_msg}

Extract 0 to 3 concise facts. Each fact must start with "user"
(e.g. "user is studying LoRA fine-tuning"). If this exchange reveals nothing
new and useful about the user, extract no facts."""

    # Forced function call instead of a "Return JSON only" prompt instruction
    # + json.loads — the prompt-only version silently degraded to "extracted
    # nothing" whenever the model added a fence or a lead-in sentence, since
    # json.loads() raised and the except below swallowed it. Private to this
    # class, not registered as an agent-invokable tool — nothing ever chooses
    # whether to call it, it only exists to constrain this call's output shape.
    _EXTRACT_FACTS_TOOL = {
        "type": "function",
        "function": {
            "name": "extract_user_facts",
            "description": (
                "Record 0 to 3 concise facts about the user extracted from this "
                "conversation exchange, for personalizing future responses."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "A concise fact starting with 'user', e.g. 'user is studying LoRA fine-tuning'",
                        },
                        "description": "0 to 3 facts; empty if nothing new and useful was revealed",
                    },
                },
                "required": ["facts"],
                "additionalProperties": False,
            },
        },
    }

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

    # Common function words stripped before computing word overlap for the
    # "update vs distinct" judgment below. Every extracted fact starts with
    # "user" and shares template words like "is"/"was" regardless of topic —
    # without stripping these, two facts about unrelated topics (e.g. "user
    # is studying rl" vs "user is studying rlhf") can share enough boilerplate
    # tokens to look like the same topic by raw overlap alone, even though
    # the actual content words ("rl" vs "rlhf") don't overlap at all.
    _STOPWORDS = frozenset({
        "user", "users", "is", "are", "was", "were", "be", "been", "a", "an",
        "the", "of", "to", "in", "on", "at", "for", "with", "and", "or",
        "but", "this", "that", "these", "those", "their", "user's", "over",
    })

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return text.lower().split()

    @staticmethod
    def _contains_sublist(haystack: List[str], needle: List[str]) -> bool:
        """Whether needle appears as a contiguous run inside haystack."""
        n, m = len(haystack), len(needle)
        if m == 0 or m > n:
            return False
        return any(haystack[i:i + m] == needle for i in range(n - m + 1))

    @classmethod
    def _classify_fact_relation(cls, new_fact: str, existing_fact: str) -> str:
        """
        Classify how a new fact relates to an existing one, for merge purposes.

        Word-boundary aware — unlike a raw substring check (the previous
        approach), which false-positives whenever one fact's text happens to
        be a literal character prefix of another's, e.g. "user is studying rl"
        is a substring of "user is studying rlhf" even though "rl" and "rlhf"
        are different topics; token-level comparison doesn't make that
        mistake since "rl" != "rlhf" as tokens.

        Returns one of:
          "duplicate"   — identical tokens, nothing to do
          "supersedes"  — new fact's tokens contain the existing fact's tokens
                          plus more (new is a more detailed version) -> replace
          "redundant"   — existing fact's tokens already contain the new
                          fact's tokens (new adds nothing) -> drop new
          "update"      — high word overlap without containment either way —
                          same topic, but not a strict refinement (e.g. a
                          preference that changed) -> replace old with new
          "distinct"    — unrelated -> keep both
        """
        new_tokens = cls._tokens(new_fact)
        old_tokens = cls._tokens(existing_fact)
        if new_tokens == old_tokens:
            return "duplicate"
        if cls._contains_sublist(new_tokens, old_tokens):
            return "supersedes"
        if cls._contains_sublist(old_tokens, new_tokens):
            return "redundant"

        # Content-word overlap only (see _STOPWORDS) — otherwise shared
        # boilerplate ("user", "is") alone can push unrelated facts over the
        # threshold, as it did for "user is studying rl" vs "...studying rlhf"
        # during testing before this filter was added.
        new_content = {t for t in new_tokens if t not in cls._STOPWORDS}
        old_content = {t for t in old_tokens if t not in cls._STOPWORDS}
        overlap = len(new_content & old_content)
        union = len(new_content | old_content)
        jaccard = overlap / union if union else 0.0
        return "update" if jaccard >= 0.5 else "distinct"

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
                tools=[self._EXTRACT_FACTS_TOOL],
                tool_choice={"type": "function", "function": {"name": "extract_user_facts"}},
            )
            call = response.choices[0].message.tool_calls[0]
            args = json.loads(call.function.arguments)
            new_facts = args.get("facts", [])
            if not isinstance(new_facts, list):
                return []
            # Schema can't enforce a string-prefix pattern or a max item count
            # under strict mode — keep these as a defensive filter, same as before.
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
            action = "add"
            replace_idx = None
            for i, e in enumerate(merged):
                if e.get("negative"):
                    continue  # dismiss-reason facts aren't merged against topic facts
                relation = self._classify_fact_relation(fact_text, e["fact"])
                if relation in ("duplicate", "redundant"):
                    action = "skip"
                    break
                if relation in ("supersedes", "update"):
                    # "supersedes": new fact is a more detailed version of an
                    # existing one. "update": same topic, high word overlap,
                    # but not a strict refinement — most often a preference
                    # that changed (e.g. "wants theory" -> "wants code
                    # examples") rather than two things both worth keeping.
                    action = "replace"
                    replace_idx = i
                    break
            if action == "add":
                merged.append(fd)
            elif action == "replace":
                merged[replace_idx] = fd
            # action == "skip": new fact adds nothing, drop it

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
