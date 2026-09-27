"""
ToolAwareAgent — observability wrapper for any BaseAgent.

Inspired by Hello-Agents ch14 ToolAwareSimpleAgent with tool_call_listener.

Adds a request-local tool-call listener that:
  - Records the tool name, arguments, result preview, and wall-clock duration
  - Optionally persists the trace to Redis under  agent_trace:{user_id}:{session_id}

The GET /chat/trace/{session_id} endpoint reads these traces to expose
full agent reasoning to developers and demos.

Usage:
    agent = ResearchAgent()
    aware = ToolAwareAgent(agent, user_id=user_id, session_id=session_id, redis_client=redis)
    result = aware.run(message, history, context)
    trace = aware.get_trace()   # list of ToolTraceEntry dicts
"""
import json
import logging
import time
from contextlib import closing
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Generator, List, Optional

from src.agents.base_agent import AgentResult, BaseAgent

logger = logging.getLogger(__name__)

TRACE_TTL_SECONDS = 3600     # 1 hour
MAX_TRACE_ENTRIES = 100


@dataclass
class ToolTraceEntry:
    tool: str
    agent: str
    args: Dict[str, str]          # values truncated to 120 chars
    result_preview: str           # first 250 chars of JSON result
    duration_ms: int
    timestamp: float = field(default_factory=time.time)


class ToolAwareAgent:
    """
    Transparent wrapper that instruments a BaseAgent's tool calls.

    All public methods delegate to the inner agent, so ToolAwareAgent is a
    drop-in replacement anywhere a BaseAgent is expected.
    """

    def __init__(
        self,
        agent: BaseAgent,
        user_id: int,
        session_id: str,
        redis_client=None,
    ):
        self._inner = agent
        self.user_id = user_id
        self.session_id = session_id
        self.redis = redis_client
        self._trace: List[ToolTraceEntry] = []

        # Expose inner agent's name for router compatibility
        self.name = agent.name

        # Never assign agent.tool_call_listener: the router caches agents and
        # another request can be using this same instance concurrently.

    @staticmethod
    def trace_key(user_id: int, session_id: str) -> str:
        return f"agent_trace:{user_id}:{session_id}"

    def _request_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        request_context = dict(context)
        request_context["_tool_call_listener"] = self._on_tool_call
        return request_context

    # ── Listener ──────────────────────────────────────────────────────────────

    def _on_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Dict,
        duration_ms: int,
    ) -> None:
        entry = ToolTraceEntry(
            tool=tool_name,
            agent=self._inner.name,
            args={k: str(v)[:120] for k, v in args.items()},
            result_preview=json.dumps(result)[:250],
            duration_ms=duration_ms,
        )
        self._trace.append(entry)
        self._persist(entry)
        logger.debug(
            f"[ToolAwareAgent] {self._inner.name}.{tool_name} "
            f"→ {duration_ms}ms"
        )

    def _persist(self, entry: ToolTraceEntry) -> None:
        """Append to Redis trace list for this session."""
        if not self.redis:
            return
        try:
            key = self.trace_key(self.user_id, self.session_id)
            # Redis list append is atomic. A transaction keeps the size cap
            # and TTL together with the append under concurrent tool calls.
            with self.redis.pipeline(transaction=True) as pipe:
                pipe.rpush(key, json.dumps(asdict(entry)))
                pipe.ltrim(key, -MAX_TRACE_ENTRIES, -1)
                pipe.expire(key, TRACE_TTL_SECONDS)
                pipe.execute()
        except Exception as e:
            logger.debug(f"ToolAwareAgent: Redis persist failed: {e}")

    # ── Delegation ────────────────────────────────────────────────────────────

    def run(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> AgentResult:
        return self._inner.run(message, conversation_history, self._request_context(context))

    def stream(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> Generator:
        request_context = self._request_context(context)
        with closing(self._inner.stream(message, conversation_history, request_context)) as events:
            for event in events:
                cancel = request_context.get("_cancel_event")
                if cancel is not None and cancel.is_set():
                    yield {"type": "cancelled"}
                    return
                yield event

    # ── Trace access ──────────────────────────────────────────────────────────

    def get_trace(self) -> List[Dict]:
        """Return in-memory trace as plain dicts."""
        return [asdict(e) for e in self._trace]

    @staticmethod
    def load_trace(user_id: int, session_id: str, redis_client) -> List[Dict]:
        """Load a persisted trace from Redis (e.g., from a prior request)."""
        if not redis_client:
            return []
        try:
            raw = redis_client.lrange(ToolAwareAgent.trace_key(user_id, session_id), 0, -1)
            return [json.loads(item) for item in raw]
        except Exception:
            return []
