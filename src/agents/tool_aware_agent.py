"""
ToolAwareAgent — observability wrapper for any BaseAgent.

Inspired by Hello-Agents ch14 ToolAwareSimpleAgent with tool_call_listener.

Attaches a tool_call_listener to any BaseAgent that:
  - Records the tool name, arguments, result preview, and wall-clock duration
  - Optionally persists the trace to Redis under  agent_trace:{session_id}

The GET /chat/trace/{session_id} endpoint reads these traces to expose
full agent reasoning to developers and demos.

Usage:
    agent = ResearchAgent()
    aware = ToolAwareAgent(agent, session_id=session_id, redis_client=redis)
    result = aware.run(message, history, context)
    trace = aware.get_trace()   # list of ToolTraceEntry dicts
"""
import json
import logging
import time
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
        session_id: str,
        redis_client=None,
    ):
        self._inner = agent
        self.session_id = session_id
        self.redis = redis_client
        self._trace: List[ToolTraceEntry] = []

        # Expose inner agent's name for router compatibility
        self.name = agent.name

        # Attach the listener to the inner agent
        agent.tool_call_listener = self._on_tool_call

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
            key = f"agent_trace:{self.session_id}"
            raw = self.redis.get(key)
            trace = json.loads(raw) if raw else []
            trace.append(asdict(entry))
            if len(trace) > MAX_TRACE_ENTRIES:
                trace = trace[-MAX_TRACE_ENTRIES:]
            self.redis.setex(key, TRACE_TTL_SECONDS, json.dumps(trace))
        except Exception as e:
            logger.debug(f"ToolAwareAgent: Redis persist failed: {e}")

    # ── Delegation ────────────────────────────────────────────────────────────

    def run(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> AgentResult:
        return self._inner.run(message, conversation_history, context)

    def stream(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> Generator:
        yield from self._inner.stream(message, conversation_history, context)

    # ── Trace access ──────────────────────────────────────────────────────────

    def get_trace(self) -> List[Dict]:
        """Return in-memory trace as plain dicts."""
        return [asdict(e) for e in self._trace]

    @staticmethod
    def load_trace(session_id: str, redis_client) -> List[Dict]:
        """Load a persisted trace from Redis (e.g., from a prior request)."""
        if not redis_client:
            return []
        try:
            raw = redis_client.get(f"agent_trace:{session_id}")
            return json.loads(raw) if raw else []
        except Exception:
            return []
