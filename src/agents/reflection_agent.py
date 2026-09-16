"""
ReflectionAgent: Generate, Critique, Refine self-correction loop.

Implements the Reflection paradigm from Hello-Agents ch4.4.

Flow:
  1. ResearchAgent produces an initial answer (standard tool-calling loop)
  2. CriticAgent scores groundedness / completeness / clarity (0-1 each)
  3. If aggregate score < threshold: inject critique and retry once
  4. Return the better of the two answers

Both run() and stream() now execute the full loop. stream() used to just
delegate to the inner agent to avoid doubling latency, which also meant it
was the only path the frontend actually calls (chat.py's non-streaming /chat
route has no real traffic), so the reflection loop never ran for a real
user. stream() now runs the same loop, emitting progress events
(generating/critiquing/critique_result/refining) so the user sees what's
happening during the extra round trip instead of a silent pause, and
forwards the inner agent's tool_call/tool_result events live during both
the draft and (if needed) the refine pass, so tool-use visibility is not
lost compared to the plain streaming path.
"""
import logging
import re
from typing import Any, Dict, Generator, List

from src.agents.base_agent import AgentResult, BaseAgent

logger = logging.getLogger(__name__)

# How many "word + trailing whitespace" units go into one simulated token
# event when re-emitting an already-fully-generated answer. The text was
# produced by a blocking call (it has to exist in full before critique can
# score it), so there is nothing to genuinely stream. This only recreates
# the same incremental-token look the frontend already renders for
# PlanAndSolveAgent/DeepResearchAgent's real stream=True output.
CHUNK_WORDS = 3


def _chunk_final_text(text: str, words_per_chunk: int = CHUNK_WORDS):
    """Yield text split into small pieces, preserving original spacing."""
    units = re.findall(r"\S+\s*", text)
    for i in range(0, len(units), words_per_chunk):
        yield "".join(units[i:i + words_per_chunk])


class ReflectionAgent(BaseAgent):
    """
    Wraps ResearchAgent with a one-shot critique-and-refine loop.

    Uses gpt-4o for generation (via ResearchAgent) and gpt-4o-mini
    for evaluation (via CriticAgent), a cost-efficient quality improvement.
    """

    name = "ReflectionAgent"
    system_prompt = ""    # inner agent owns its prompt
    tool_schemas = []     # inner agent owns its tools

    def __init__(self):
        from src.agents.research_agent import ResearchAgent
        from src.agents.critic_agent import CriticAgent
        # Created before super().__init__() so the tool_call_listener
        # property below (which forwards onto self._inner) has something
        # to forward onto by the time BaseAgent.__init__ sets it.
        self._inner = ResearchAgent()
        self._critic = CriticAgent()
        super().__init__()  # sets self.client and (via the property) self.tool_call_listener

    # ReflectionAgent never dispatches a tool call itself — self._inner
    # does, in both run() and stream(). ToolAwareAgent (the streaming
    # observability wrapper) only sets tool_call_listener on whatever
    # top-level agent AgentRouter handed it, which is this ReflectionAgent
    # instance, not self._inner. Without forwarding the assignment,
    # ToolAwareAgent's listener would sit on an object that never calls it,
    # and agent_trace:{session_id} would go silently empty for every
    # ReflectionAgent-routed request even though the SSE tool_call/
    # tool_result events (yielded directly by self._inner.stream(),
    # independent of this listener) still show up fine.
    @property
    def tool_call_listener(self):
        return self._inner.tool_call_listener

    @tool_call_listener.setter
    def tool_call_listener(self, value):
        self._inner.tool_call_listener = value

    def run(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> AgentResult:
        # Phase 1 — Generate
        result = self._inner.run(message, conversation_history, context)
        logger.info(f"[ReflectionAgent] Draft: {len(result.reply)} chars")

        # Phase 2 — Critique
        crit = self._critic.evaluate(message, result.reply, result.citations)

        if not crit.should_retry or not crit.critique:
            logger.info(f"[ReflectionAgent] Quality OK (agg={crit.aggregate:.2f}), returning draft")
            return result

        # Phase 3 — Refine with critique injected as context
        logger.info(
            f"[ReflectionAgent] agg={crit.aggregate:.2f} below threshold, "
            f"refining. Critique: {crit.critique}"
        )
        refined_prompt = (
            f"{message}\n\n"
            f"[Reflection note — your previous draft scored {crit.aggregate:.0%} on quality. "
            f"Specific issue: {crit.critique} "
            f"Please produce a revised, improved answer that addresses this issue directly.]"
        )

        try:
            refined = self._inner.run(refined_prompt, conversation_history, context)
            if len(refined.reply) >= len(result.reply) * 0.7:
                logger.info(f"[ReflectionAgent] Using refined answer ({len(refined.reply)} chars)")
                return refined
        except Exception as e:
            logger.warning(f"[ReflectionAgent] Refinement step failed: {e}")

        return result

    def _stream_inner(self, message, conversation_history, context):
        """
        Run the inner agent's stream(), forwarding tool_call/tool_result
        events live (so tool-use progress stays visible) while buffering
        token fragments (nothing should be shown to the user until this
        draft has been through critique). Returns (text, tools_called,
        citations); yields forwarded events and, once, an "error" event if
        the inner stream fails, in which case the returned text is "".
        """
        parts: List[str] = []
        tools_called: List[str] = []
        citations: List[Dict] = []
        for event in self._inner.stream(message, conversation_history, context):
            etype = event.get("type")
            if etype == "token":
                parts.append(event["value"])
            elif etype in ("tool_call", "tool_result"):
                yield event
            elif etype == "done":
                tools_called = event.get("tools_called", [])
                citations = event.get("citations", [])
            elif etype == "error":
                yield event
                return None, [], []   # None marks "already reported", vs. "" a valid empty draft
            # "generating" from the inner agent is swallowed — this agent
            # emits its own generating/critiquing/refining progress events.
        return "".join(parts), tools_called, citations

    def stream(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> Generator:
        """
        Streaming path: runs the same generate/critique/refine loop as
        run(), emitting progress events for each phase, then re-emits the
        final chosen answer as simulated token events (see _chunk_final_text
        — the text already exists in full by this point, there is nothing
        left to genuinely stream).
        """
        # Phase 1 — Generate. `yield from` both forwards every event
        # _stream_inner yields (tool_call/tool_result/error) and captures
        # its `return` tuple into draft_text/tools_called/citations.
        # A plain `for item in generator: ...` loop cannot reach a
        # generator's return value, only `yield from` (or manual
        # next()/StopIteration handling) can.
        yield {"type": "generating"}
        draft_text, tools_called, citations = yield from self._stream_inner(
            message, conversation_history, context
        )

        if draft_text is None:
            return  # inner stream already reported its own "error" event

        if not draft_text:
            yield {"type": "done", "tools_called": tools_called, "citations": citations}
            return

        logger.info(f"[ReflectionAgent] Draft (stream): {len(draft_text)} chars")

        # Phase 2 — Critique
        yield {"type": "critiquing"}
        crit = self._critic.evaluate(message, draft_text, citations)
        yield {"type": "critique_result", **crit.to_dict()}

        final_text, final_tools, final_citations = draft_text, tools_called, citations

        if crit.should_retry and crit.critique:
            logger.info(
                f"[ReflectionAgent] agg={crit.aggregate:.2f} below threshold, "
                f"refining (stream). Critique: {crit.critique}"
            )
            yield {"type": "refining"}
            refined_prompt = (
                f"{message}\n\n"
                f"[Reflection note — your previous draft scored {crit.aggregate:.0%} on quality. "
                f"Specific issue: {crit.critique} "
                f"Please produce a revised, improved answer that addresses this issue directly.]"
            )
            # Non-streaming here, deliberately: same call run() makes. A
            # failure here should fall back to the perfectly good draft in
            # hand, silently, the same way run() does. Forwarding a live
            # "error" event from a nested stream would flash an error at
            # the user right before a normal answer still follows, and the
            # frontend's error handler overwrites the message's content on
            # that event, which later token events would then just append
            # onto rather than replace.
            try:
                refined = self._inner.run(refined_prompt, conversation_history, context)
                if len(refined.reply) >= len(draft_text) * 0.7:
                    logger.info(f"[ReflectionAgent] Using refined answer (stream, {len(refined.reply)} chars)")
                    final_text = refined.reply
                    final_tools = refined.tools_called
                    final_citations = refined.citations
            except Exception as e:
                logger.warning(f"[ReflectionAgent] Streaming refinement failed: {e}")

        # Phase 3 — Emit the final chosen answer
        yield {"type": "generating"}
        for chunk in _chunk_final_text(final_text):
            yield {"type": "token", "value": chunk}

        yield {"type": "done", "tools_called": final_tools, "citations": final_citations}
