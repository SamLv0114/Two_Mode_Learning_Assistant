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
from typing import Any, Dict, Generator, List

from src.agents.base_agent import AgentResult, BaseAgent

logger = logging.getLogger(__name__)

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
        self._inner = ResearchAgent()
        self._critic = CriticAgent()
        super().__init__()

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
            # Re-score the refined draft instead of only checking it isn't
            # suspiciously short — a length check can't tell "improved" from
            # "still bad but not truncated". Only swap in the refined answer
            # if it actually scored better than what it's replacing.
            refined_crit = self._critic.evaluate(message, refined.reply, refined.citations)
            if refined_crit.aggregate >= crit.aggregate:
                logger.info(
                    f"[ReflectionAgent] Using refined answer "
                    f"(agg={refined_crit.aggregate:.2f} vs draft {crit.aggregate:.2f})"
                )
                return refined
            logger.info(
                f"[ReflectionAgent] Refined answer did not improve "
                f"(agg={refined_crit.aggregate:.2f} vs draft {crit.aggregate:.2f}) — keeping draft"
            )
        except Exception as e:
            logger.warning(f"[ReflectionAgent] Refinement step failed: {e}")

        return result

    def _stream_inner(self, message, conversation_history, context, display_draft=True):
        """
        Run the inner agent's stream(), forwarding tool_call/tool_result
        events live (so tool-use progress stays visible), forwarding draft
        chunks provisionally when requested. Returns (text, tools_called,
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
                if display_draft:
                    yield {"type": "draft_token", "value": event["value"]}
            elif etype in ("tool_call", "tool_result"):
                yield event
            elif etype == "done":
                tools_called = event.get("tools_called", [])
                citations = event.get("citations", [])
            elif etype == "error":
                if display_draft:
                    yield event
                return None, [], []   # None marks "already reported", vs. "" a valid empty draft
            elif etype == "cancelled":
                if display_draft:
                    yield event
                return None, [], []
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
        Stream the provisional draft immediately, then replace it only when
        critique verifies that a refined answer is better.
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

        if self._cancelled(context):
            yield {"type": "cancelled"}
            return

        logger.info(f"[ReflectionAgent] Draft (stream): {len(draft_text)} chars")

        # Phase 2 — Critique
        yield {"type": "critiquing"}
        crit = self._critic.evaluate(message, draft_text, citations)
        if self._cancelled(context):
            yield {"type": "cancelled"}
            return
        yield {"type": "critique_result", **crit.to_dict()}

        final_text, final_tools, final_citations = draft_text, tools_called, citations

        if crit.should_retry and crit.critique:
            if self._cancelled(context):
                yield {"type": "cancelled"}
                return
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
            try:
                refined_text, refined_tools, refined_citations = yield from self._stream_inner(
                    refined_prompt, conversation_history, context, display_draft=False,
                )
                if self._cancelled(context):
                    yield {"type": "cancelled"}
                    return
                if refined_text is None:
                    refined_text = ""
                # Re-score the refined draft rather than only checking length
                # (see run()'s identical fix). Only swap it in — and only
                # then update the quality badge the frontend already showed
                # for the draft — if it actually scored better; otherwise the
                # draft's score stays correct for the draft that's still
                # what gets shown.
                refined_crit = (self._critic.evaluate(message, refined_text, refined_citations)
                                if refined_text else None)
                if refined_crit and refined_crit.aggregate >= crit.aggregate:
                    logger.info(
                        f"[ReflectionAgent] Using refined answer (stream, "
                        f"agg={refined_crit.aggregate:.2f} vs draft {crit.aggregate:.2f})"
                    )
                    final_text = refined_text
                    final_tools = refined_tools
                    final_citations = refined_citations
                    yield {"type": "critique_result", **refined_crit.to_dict()}
                else:
                    logger.info(
                        f"[ReflectionAgent] Refined answer did not improve (stream, "
                        f"agg={refined_crit.aggregate if refined_crit else 'unavailable'} vs draft {crit.aggregate:.2f}) — keeping draft"
                    )
            except Exception as e:
                logger.warning(f"[ReflectionAgent] Streaming refinement failed: {e}")

        if final_text != draft_text:
            yield {"type": "replace", "value": final_text}

        yield {"type": "done", "tools_called": final_tools, "citations": final_citations}
