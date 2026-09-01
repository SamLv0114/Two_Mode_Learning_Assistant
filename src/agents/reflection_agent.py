"""
ReflectionAgent — Generate → Critique → Refine self-correction loop.

Implements the Reflection paradigm from Hello-Agents ch4.4.

Flow:
  1. ResearchAgent produces an initial answer (standard tool-calling loop)
  2. CriticAgent scores groundedness / completeness / clarity (0–1 each)
  3. If aggregate score < threshold: inject critique and retry once
  4. Return the better of the two answers

The streaming path delegates directly to the inner agent (avoids doubling
latency for real-time token output; reflection runs in the non-streaming path).
"""
import logging
from typing import Any, Dict, Generator, List

from src.agents.base_agent import AgentResult, BaseAgent

logger = logging.getLogger(__name__)


class ReflectionAgent(BaseAgent):
    """
    Wraps ResearchAgent with a one-shot critique-and-refine loop.

    Uses gpt-4o for generation (via ResearchAgent) and gpt-4o-mini
    for evaluation (via CriticAgent) — cost-efficient quality improvement.
    """

    name = "ReflectionAgent"
    system_prompt = ""    # inner agent owns its prompt
    tool_schemas = []     # inner agent owns its tools

    def __init__(self):
        super().__init__()  # sets self.client and self.tool_call_listener
        from src.agents.research_agent import ResearchAgent
        from src.agents.critic_agent import CriticAgent
        self._inner = ResearchAgent()
        self._critic = CriticAgent()

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
            f"[ReflectionAgent] agg={crit.aggregate:.2f} < threshold — "
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

    def stream(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> Generator:
        """
        Streaming path: delegate directly to inner agent for low latency.
        Reflection runs on the non-streaming path only.
        """
        yield from self._inner.stream(message, conversation_history, context)
