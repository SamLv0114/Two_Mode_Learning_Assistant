"""
LLM-as-Judge: evaluates agent responses on 4 quality dimensions.

Now backed by CriticAgent as the shared evaluation engine.
CriticAgent's structured 3-dimension result (groundedness/completeness/clarity, 0-1)
is converted to the 4-dimension 0-10 JudgeScore that the eval API expects:

  groundedness  → accuracy   (factual grounding)
  completeness  → completeness
  clarity       → usefulness  (well-written = more actionable)
  relevance     = (groundedness + completeness) / 2  (on-topic + comprehensive ≈ relevant)

Using CriticAgent as the backend means both inline quality control (ReflectionAgent)
and batch evaluation (eval endpoint) share the same evaluation logic and LLM call.
The critique string produced by CriticAgent is surfaced as the reasoning field.
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

from src.agents.critic_agent import CriticAgent

logger = logging.getLogger(__name__)


@dataclass
class JudgeScore:
    relevance: float       # 0-10
    accuracy: float        # 0-10
    completeness: float    # 0-10
    usefulness: float      # 0-10
    overall: float         # weighted average
    reasoning: str         # critique from CriticAgent

    @property
    def groundedness(self) -> float:
        """Alias: accuracy maps to groundedness in CriticAgent terms."""
        return self.accuracy / 10.0

    def to_dict(self) -> Dict:
        return {
            "relevance": round(self.relevance, 1),
            "accuracy": round(self.accuracy, 1),
            "completeness": round(self.completeness, 1),
            "usefulness": round(self.usefulness, 1),
            "overall": round(self.overall, 2),
            "reasoning": self.reasoning,
        }


class LLMJudge:
    """
    Evaluates a (question, response) pair using CriticAgent as the scoring engine.

    Dimension mapping from CriticAgent (0-1) to JudgeScore (0-10):
      groundedness  → accuracy      × 10
      completeness  → completeness  × 10
      clarity       → usefulness    × 10
      relevance     = avg(groundedness, completeness) × 10

    overall = relevance×0.30 + accuracy×0.30 + completeness×0.20 + usefulness×0.20
    """

    _WEIGHTS = {
        "relevance": 0.30,
        "accuracy": 0.30,
        "completeness": 0.20,
        "usefulness": 0.20,
    }

    def __init__(self):
        self._critic = CriticAgent()

    def evaluate(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
        citations: Optional[List[Dict]] = None,
    ) -> JudgeScore:
        """
        Score a response using CriticAgent and return a JudgeScore.

        Args:
            question:  The original user question.
            response:  The agent's reply to evaluate.
            context:   (legacy) retrieved context string — unused but kept for compat.
            citations: List of citation dicts passed through to CriticAgent.
        """
        crit = self._critic.evaluate(question, response, citations=citations)

        # Convert 0-1 → 0-10
        accuracy = round(crit.groundedness * 10, 1)
        completeness = round(crit.completeness * 10, 1)
        usefulness = round(crit.clarity * 10, 1)
        # Relevance: on-topic (groundedness) + covers everything (completeness)
        relevance = round(((crit.groundedness + crit.completeness) / 2) * 10, 1)

        overall = round(
            relevance    * self._WEIGHTS["relevance"]
            + accuracy   * self._WEIGHTS["accuracy"]
            + completeness * self._WEIGHTS["completeness"]
            + usefulness * self._WEIGHTS["usefulness"],
            2,
        )

        return JudgeScore(
            relevance=relevance,
            accuracy=accuracy,
            completeness=completeness,
            usefulness=usefulness,
            overall=overall,
            reasoning=crit.critique or f"Aggregate score: {crit.aggregate:.2f}",
        )
