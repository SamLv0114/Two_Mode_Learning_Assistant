"""
CriticAgent — LLM-as-judge for evaluating Q&A quality.

Implements the Reflection evaluation step from Hello-Agents ch4.4.
Scores an answer on groundedness, completeness, and clarity using gpt-4o-mini.
Returned CriticResult.should_retry drives the ReflectionAgent's retry decision.
"""
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI

from src.utils.config import settings

logger = logging.getLogger(__name__)

CRITIC_MODEL = "gpt-4o-mini"
DEFAULT_THRESHOLD = 0.70


@dataclass
class CriticResult:
    groundedness: float   # 0–1: is the answer grounded in cited sources?
    completeness: float   # 0–1: does it address all parts of the question?
    clarity: float        # 0–1: is it well-structured and precise?
    critique: str         # one sentence of specific, actionable feedback
    should_retry: bool    # True when aggregate < threshold

    @property
    def aggregate(self) -> float:
        return (self.groundedness + self.completeness + self.clarity) / 3

    def to_dict(self) -> Dict:
        return {
            "groundedness": round(self.groundedness, 3),
            "completeness": round(self.completeness, 3),
            "clarity": round(self.clarity, 3),
            "aggregate": round(self.aggregate, 3),
            "critique": self.critique,
            "should_retry": self.should_retry,
        }


class CriticAgent:
    """
    Scores a (question, answer, citations) triple using gpt-4o-mini.

    Used by ReflectionAgent to decide whether to retry, and optionally
    by the router for non-streaming quality validation.
    """

    _PROMPT = """\
You are an expert evaluator for a machine learning research assistant.

Score the following question-answer pair on three dimensions (each 0.0–1.0):

- groundedness: Is the answer factually supported by the cited sources?
  Penalise hallucination or claims with no source backing.
- completeness: Does the answer address ALL parts of the question?
  Penalise vague or partial responses.
- clarity: Is the answer well-structured, concise, and easy to understand?
  Penalise rambling, jargon without explanation, or poor organisation.

Question: {question}

Answer: {answer}

Sources cited: {citations}

Respond with a JSON object ONLY (no markdown, no explanation outside JSON):
{{
  "groundedness": <float 0-1>,
  "completeness": <float 0-1>,
  "clarity": <float 0-1>,
  "critique": "<one sentence of specific, actionable improvement feedback>"
}}"""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.threshold = threshold

    def evaluate(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        threshold: Optional[float] = None,
    ) -> CriticResult:
        """
        Score the answer. Safe — always returns a CriticResult even on API error.
        """
        thr = threshold if threshold is not None else self.threshold
        citation_str = (
            ", ".join(c.get("title", c.get("url", "")) for c in citations)
            if citations else "none"
        )

        prompt = self._PROMPT.format(
            question=question[:600],
            answer=answer[:2000],
            citations=citation_str[:400],
        )

        try:
            response = self.client.chat.completions.create(
                model=CRITIC_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            g = max(0.0, min(1.0, float(data.get("groundedness", 0.5))))
            co = max(0.0, min(1.0, float(data.get("completeness", 0.5))))
            cl = max(0.0, min(1.0, float(data.get("clarity", 0.5))))
            critique = str(data.get("critique", ""))
            agg = (g + co + cl) / 3
            result = CriticResult(
                groundedness=g,
                completeness=co,
                clarity=cl,
                critique=critique,
                should_retry=agg < thr,
            )
            logger.info(
                f"[CriticAgent] G={g:.2f} Co={co:.2f} Cl={cl:.2f} "
                f"agg={agg:.2f} retry={result.should_retry}"
            )
            return result
        except Exception as e:
            logger.warning(f"CriticAgent evaluation failed: {e}")
            return CriticResult(
                groundedness=0.5,
                completeness=0.5,
                clarity=0.5,
                critique="",
                should_retry=False,
            )
