"""
LLM-as-Judge: evaluates agent responses on 4 quality dimensions.

Dimensions (each 0-10):
  relevance     (30%) — Does the answer address the question?
  accuracy      (30%) — Is it factually grounded in sources?
  completeness  (20%) — Does it cover all aspects?
  usefulness    (20%) — Is it actionable and helpful?

overall = weighted average of the four dimensions.
"""
import json
import logging
from dataclasses import dataclass
from typing import Optional

import openai

from src.utils.config import settings

logger = logging.getLogger(__name__)

_WEIGHTS = {"relevance": 0.30, "accuracy": 0.30, "completeness": 0.20, "usefulness": 0.20}

_JUDGE_PROMPT = """\
You are an expert evaluator assessing an AI research assistant's response quality.

Question asked: {question}

Agent response: {response}
{context_block}
Score the response on each dimension (0 = completely fails, 10 = perfect):

- relevance: Does the response directly answer the question?
- accuracy: Is the information correct and grounded in sources (if any were retrieved)?
- completeness: Does it cover all important aspects of the question?
- usefulness: Is it actionable and genuinely helpful to a researcher?

Reply with JSON only:
{{
  "relevance": <0-10>,
  "accuracy": <0-10>,
  "completeness": <0-10>,
  "usefulness": <0-10>,
  "reasoning": "<one sentence explaining your overall assessment>"
}}"""


@dataclass
class JudgeScore:
    relevance: float
    accuracy: float
    completeness: float
    usefulness: float
    overall: float
    reasoning: str

    @classmethod
    def from_dict(cls, data: dict) -> "JudgeScore":
        r = float(data.get("relevance", 5))
        a = float(data.get("accuracy", 5))
        c = float(data.get("completeness", 5))
        u = float(data.get("usefulness", 5))
        overall = r * 0.30 + a * 0.30 + c * 0.20 + u * 0.20
        return cls(
            relevance=r,
            accuracy=a,
            completeness=c,
            usefulness=u,
            overall=round(overall, 2),
            reasoning=data.get("reasoning", ""),
        )

    def to_dict(self) -> dict:
        return {
            "relevance": self.relevance,
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "usefulness": self.usefulness,
            "overall": self.overall,
            "reasoning": self.reasoning,
        }


class LLMJudge:
    """Evaluates a (question, response) pair using a second LLM call."""

    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    def evaluate(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
    ) -> JudgeScore:
        """
        Score a response on 4 quality dimensions.

        Args:
            question: The original user question.
            response: The agent's reply to evaluate.
            context: Optional retrieved context that was available to the agent.
        """
        context_block = (
            f"\nRetrieved context available to the agent:\n{context[:1000]}\n"
            if context
            else ""
        )
        prompt = _JUDGE_PROMPT.format(
            question=question,
            response=response[:2000],
            context_block=context_block,
        )

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return JudgeScore.from_dict(data)
        except Exception as e:
            logger.error(f"LLMJudge evaluation failed: {e}")
            return JudgeScore(
                relevance=5.0,
                accuracy=5.0,
                completeness=5.0,
                usefulness=5.0,
                overall=5.0,
                reasoning=f"Evaluation failed: {e}",
            )
