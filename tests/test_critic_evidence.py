"""The quality judge must read source text, not just citation titles."""

import json
import unittest
from types import SimpleNamespace

from src.agents.critic_agent import CriticAgent, CriticResult
from src.evaluation.llm_judge import LLMJudge


class _Completion:
    def __init__(self):
        self.prompt = ""

    def create(self, **kwargs):
        self.prompt = "\n".join(message["content"] for message in kwargs["messages"])
        payload = json.dumps({"groundedness": 0.9, "completeness": 0.8, "clarity": 0.8, "critique": ""})
        call = SimpleNamespace(function=SimpleNamespace(arguments=payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))])


class CriticEvidenceTests(unittest.TestCase):
    def test_critic_receives_actual_excerpt(self):
        completion = _Completion()
        critic = CriticAgent.__new__(CriticAgent)
        critic.client = SimpleNamespace(chat=SimpleNamespace(completions=completion))
        critic.threshold = 0.7

        result = critic.evaluate(
            "What did the paper find?", "It found a benefit.",
            citations=[{"title": "A Paper", "evidence": "The measured benefit was 12 percent."}],
        )
        self.assertEqual(result.groundedness, 0.9)
        self.assertIn("The measured benefit was 12 percent.", completion.prompt)
        self.assertIn("<untrusted_source", completion.prompt)

    def test_llm_judge_passes_retrieved_context(self):
        class _Critic:
            def evaluate(self, *_args, **kwargs):
                self.context = kwargs["context"]
                return CriticResult(groundedness=0.8, completeness=0.8, clarity=0.8, critique="", should_retry=False)

        judge = LLMJudge.__new__(LLMJudge)
        judge._critic = _Critic()
        judge.evaluate("question", "answer", context="retrieved paragraph")
        self.assertEqual(judge._critic.context, "retrieved paragraph")


if __name__ == "__main__":
    unittest.main()
