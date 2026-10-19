"""Structured model output: validate, repair once, then fail cleanly so the caller falls back."""
import json
import unittest
from types import SimpleNamespace
from typing import List

from pydantic import BaseModel, field_validator

from src.agents.base_agent import BaseAgent
from src.agents.coding_practice import PracticeError
from src.agents.coding_services import CodingServices
from src.agents.core import structured
from src.agents.core.structured import StructuredOutputError, structured_call


class Verdict(BaseModel):
    label: str
    score: float
    reasons: List[str]

    @field_validator("score")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return max(0.0, min(1.0, value))


def _reply(content, finish_reason="stop", refusal=None):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason,
                                                    message=SimpleNamespace(content=content, refusal=refusal))])


class _Client:
    """Replies in order and keeps every request it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)


def _count(task, outcome):
    return structured.STRUCTURED_OUTPUTS.labels(task=task, outcome=outcome)._value.get()


GOOD = json.dumps({"label": "ok", "score": 3, "reasons": ["clear"]})
MESSAGES = [{"role": "user", "content": "judge this"}]


class StructuredCallTests(unittest.TestCase):
    def call(self, client, task):
        return structured_call(client, Verdict, task=task, model="m", max_tokens=100, messages=MESSAGES)

    def test_valid_reply_is_validated_and_asked_for_with_a_strict_schema(self):
        client = _Client(_reply(GOOD))
        before = _count("t_ok", "ok")
        verdict = self.call(client, "t_ok")
        self.assertEqual((verdict.label, verdict.score), ("ok", 1.0))  # validator clamps
        fmt = client.requests[0]["response_format"]
        self.assertEqual((fmt["type"], fmt["json_schema"]["strict"]), ("json_schema", True))
        self.assertFalse(fmt["json_schema"]["schema"]["additionalProperties"])
        self.assertEqual(_count("t_ok", "ok"), before + 1)

    def test_invalid_reply_is_shown_back_once_and_repaired(self):
        client = _Client(_reply('{"label": "ok"}'), _reply(GOOD))
        before = _count("t_repair", "repaired")
        self.assertEqual(self.call(client, "t_repair").label, "ok")
        repair = client.requests[1]["messages"]
        self.assertEqual(repair[:1], MESSAGES)
        self.assertEqual(repair[1], {"role": "assistant", "content": '{"label": "ok"}'})
        self.assertIn("score", repair[2]["content"])  # the validation error names the missing field
        self.assertEqual(_count("t_repair", "repaired"), before + 1)

    def test_cut_off_reply_is_repaired(self):
        client = _Client(_reply('{"label": "o', finish_reason="length"), _reply(GOOD))
        self.assertEqual(self.call(client, "t_cut").label, "ok")
        self.assertIn("cut off", client.requests[1]["messages"][-1]["content"])

    def test_still_invalid_after_the_repair_raises_for_the_callers_fallback(self):
        client = _Client(_reply("not json"), _reply("```json\n{}```"))
        before = _count("t_fail", "failed")
        with self.assertRaises(StructuredOutputError):
            self.call(client, "t_fail")
        self.assertEqual(len(client.requests), 2)  # bounded: one repair, no more
        self.assertEqual(_count("t_fail", "failed"), before + 1)

    def test_refusal_is_not_repaired(self):
        client = _Client(_reply(None, refusal="I can't help with that"))
        with self.assertRaises(StructuredOutputError):
            self.call(client, "t_refuse")
        self.assertEqual(len(client.requests), 1)


class CodingAgentJsonTests(unittest.TestCase):
    def services(self, *replies):
        services = CodingServices.__new__(CodingServices)
        services.client = _Client(*replies)
        services.input_tokens = services.output_tokens = 0
        return services

    def test_unparseable_reply_gets_one_repair_turn(self):
        services = self.services(_reply("Sure! Here is the JSON:"), _reply('{"code": "print(1)"}'))
        self.assertEqual(services._json("sys", "user"), {"code": "print(1)"})
        self.assertIn("not a single valid JSON object", services.client.requests[1]["messages"][-1]["content"])

    def test_two_bad_replies_fail_the_step_cleanly(self):
        services = self.services(_reply("nope"), _reply("[1, 2]"))
        with self.assertRaises(PracticeError):
            services._json("sys", "user")


class MalformedToolArgumentsTests(unittest.TestCase):
    def test_tool_is_not_run_and_the_model_is_told(self):
        agent = BaseAgent.__new__(BaseAgent)
        agent.name = "Test"
        agent.tool_schemas = [{"function": {"name": "search_knowledge_base"}}]
        executed = []
        agent.registry = SimpleNamespace(execute=lambda *args: executed.append(args) or {})
        agent.tool_call_listener = None
        agent.context_budget = SimpleNamespace(fit_tool_result=lambda result, *_: json.dumps(result))
        messages = []
        call = SimpleNamespace(id="c1", function=SimpleNamespace(name="search_knowledge_base", arguments='{"query":'))
        result = agent._dispatch_tool_call(call, {}, [], [], messages)
        self.assertEqual(executed, [])
        self.assertIn("not valid JSON", result["error"])
        self.assertIn("not valid JSON", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
