"""Model usage accounting, per-request token budgets, model routing and fallback."""
import contextvars
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai

from src.utils import llm_usage
from src.utils.llm_usage import (AccountedClient, BudgetExceeded, budget_allows, current_budget, model_for,
                                 price, request_budget)


def _usage(prompt, completion):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)


def _error(cls, status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    return cls("provider trouble", response=response, body=None)


class _Completions:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client(*outcomes):
    completions = _Completions(*outcomes)
    return AccountedClient(SimpleNamespace(chat=SimpleNamespace(completions=completions))), completions


class PricingAndRoutingTests(unittest.TestCase):
    def test_dated_snapshots_get_their_familys_price(self):
        self.assertEqual(price("gpt-4o-mini-2024-07-18"), (0.15, 0.075, 0.60))  # not gpt-4o's
        self.assertEqual(price("gpt-4o-2024-08-06"), (2.50, 1.25, 10.00))
        self.assertEqual(price("some-new-model"), (0.0, 0.0, 0.0))

    def test_cached_input_is_priced_at_the_cached_rate(self):
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=0,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=400))
        self.assertAlmostEqual(llm_usage.cost_usd("gpt-4o-mini", usage), (600 * 0.15 + 400 * 0.075) / 1e6)

    def test_routes_override_the_default_model(self):
        with patch.object(llm_usage.settings, "MODEL_ROUTES", '{"intent": "gpt-4.1-nano"}'), \
                patch.object(llm_usage.settings, "LLM_MODEL", "gpt-4o-mini"):
            self.assertEqual(model_for("intent"), "gpt-4.1-nano")
            self.assertEqual(model_for("critic"), "gpt-4o-mini")
            self.assertEqual(model_for("planner", "gpt-4o"), "gpt-4o")
        with patch.object(llm_usage.settings, "MODEL_ROUTES", "not json"):
            self.assertEqual(model_for("intent", "gpt-4o-mini"), "gpt-4o-mini")  # ignored, not fatal


class BudgetTests(unittest.TestCase):
    def test_calls_are_charged_to_the_open_budget_and_refused_once_it_is_spent(self):
        client, completions = _client(SimpleNamespace(usage=_usage(900, 100)), SimpleNamespace(usage=_usage(1, 1)))
        with request_budget("test", max_tokens=1000) as budget:
            client.chat.completions.create(model="gpt-4o-mini", messages=[])
            self.assertEqual(budget.used_tokens, 1000)
            self.assertAlmostEqual(budget.cost_usd, (900 * 0.15 + 100 * 0.60) / 1e6)
            with self.assertRaises(BudgetExceeded):
                client.chat.completions.create(model="gpt-4o-mini", messages=[])
        self.assertEqual(len(completions.requests), 1)  # the refused call never reached the provider

    def test_reserve_check_and_no_budget_outside_a_request(self):
        self.assertIsNone(current_budget())
        self.assertTrue(budget_allows(10**9))
        with request_budget("test", max_tokens=100) as budget:
            budget.charge(60, 0.0)
            self.assertTrue(budget_allows(39))
            self.assertFalse(budget_allows(40))
        self.assertIsNone(current_budget())

    def test_threads_started_with_a_copied_context_share_the_budget(self):
        client, _ = _client(SimpleNamespace(usage=_usage(10, 5)))
        with request_budget("test", max_tokens=1000) as budget:
            worker = threading.Thread(target=contextvars.copy_context().run,
                                      args=(client.chat.completions.create,), kwargs={"model": "gpt-4o-mini"})
            worker.start()
            worker.join()
        self.assertEqual(budget.used_tokens, 15)


class StreamTests(unittest.TestCase):
    def test_streamed_usage_is_recorded_and_its_chunk_hidden(self):
        chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))], usage=None),
                  SimpleNamespace(choices=[], usage=_usage(20, 2))]
        client, completions = _client(iter(chunks))
        with request_budget("test", max_tokens=1000) as budget:
            stream = client.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True)
            received = list(stream)
        self.assertEqual(completions.requests[0]["stream_options"], {"include_usage": True})
        self.assertEqual(len(received), 1)  # callers never see a chunk without choices
        self.assertEqual(budget.used_tokens, 22)


class FallbackTests(unittest.TestCase):
    def test_provider_outage_is_retried_once_on_the_fallback_model(self):
        client, completions = _client(_error(openai.InternalServerError, 500), SimpleNamespace(usage=None))
        with patch.object(llm_usage.settings, "LLM_FALLBACK_MODEL", "gpt-4.1-mini"):
            client.chat.completions.create(model="gpt-4o-mini", messages=[])
        self.assertEqual([r["model"] for r in completions.requests], ["gpt-4o-mini", "gpt-4.1-mini"])

    def test_no_fallback_when_unset_or_when_the_request_itself_is_wrong(self):
        client, _ = _client(_error(openai.RateLimitError, 429))
        with patch.object(llm_usage.settings, "LLM_FALLBACK_MODEL", None), self.assertRaises(openai.RateLimitError):
            client.chat.completions.create(model="gpt-4o-mini", messages=[])
        client, completions = _client(_error(openai.BadRequestError, 400))
        with patch.object(llm_usage.settings, "LLM_FALLBACK_MODEL", "gpt-4.1-mini"), \
                self.assertRaises(openai.BadRequestError):
            client.chat.completions.create(model="gpt-4o-mini", messages=[])
        self.assertEqual(len(completions.requests), 1)


class ReActBudgetTests(unittest.TestCase):
    def test_low_budget_takes_the_tools_away_so_the_next_call_answers(self):
        from src.agents.base_agent import BaseAgent
        agent = BaseAgent.__new__(BaseAgent)
        agent.name = "Test"
        agent.context_budget = SimpleNamespace(count_messages=lambda messages: 500)
        kwargs = {"tools": [{}], "tool_choice": "auto"}
        messages = []
        with request_budget("test", max_tokens=100_000):
            agent._answer_now_if_budget_low(kwargs, messages)
        self.assertIn("tools", kwargs)  # plenty left: keep searching
        with request_budget("test", max_tokens=2_000):
            agent._answer_now_if_budget_low(kwargs, messages)
        self.assertNotIn("tools", kwargs)
        self.assertIn("do not call tools", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
