"""Token limits and rolling-summary retention on long conversations."""

import json
import unittest
from types import SimpleNamespace

from src.agents.context_budget import CONTEXT_LIMIT, OUTPUT_RESERVE, USER_BUDGET, ContextBudget


class _Redis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, _ttl, value):
        self.values[key] = value


class _Client:
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Earlier user requested citation-heavy answers about RAG."))])


class ContextBudgetTests(unittest.TestCase):
    def test_omitted_history_is_summarized_once_and_user_scoped(self):
        budget = ContextBudget("gpt-4o-mini")
        client, redis = _Client(), _Redis()
        history = [
            {"id": "one", "role": "user", "content": "I want citation-heavy answers about RAG."},
            {"id": "two", "role": "assistant", "content": "Understood."},
            {"id": "three", "role": "user", "content": "What is RRF?"},
        ]
        context = {"user": SimpleNamespace(id=1), "session_id": "s", "redis_client": redis}
        messages = budget.build("System", "profile", history[-1:], history, "Explain RRF", [], context, client, "gpt-4o-mini")
        self.assertIn("Earlier user requested citation-heavy", messages[0]["content"])
        self.assertEqual(client.calls, 1)
        budget.build("System", "profile", history[-1:], history, "Explain RRF", [], context, client, "gpt-4o-mini")
        self.assertEqual(client.calls, 1)
        self.assertIn("chat_summary:1:s", redis.values)

        other = {"user": SimpleNamespace(id=2), "session_id": "s", "redis_client": redis}
        budget.build("System", "profile", history[-1:], history, "Explain RRF", [], other, client, "gpt-4o-mini")
        self.assertEqual(client.calls, 2)

    def test_large_tool_result_stays_valid_json_and_bounded(self):
        budget = ContextBudget("gpt-4o-mini")
        payload = {"results": [{"title": "A", "content": "paper " * 5000}], "count": 1}
        text = budget.fit_tool_result(payload, [{"role": "system", "content": "System"}], [])
        self.assertTrue(json.loads(text)["truncated"])
        self.assertLessEqual(budget.count(text), 1600)

    def test_prompt_reserves_output_tokens(self):
        budget = ContextBudget("gpt-4o-mini")
        messages = budget.build("System", "", [], [], "hello", [], {}, _Client(), "gpt-4o-mini")
        self.assertLessEqual(budget.count_messages(messages) + OUTPUT_RESERVE, CONTEXT_LIMIT)

    def test_current_user_message_is_rejected_instead_of_silently_cut(self):
        budget = ContextBudget("gpt-4o-mini")
        with self.assertRaisesRegex(ValueError, "Please shorten"):
            budget.build("System", "", [], [], "important " * (USER_BUDGET + 1),
                         [], {}, _Client(), "gpt-4o-mini")

    def test_expired_in_process_summary_is_not_reused(self):
        from time import monotonic
        key = "chat_summary:9:expired"
        ContextBudget._fallback[key] = (monotonic() - 1, {"summary": "stale", "covered_ids": []})
        budget = ContextBudget("gpt-4o-mini")
        _, state = budget._summary_state({"user": SimpleNamespace(id=9),
                                          "session_id": "expired", "redis_client": None})
        self.assertEqual(state["summary"], "")


if __name__ == "__main__":
    unittest.main()
