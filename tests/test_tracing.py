"""LangSmith tracing: off by default, wraps the OpenAI client when on, and never ships the request context."""
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import openai
from langsmith import tracing_context

from src.utils.llm_usage import AccountedClient
from src.agents.core.tool_registry import ToolRegistry
from src.utils import tracing
from src.utils.config import settings


class TracingSwitchTests(unittest.TestCase):
    def test_off_by_default_returns_a_plain_client(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "", "LANGSMITH_API_KEY": ""}), \
             patch("langsmith.wrappers.wrap_openai") as wrap:
            client = tracing.openai_client()
        self.assertIsInstance(client, AccountedClient)  # usage and budget accounting, always on
        self.assertIsInstance(client._inner, openai.OpenAI)
        wrap.assert_not_called()

    def test_on_wraps_every_client_for_langsmith(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "ls-test"}), \
             patch("langsmith.wrappers.wrap_openai",
                   side_effect=lambda c: SimpleNamespace(tag="wrapped", chat=c.chat)) as wrap:
            client = tracing.openai_client(timeout=5)
        self.assertEqual(client._inner.tag, "wrapped")  # accounting sits outside the LangSmith wrapper
        wrap.assert_called_once()

    def test_settings_are_exported_only_with_a_key(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.multiple(settings, LANGSMITH_TRACING=True, LANGSMITH_API_KEY=None):
            tracing._export_settings()
            self.assertNotIn("LANGSMITH_TRACING", os.environ)
        with patch.dict(os.environ, {}, clear=True), \
             patch.multiple(settings, LANGSMITH_TRACING=True, LANGSMITH_API_KEY="ls-test",
                            LANGSMITH_PROJECT="researchmate", LANGSMITH_HIDE_INPUTS=True):
            tracing._export_settings()
            self.assertEqual((os.environ["LANGSMITH_TRACING"], os.environ["LANGSMITH_PROJECT"],
                              os.environ["LANGSMITH_HIDE_INPUTS"]), ("true", "researchmate", "true"))

    def test_stream_summary_keeps_the_answer_not_every_token(self):
        events = [{"type": "token", "value": "Hello "}, {"type": "token", "value": "world"},
                  {"type": "done", "tools_called": ["search_knowledge_base"], "citations": [{}, {}]}]
        self.assertEqual(tracing.stream_summary(events), {"answer": "Hello world", "events": 3,
                                                         "tools_called": ["search_knowledge_base"], "citations": 2})


class TracedPayloadTests(unittest.TestCase):
    def test_tool_runs_carry_name_and_arguments_but_not_the_request_context(self):
        registry = ToolRegistry().register(
            {"type": "function", "function": {"name": "echo", "parameters": {"type": "object", "properties": {}}}},
            lambda args, context: {"echo": args["text"]})
        context = {"db": object(), "user": SimpleNamespace(id=7, email="private@example.com"), "secret": "s3cr3t"}
        langsmith_client = MagicMock()
        with tracing_context(enabled=True, client=langsmith_client, project_name="test"):
            result = registry.execute("echo", {"text": "hi"}, context)
        self.assertEqual(result["echo"], "hi")
        sent = json.dumps([str(call) for call in langsmith_client.mock_calls])
        self.assertIn("echo", sent)
        self.assertIn('hi', sent)
        self.assertNotIn("private@example.com", sent)
        self.assertNotIn("s3cr3t", sent)

    def test_runs_inside_a_tool_nest_under_the_tool_despite_the_worker_thread(self):
        from langsmith import traceable

        @traceable(name="inner_retrieve")
        def inner(query):
            return query

        registry = ToolRegistry().register(
            {"type": "function", "function": {"name": "kb", "parameters": {"type": "object", "properties": {}}}},
            lambda args, context: {"hits": inner(args["q"])})
        langsmith_client = MagicMock()
        with tracing_context(enabled=True, client=langsmith_client, project_name="test"):
            registry.execute("kb", {"q": "lora"}, {})
        posted = {str(c.kwargs.get("id")): c.kwargs for c in langsmith_client.mock_calls if c.kwargs.get("name")}
        inner_run = next(run for run in posted.values() if run["name"] == "inner_retrieve")
        self.assertEqual(posted[str(inner_run["parent_run_id"])]["name"], "tool")


if __name__ == "__main__":
    unittest.main()
