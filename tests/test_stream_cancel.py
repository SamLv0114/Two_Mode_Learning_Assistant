"""Real SDK chunks and cooperative cancellation in BaseAgent.stream."""

import threading
import unittest
from unittest.mock import patch
from html import unescape
from types import SimpleNamespace

from src.agents.base_agent import BaseAgent
from src.agents.context_budget import ContextBudget


def _chunk(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))])


def _tool(index, id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))


class _Stream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        self.closed = True


class _Completions:
    def __init__(self, streams):
        self.streams = streams
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.streams.pop(0)


class StreamCancelTests(unittest.TestCase):
    def _agent(self, streams):
        agent = BaseAgent.__new__(BaseAgent)
        agent.name = "TestAgent"
        agent.system_prompt = "System"
        agent.tool_schemas = [{"type": "function", "function": {"name": "search_knowledge_base"}}]
        agent.context_budget = ContextBudget("gpt-4o-mini")
        agent.tool_call_listener = None
        completions = _Completions(streams)
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        agent.registry = SimpleNamespace(execute=lambda _name, _args, _context: {"results": [], "count": 0})
        return agent, completions

    def test_stream_forwards_chunks_and_reassembles_tool_call(self):
        first = _Stream([_chunk(tool_calls=[_tool(0, "call1", "search_knowledge_base", '{"query":')]),
                         _chunk(tool_calls=[_tool(0, arguments='"RAG"}')])])
        second = _Stream([_chunk(content="Hello"), _chunk(content=" world")])
        agent, completions = self._agent([first, second])
        events = list(agent.stream("hello", [], {}))
        self.assertEqual([e["value"] for e in events if e["type"] == "token"], ["Hello", " world"])
        self.assertEqual([e["tool"] for e in events if e["type"] == "tool_call"], ["search_knowledge_base"])
        self.assertTrue(all(call["stream"] for call in completions.calls))
        self.assertIn('"count": 0', unescape(completions.calls[1]["messages"][-1]["content"]))
        self.assertIn("<untrusted_source", completions.calls[1]["messages"][-1]["content"])
        self.assertTrue(first.closed and second.closed)

    def test_cancel_closes_model_stream_before_more_chunks(self):
        stream = _Stream([_chunk(content="first"), _chunk(content="second")])
        agent, _ = self._agent([stream])
        event = threading.Event()
        generator = agent.stream("hello", [], {"_cancel_event": event})
        self.assertEqual(next(generator)["type"], "generating")
        self.assertEqual(next(generator), {"type": "token", "value": "first"})
        event.set()
        self.assertEqual(next(generator)["type"], "cancelled")
        with self.assertRaises(StopIteration):
            next(generator)
        self.assertTrue(stream.closed)

    def test_tool_limit_returns_observed_evidence(self):
        stream = _Stream([_chunk(tool_calls=[_tool(0, "call1", "search_knowledge_base", '{"query":"RAG"}')])])
        agent, _ = self._agent([stream])
        agent.registry = SimpleNamespace(execute=lambda *_: {
            "results": [{"title": "Paper A", "content": "Finding A", "url": "https://example.org/a"}],
            "count": 1,
        })
        with patch("src.agents.base_agent.MAX_TOOL_ITERATIONS", 1):
            events = list(agent.stream("question", [], {}))
        self.assertIn("Finding A", "".join(e["value"] for e in events if e["type"] == "token"))
        self.assertEqual(events[-1]["stop_reason"], "tool_step_limit")

    def test_context_budget_stops_before_another_model_call(self):
        stream = _Stream([_chunk(tool_calls=[_tool(0, "call1", "search_knowledge_base", '{"query":"RAG"}')])])
        agent, completions = self._agent([stream])
        agent.context_budget.remaining = lambda *_: 0
        events = list(agent.stream("question", [], {}))
        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(events[-1]["stop_reason"], "context_budget")


if __name__ == "__main__":
    unittest.main()
