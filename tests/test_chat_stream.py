"""
End-to-end behaviour of POST /chat/stream with the agent, memory and caches faked.

Pins what the endpoint promises the frontend and the rest of the system: the
event order, that a finished answer is remembered and fed to fact extraction
and metrics, that a failed or cached turn is handled without the agent's
side effects, and that resuming an unknown research run is reported.
"""
import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.deps import get_current_user, get_embedding_manager
from src.api.routers import chat


class _Memory:
    """In-process ConversationMemory stand-in that records what the turn stored."""
    stored: list = []

    def __init__(self, redis_client=None, user_id=None):
        self.user_id = user_id

    @staticmethod
    def new_session_id():
        return "new-session"

    def get_history(self, session_id, max_messages=10):
        return []

    def add_message(self, session_id, role, content):
        _Memory.stored.append((session_id, role, content))


class _Agent:
    name = "FakeAgent"
    system_prompt = "You are fake."

    def __init__(self, events):
        self.events = events
        self.calls = 0

    def stream(self, question, history, context):
        self.calls += 1
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield event


def _events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


class ChatStreamTests(unittest.TestCase):
    def setUp(self):
        _Memory.stored = []
        self.app = FastAPI()
        self.app.include_router(chat.router)
        self.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7)
        embedding = SimpleNamespace(model=None, collection=SimpleNamespace(count=lambda: 3),
                                    generate_embedding=lambda text: [0.0])
        self.app.dependency_overrides[get_embedding_manager] = lambda: embedding
        self.extracted, self.metrics = [], []
        self.stack = ExitStack()
        for target, value in [
            ("src.api.routers.chat.get_redis", lambda: None),
            ("src.api.routers.chat.ConversationMemory", _Memory),
            ("src.api.routers.chat.ToolAwareAgent", lambda agent, **_: agent),
            ("src.api.routers.chat._extract_facts_bg", lambda *args: self.extracted.append(args)),
            ("src.api.routers.chat.record_request", lambda **kw: self.metrics.append(kw)),
            ("src.rag.retriever.Retriever", MagicMock()),
            ("src.database.models.SessionLocal", MagicMock()),
        ]:
            self.stack.enter_context(patch(target, value))
        self.addCleanup(self.stack.close)

    def _route_to(self, agent):
        router = SimpleNamespace(route_stream=lambda question: (agent, "research_qa", "keyword", 0.9))
        self.stack.enter_context(patch.object(chat, "_get_agent_router", return_value=router))

    def _post(self, **body):
        with TestClient(self.app) as client:
            return _events(client.post("/chat/stream", json={"message": "What is LoRA?", **body}))

    def test_answer_is_streamed_remembered_and_measured(self):
        self._route_to(_Agent([
            {"type": "tool_call", "tool": "search_knowledge_base"},
            {"type": "token", "value": "LoRA adds "},
            {"type": "token", "value": "low-rank adapters."},
            {"type": "done", "tools_called": ["search_knowledge_base"], "citations": [{"title": "LoRA"}]},
        ]))
        events = self._post(session_id="s1")
        self.assertEqual([e["type"] for e in events],
                         ["session", "intent", "agent", "tool_call", "token", "token", "done"])
        self.assertEqual(events[0]["session_id"], "s1")
        self.assertEqual(events[1], {"type": "intent", "value": "research_qa", "method": "keyword", "confidence": 0.9})
        self.assertEqual(_Memory.stored, [("s1", "user", "What is LoRA?"),
                                          ("s1", "assistant", "LoRA adds low-rank adapters.")])
        self.assertEqual(self.extracted, [(7, "What is LoRA?", "LoRA adds low-rank adapters.", None)])
        self.assertEqual(self.metrics[0]["tools_called"], ["search_knowledge_base"])
        self.assertEqual(self.metrics[0]["agent"], "FakeAgent")

    def test_replace_event_overrides_the_streamed_draft(self):
        self._route_to(_Agent([
            {"type": "draft_token", "value": "rough"},
            {"type": "replace", "value": "polished"},
            {"type": "done"},
        ]))
        self._post(session_id="s1")
        self.assertEqual(_Memory.stored[-1], ("s1", "assistant", "polished"))

    def test_agent_failure_is_reported_and_nothing_is_remembered(self):
        self._route_to(_Agent([{"type": "token", "value": "partial"}, RuntimeError("model down")]))
        events = self._post(session_id="s1")
        self.assertEqual(events[-1], {"type": "error", "value": "model down"})
        self.assertEqual(_Memory.stored, [])
        self.assertEqual(self.extracted, [])

    def test_unfinished_answer_is_not_remembered(self):
        self._route_to(_Agent([{"type": "token", "value": "no done event"}]))
        self._post(session_id="s1")
        self.assertEqual(_Memory.stored, [])
        self.assertEqual(len(self.metrics), 1)

    def test_cached_answer_is_replayed_without_running_the_agent(self):
        agent = _Agent([{"type": "done"}])
        self._route_to(agent)
        cache = MagicMock()
        cache.get.return_value = {"reply": "cached reply", "citations": [{"title": "C"}]}
        self.stack.enter_context(patch.object(chat, "get_redis", return_value=MagicMock()))
        self.stack.enter_context(patch.object(chat.settings, "ANSWER_CACHE_ENABLED", True))
        self.stack.enter_context(patch("src.rag.answer_cache.eligible_answer", return_value=True))
        self.stack.enter_context(patch("src.rag.answer_cache.SemanticAnswerCache", return_value=cache))
        self.stack.enter_context(patch.object(chat._fact_memory, "get_context_string", return_value=""))
        events = self._post(session_id="s1")
        self.assertEqual(agent.calls, 0)
        self.assertEqual(events[-2], {"type": "token", "value": "cached reply"})
        self.assertEqual(events[-1]["type"], "done")
        self.assertTrue(events[-1]["cache_hit"])
        cache.put.assert_not_called()
        self.assertEqual(_Memory.stored[-1], ("s1", "assistant", "cached reply"))

    def test_fresh_answer_is_offered_to_the_cache(self):
        self._route_to(_Agent([{"type": "token", "value": "fresh"}, {"type": "done", "citations": []}]))
        cache = MagicMock()
        cache.get.return_value = None
        self.stack.enter_context(patch.object(chat, "get_redis", return_value=MagicMock()))
        self.stack.enter_context(patch.object(chat.settings, "ANSWER_CACHE_ENABLED", True))
        self.stack.enter_context(patch("src.rag.answer_cache.eligible_answer", return_value=True))
        self.stack.enter_context(patch("src.rag.answer_cache.cacheable_result", return_value=True))
        self.stack.enter_context(patch("src.rag.answer_cache.SemanticAnswerCache", return_value=cache))
        self.stack.enter_context(patch.object(chat._fact_memory, "get_context_string", return_value=""))
        self._post(session_id="s1")
        cache.put.assert_called_once()
        self.assertEqual(cache.put.call_args.args[2], {"reply": "fresh", "citations": []})

    def test_resuming_an_unknown_research_run_reports_an_error(self):
        store = MagicMock()
        store.return_value.load.return_value = None
        self.stack.enter_context(patch("src.agents.research_run_store.ResearchRunStore", store))
        events = self._post(message="", resume_run_id="missing")
        self.assertEqual(events[-1], {"type": "error", "value": "Research run not found for this user"})

    def test_empty_message_is_rejected(self):
        with TestClient(self.app) as client:
            self.assertEqual(client.post("/chat/stream", json={"message": "  "}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
