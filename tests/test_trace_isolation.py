"""Trace ownership and concurrency regressions."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import Depends, FastAPI, Header
from fastapi.testclient import TestClient

from src.agents.base_agent import AgentResult, BaseAgent
from src.agents.tool_aware_agent import ToolAwareAgent
from src.api.deps import get_current_user
from src.api.routers import chat


class _Redis:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()

    def pipeline(self, transaction=True):
        assert transaction
        return _Pipeline(self)

    def lrange(self, key, start, stop):
        with self.lock:
            values = self.data.get(key, [])
            return list(values[start:] if stop == -1 else values[start:stop + 1])


class _Pipeline:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def rpush(self, *args):
        self.commands.append(("rpush", args))
        return self

    def ltrim(self, *args):
        self.commands.append(("ltrim", args))
        return self

    def expire(self, *args):
        self.commands.append(("expire", args))
        return self

    def execute(self):
        with self.redis.lock:
            for command, args in self.commands:
                if command == "rpush":
                    key, value = args
                    self.redis.data.setdefault(key, []).append(value)
                elif command == "ltrim":
                    key, start, stop = args
                    self.redis.data[key] = self.redis.data[key][start:] if stop == -1 else self.redis.data[key][start:stop + 1]


class TraceIsolationTests(unittest.TestCase):
    def test_shared_agent_uses_request_local_listener(self):
        redis = _Redis()
        barrier = threading.Barrier(2)
        agent = BaseAgent.__new__(BaseAgent)
        agent.name = "SharedAgent"
        agent.tool_call_listener = None

        def run(_message, _history, context):
            barrier.wait(timeout=5)
            user_id = context["user_id"]
            agent._fire_listener("search", {"user_id": user_id}, {"count": 1}, 1, context)
            return AgentResult(reply="ok")

        agent.run = run
        a = ToolAwareAgent(agent, user_id=1, session_id="same", redis_client=redis)
        b = ToolAwareAgent(agent, user_id=2, session_id="same", redis_client=redis)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(a.run, "a", [], {"user_id": 1})
            second = pool.submit(b.run, "b", [], {"user_id": 2})
            self.assertEqual(first.result().reply, "ok")
            self.assertEqual(second.result().reply, "ok")

        self.assertIsNone(agent.tool_call_listener)
        self.assertEqual(ToolAwareAgent.load_trace(1, "same", redis)[0]["args"]["user_id"], "1")
        self.assertEqual(ToolAwareAgent.load_trace(2, "same", redis)[0]["args"]["user_id"], "2")

    def test_concurrent_appends_do_not_lose_entries(self):
        redis = _Redis()
        agent = SimpleNamespace(name="SharedAgent")
        aware = ToolAwareAgent(agent, user_id=1, session_id="session", redis_client=redis)
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda i: aware._on_tool_call("search", {"i": i}, {}, 1), range(80)))
        self.assertEqual(len(ToolAwareAgent.load_trace(1, "session", redis)), 80)

    def test_trace_endpoint_is_scoped_to_current_user(self):
        redis = _Redis()
        agent = SimpleNamespace(name="SharedAgent")
        ToolAwareAgent(agent, user_id=1, session_id="known", redis_client=redis)._on_tool_call("search", {}, {}, 1)

        app = FastAPI()

        async def user(x_test_user: int = Header()):
            return SimpleNamespace(id=x_test_user)

        app.dependency_overrides[get_current_user] = user
        app.include_router(chat.router)
        with patch.object(chat, "_get_redis", return_value=redis), TestClient(app) as client:
            owner = client.get("/chat/trace/known", headers={"x-test-user": "1"}).json()
            other = client.get("/chat/trace/known", headers={"x-test-user": "2"}).json()
        self.assertEqual(owner["count"], 1)
        self.assertEqual(other["count"], 0)


if __name__ == "__main__":
    unittest.main()
