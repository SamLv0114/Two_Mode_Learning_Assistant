import asyncio
import unittest
from types import SimpleNamespace

from mcp import Client

from src.agents.mcp_server import authenticated_context, create_mcp_server
from src.api.security import create_refresh_token


class McpServerTests(unittest.TestCase):
    def test_requires_identity_and_interoperates_with_sdk_client(self):
        with self.assertRaises(PermissionError):
            create_mcp_server("")
        refresh, _ = create_refresh_token(7)
        with self.assertRaises(PermissionError):
            authenticated_context(refresh)

        calls = []
        sessions = []

        def execute(name, args, context):
            self.assertIn("db", context)
            calls.append((name, args, context["user"].id))
            return {"results": [{"title": "Paper A", "content": "evidence"}], "count": 1}

        def context_for(token):
            self.assertEqual(token, "access-token")
            session = SimpleNamespace(closed=False)
            session.close = lambda: setattr(session, "closed", True)
            sessions.append(session)
            return {"user": SimpleNamespace(id=7), "db": session}

        server = create_mcp_server(
            "access-token", context_for,
            SimpleNamespace(execute=execute),
        )

        async def exercise():
            async with Client(server) as client:
                tools = await client.list_tools()
                names = {tool.name for tool in tools.tools}
                self.assertEqual(names, {"search_knowledge_base", "fetch_full_paper"})
                result = await client.call_tool("search_knowledge_base", {"query": "RAG", "n_results": 30})
                self.assertFalse(result.is_error)
                self.assertEqual(result.structured_content["result"]["count"], 1)
                result = await client.call_tool("fetch_full_paper", {"arxiv_id": "2401.00001", "query": "method"})
                self.assertFalse(result.is_error)
                rejected = await client.call_tool("fetch_full_paper", {"arxiv_id": "http://localhost/private", "query": "x"})
                self.assertTrue(rejected.is_error)

        asyncio.run(exercise())
        self.assertEqual(calls[0], ("search_knowledge_base", {"query": "RAG", "n_results": 10}, 7))
        self.assertEqual(calls[1][0], "fetch_full_paper")
        self.assertTrue(all(session.closed for session in sessions))


if __name__ == "__main__":
    unittest.main()
