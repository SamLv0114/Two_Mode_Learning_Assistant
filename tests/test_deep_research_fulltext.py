import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.agents.deep_research_agent import TaskSummarizer, TodoItem
from src.agents.tools import _exec_search_knowledge_base


class TaskSummarizerFullTextTests(unittest.TestCase):
    def test_knowledge_base_exposes_paper_id_for_fulltext_lookup(self):
        retriever = SimpleNamespace(retrieve=lambda **_: [{
            "id": "paper_2608.14407", "document": "Abstract", "distance": 0.1,
            "metadata": {"type": "paper", "paper_id": "2608.14407",
                         "title": "AI Scientists", "url": "https://arxiv.org/abs/2608.14407"},
        }])
        with patch("src.agents.tools._expand_query", return_value=["AI Scientists"]):
            result = _exec_search_knowledge_base(
                {"query": "AI Scientists"},
                {"retriever": retriever, "user": SimpleNamespace(id=1)},
            )
        self.assertEqual(result["results"][0]["arxiv_id"], "2608.14407")

    def test_searches_one_matching_papers_full_text_and_passes_evidence_to_model(self):
        calls = []
        messages = []

        def execute(name, args, context):
            calls.append((name, args, context))
            if name == "search_knowledge_base":
                return {"results": [
                    {"type": "paper", "arxiv_id": "2608.14407", "title": "AI Scientists",
                     "url": "https://arxiv.org/abs/2608.14407", "content": "Abstract"},
                    {"type": "paper", "arxiv_id": "2608.14408", "title": "Other paper",
                     "url": "https://arxiv.org/abs/2608.14408", "content": "Other abstract"},
                ]}
            if name == "fetch_full_paper":
                self.assertIsNone(context["db"])
                return {"title": "AI Scientists", "results": [
                    {"content": "The experiment measured a concrete outcome."}
                ]}
            return {"results": [{"title": "Web source", "url": "https://example.org",
                                 "content": "Web evidence"}]}

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: (messages.append(kwargs["messages"]), SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded summary"))]))[1]
        )))
        with patch("src.agents.deep_research_agent.execute_tool", side_effect=execute):
            note = TaskSummarizer(client).summarize(
                TodoItem(1, "Experiments", "experimental results"), {"db": object()}
            )

        self.assertEqual([name for name, _, _ in calls],
                         ["search_knowledge_base", "fetch_full_paper", "search_web"])
        self.assertEqual(calls[1][1]["arxiv_id"], "2608.14407")
        self.assertIn("The experiment measured a concrete outcome.", messages[0][-1]["content"])
        self.assertIn("https://arxiv.org/abs/2608.14407", [c["url"] for c in note.citations])
        self.assertIn("fetch_full_paper", note.tools_called)

    def test_no_arxiv_id_keeps_abstract_and_web_path(self):
        calls = []

        def execute(name, args, context):
            calls.append(name)
            if name == "search_knowledge_base":
                return {"results": [{"type": "paper", "arxiv_id": "untrusted/url",
                                     "title": "Paper", "content": "Abstract"}]}
            return {"results": []}

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_: SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="Abstract summary"))])
        )))
        with patch("src.agents.deep_research_agent.execute_tool", side_effect=execute):
            note = TaskSummarizer(client).summarize(
                TodoItem(1, "Overview", "overview"), {}
            )
        self.assertEqual(calls, ["search_knowledge_base", "search_web"])
        self.assertNotIn("fetch_full_paper", note.tools_called)


if __name__ == "__main__":
    unittest.main()
