import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation import draft_generation_answers, review_golden


ROW = {
    "id": "gq000", "question": "What improves recall?",
    "sources": [{"id": "paper:1", "text": "Hybrid retrieval combines sparse and dense search."}],
    "reference_answer": "", "review_status": "pending",
    "expected_tools": ["search_knowledge_base"], "forbidden_tools": [], "before": [],
}


class GenerationDraftReviewTests(unittest.TestCase):
    def test_draft_remains_unverified(self):
        answer = json.dumps({"answerable": True,
                             "reference_answer": "Hybrid retrieval combines sparse and dense search.",
                             "limitation": "The excerpt does not quantify recall."})
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_: SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=answer))]))))
        draft = draft_generation_answers.draft_one(client, ROW, "gpt-4o-mini")
        self.assertEqual(draft["review_status"], "ai_draft")
        self.assertEqual(draft["source_digest"], review_golden.source_digest(ROW))
        self.assertNotIn("reviewer", draft)

    def test_accepting_draft_requires_answer_and_tool_review(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            candidates, drafts, golden = (path / name for name in
                                          ("candidates.jsonl", "drafts.jsonl", "golden.jsonl"))
            review_golden.write_rows(candidates, [ROW])
            review_golden.write_rows(drafts, [{
                "id": ROW["id"], "source_digest": review_golden.source_digest(ROW),
                "answerable": True, "draft_answer": "Hybrid retrieval combines sparse and dense search.",
                "review_status": "ai_draft",
            }])
            with patch.object(review_golden, "CANDIDATES", candidates), \
                 patch.object(review_golden, "DRAFTS", drafts), \
                 patch.object(review_golden, "GOLDEN", golden):
                answers = iter(["", "State the finding only", "Checked excerpt", "no"])
                review_golden.review("reviewer-1", lambda _: next(answers))
                self.assertFalse(golden.exists())

                answers = iter(["", "State the finding only", "Checked excerpt", "yes", "no"])
                review_golden.review("reviewer-1", lambda _: next(answers))
                rows = review_golden.read_rows(golden)
                self.assertEqual(rows[0]["review_status"], "human_verified")
                self.assertNotIn("expected_tools", rows[0])
                self.assertEqual(rows[0]["reviewer"], "reviewer-1")


if __name__ == "__main__":
    unittest.main()
