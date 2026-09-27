"""Metric and golden-set integrity checks that run without model calls."""

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.runners.run_generation_eval import (
    calibration_stats, context_precision, judge_answer, judge_correctness,
    load_judgment_checkpoint, validate_golden,
)
from evaluation.runners import run_generation_eval
from evaluation.runners.run_trajectory_eval import score_trajectory


class OfflineEvalTests(unittest.TestCase):
    def test_context_precision_uses_relevant_source_positions(self):
        self.assertAlmostEqual(context_precision(["a", "b", "c"], ["b", "c"]), (1 / 2 + 2 / 3) / 2)
        self.assertEqual(context_precision(["a"], ["b"]), 0.0)
        self.assertIsNone(context_precision(["a"], None))

    def test_trajectory_accepts_extra_tools_but_rejects_missing_and_forbidden(self):
        good = score_trajectory(["search_knowledge_base", "search_web", "fetch_full_paper"],
                                ["search_knowledge_base"], ["delete_document"],
                                [["search_knowledge_base", "fetch_full_paper"]])
        self.assertTrue(good["passed"])
        bad = score_trajectory(["fetch_full_paper", "delete_document"],
                               ["search_knowledge_base"], ["delete_document"],
                               [["fetch_full_paper", "delete_document"]])
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["missing"], ["search_knowledge_base"])
        self.assertEqual(bad["forbidden_used"], ["delete_document"])

    def test_pending_cases_cannot_be_published_as_human_verified(self):
        candidate = {
            "id": "gq000", "question": "q", "reference_answer": "a",
            "sources": [{"id": "p1", "text": "evidence"}],
            "review_status": "pending", "reviewer": "",
        }
        with self.assertRaisesRegex(ValueError, "50–100"):
            validate_golden([candidate])
        with self.assertRaisesRegex(ValueError, "human review"):
            validate_golden([{**candidate, "id": f"gq{i:03d}"} for i in range(50)])

    def test_judge_calibration_reports_error_and_pass_agreement(self):
        scored = [{"id": "a", "judgment": {"faithfulness": 0.8, "answer_relevancy": 0.5}}]
        labels = [{"id": "a", "human_faithfulness": 0.9, "human_answer_relevancy": 0.8}]
        stats = calibration_stats(scored, labels)
        self.assertAlmostEqual(stats["faithfulness"]["mae"], 0.1)
        self.assertEqual(stats["faithfulness"]["pass_agreement"], 1)
        self.assertEqual(stats["answer_relevancy"]["pass_agreement"], 0)

    def test_correctness_receives_gold_but_faithfulness_only_receives_retrieved_evidence(self):
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            if len(requests) == 1:
                result = {"faithfulness": 1, "answer_relevancy": 0.5,
                          "unsupported_claims": [], "reason": "supported"}
            else:
                result = {"answer_correctness": 0.25,
                          "missing_key_points": ["70% fewer pilots"],
                          "contradictions": [], "reason": "incomplete"}
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(result)))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        row = {"question": "What changed?", "reference_answer": "70% fewer pilots",
               "sources": [{"id": "p1", "text": "The method reduced pilots by 70%."}]}
        faithfulness = judge_answer(client, row["question"], "It uses a GAN.",
                                    [{"url": "p2", "evidence": "It uses a GAN."}], "faith rubric")
        correctness = judge_correctness(client, row, "It uses a GAN.", "correctness rubric")
        self.assertEqual(faithfulness["faithfulness"], 1)
        self.assertEqual(correctness["answer_correctness"], 0.25)
        self.assertNotIn("reference_answer", requests[0]["messages"][1]["content"])
        self.assertIn("70% fewer pilots", requests[1]["messages"][1]["content"])

    def test_judgment_checkpoint_resumes_only_matching_input_signature(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "partial.jsonl"
            meta = Path(folder) / "meta.json"
            checkpoint.write_text(json.dumps({"id": "gq000", "judgment": {}}) + "\n")
            meta.write_text(json.dumps({"signature": "correct"}))
            rows = [{"id": "gq000"}, {"id": "gq001"}]
            with patch.object(run_generation_eval, "JUDGMENTS_PATH", checkpoint), \
                 patch.object(run_generation_eval, "JUDGMENTS_META_PATH", meta):
                self.assertEqual(len(load_judgment_checkpoint("correct", rows)), 1)
                self.assertEqual(load_judgment_checkpoint("changed", rows), [])


if __name__ == "__main__":
    unittest.main()
