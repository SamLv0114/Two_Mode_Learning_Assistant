"""Cost estimates must use measured usage and disclose unknown models."""

import unittest

from evaluation._harness import UsageTracker, estimated_cost_usd, result_header


class CostMetricTests(unittest.TestCase):
    def test_cached_and_uncached_prompt_tokens_use_distinct_rates(self):
        usage = {"by_model": {"gpt-4o-mini": {
            "prompt_tokens": 1_000_000, "cached_tokens": 200_000,
            "completion_tokens": 100_000, "calls": 1,
        }}}
        self.assertAlmostEqual(estimated_cost_usd(usage), 0.8 * 0.15 + 0.2 * 0.075 + 0.1 * 0.60)

    def test_unknown_model_is_not_reported_as_free(self):
        self.assertIsNone(estimated_cost_usd({"by_model": {"unknown-new-model": {
            "prompt_tokens": 10, "cached_tokens": 0, "completion_tokens": 10,
        }}}))

    def test_usage_is_partitioned_by_stage_without_double_counting(self):
        from types import SimpleNamespace
        tracker = UsageTracker()
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=None)
        tracker.record(usage, "gpt-4o-mini", "intent_recognizer._llm_classify")
        tracker.record(usage, "gpt-4o-mini", "base_agent.stream")
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["llm_calls"], 2)
        self.assertEqual(snapshot["by_model"]["gpt-4o-mini"]["prompt_tokens"], 200)
        self.assertEqual(snapshot["by_stage"]["base_agent.stream"]["gpt-4o-mini"]["calls"], 1)

    def test_eval_header_discloses_dirty_tree_and_source_fingerprint(self):
        header = result_header("sample", {"dataset": "fixed"})
        self.assertIn("worktree_dirty", header)
        self.assertIn("source_sha256", header)


if __name__ == "__main__":
    unittest.main()
