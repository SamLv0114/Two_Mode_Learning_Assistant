"""The model's per-paper judgements and the fixed rule that turns them into a feed."""
import re
import unittest
from datetime import datetime

from _fakes import TaskClient, user_text
from src.collectors import PaperData
from src.pipelines.feed import judge


def paper(arxiv_id, title="T"):
    return PaperData(arxiv_id=arxiv_id, title=title, authors=[], abstract="abstract", categories=[],
                     published_date=datetime(2026, 1, 1), arxiv_url="", pdf_url="", citation_count=0)


def verdict(index, fit, readiness="right_level", paper_type="method", reason="Serves your goal."):
    return judge.PaperJudgement(index=index, goal_fit=fit, readiness=readiness, paper_type=paper_type, reason=reason)


PROFILE = {"goal": "implement diffusion", "level": "intermediate", "known": [], "interests": ["diffusion"],
           "preferences": [], "avoid_types": ["survey"]}


class CombineTests(unittest.TestCase):
    def test_goal_fit_outweighs_a_small_prerank_lead(self):
        papers = [paper("a"), paper("b")]
        scored = judge.combine(papers, {"a": 0.9, "b": 0.8},
                               {"a": verdict(0, 1), "b": verdict(1, 3)}, PROFILE)
        self.assertEqual([p.arxiv_id for p, _ in scored], ["b", "a"])

    def test_avoided_types_wrong_level_and_off_topic_are_demoted(self):
        papers = [paper("survey"), paper("hard"), paper("off"), paper("ok")]
        judgements = {"survey": verdict(0, 3, paper_type="survey"), "hard": verdict(1, 3, readiness="too_advanced"),
                      "off": verdict(2, 0), "ok": verdict(3, 2)}
        scored = judge.combine(papers, {"survey": 1.0, "hard": 1.0, "off": 1.0, "ok": 0.0}, judgements, PROFILE)
        order = [p.arxiv_id for p, _ in scored]
        self.assertEqual(order[-1], "off")                  # off-topic goes last whatever its pre-rank
        self.assertLess(order.index("hard"), order.index("survey"))
        self.assertEqual(order[0], "hard")

    def test_readiness_is_ignored_when_no_level_is_stated(self):
        papers = [paper("a"), paper("b")]
        judgements = {"a": verdict(0, 2, readiness="too_basic"), "b": verdict(1, 2)}
        scored = dict((p.arxiv_id, s) for p, s in judge.combine(papers, {"a": 1, "b": 1}, judgements,
                                                                {**PROFILE, "level": ""}))
        self.assertAlmostEqual(scored["a"], scored["b"])


class HedgeTests(unittest.TestCase):
    def test_a_reason_that_hedges_caps_the_fit_at_one(self):
        self.assertEqual(verdict(0, 3, reason="It could improve speech synthesis.").goal_fit, 1)
        self.assertEqual(verdict(0, 2, reason="Potentially useful.").goal_fit, 1)
        self.assertEqual(verdict(0, 3, reason="Introduces the vocoder your goal names.").goal_fit, 3)
        self.assertEqual(verdict(0, 0, reason="").goal_fit, 0)


class SelectTests(unittest.TestCase):
    def test_a_good_graph_paper_takes_the_last_slot_from_a_content_only_feed(self):
        papers = [paper(f"c{i}") for i in range(3)] + [paper("g")]
        scored = [(p, 1.0 - i / 10) for i, p in enumerate(papers)]
        found = {"c0": ["content"], "c1": ["content"], "c2": ["content"], "g": ["graph"]}
        chosen = judge.select(scored, found, {"g": verdict(3, 2)}, top_k=2)
        self.assertEqual([p.arxiv_id for p, _ in chosen], ["c0", "g"])

    def test_a_weak_graph_paper_does_not(self):
        papers = [paper("c0"), paper("c1"), paper("g")]
        scored = [(p, 1.0) for p in papers]
        chosen = judge.select(scored, {"c0": ["content"], "c1": ["content"], "g": ["graph"]},
                              {"g": verdict(2, 1)}, top_k=2)
        self.assertEqual([p.arxiv_id for p, _ in chosen], ["c0", "c1"])


class JudgeCallTests(unittest.TestCase):
    def test_batches_are_judged_in_parallel_and_a_failed_batch_is_skipped(self):
        papers = [paper(f"p{i}", title=f"Paper {i}") for i in range(23)]

        def reply(kwargs):
            indexes = [int(i) for i in re.findall(r"^\[(\d+)\]", user_text(kwargs), re.M)]
            if indexes[0] == 10:
                raise RuntimeError("provider error")
            return {"papers": [{"index": i, "goal_fit": 2, "readiness": "right_level", "paper_type": "method",
                                "reason": f"Because {i}."} for i in indexes] + [
                       {"index": 99, "goal_fit": 3, "readiness": "right_level", "paper_type": "method", "reason": ""}]}

        client = TaskClient(feed_judge=reply)
        result = judge.judge(client, PROFILE, papers)
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(sorted(result), sorted(f"p{i}" for i in list(range(10)) + list(range(20, 23))))
        self.assertEqual(result["p21"].reason, "Because 21.")  # indexes map back to the right paper

    def test_reasons_record_channels_evidence_and_judgement(self):
        record = judge.reasons(["graph"], ["Recommended from “DDPM”"], verdict(0, 3, reason=" Fits. "))
        self.assertEqual(record, {"reason": "Fits.", "goal_fit": 3, "readiness": "right_level", "paper_type": "method",
                                  "channels": ["graph"], "evidence": ["Recommended from “DDPM”"]})
        self.assertIsNone(judge.reasons(["content"], [], None)["reason"])


if __name__ == "__main__":
    unittest.main()
