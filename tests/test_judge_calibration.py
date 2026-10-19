"""Judge calibration: cases with known answers, and the agreement metrics computed from them."""
import unittest

from evaluation.runners import run_judge_calibration as calibration

ABSTRACT_A = ("We study low-rank adaptation of large language models. "
              "Our method reduces the number of trainable parameters by 10000 times while matching quality. "
              "It also lowers GPU memory use during training by a large margin. ")
ABSTRACT_B = ("We introduce a retrieval benchmark for scientific question answering. "
              "The benchmark contains questions written by domain experts across many fields. ")
PAPERS = [{"arxiv_id": "a", "title": "LoRA", "abstract": ABSTRACT_A},
          {"arxiv_id": "b", "title": "Bench", "abstract": ABSTRACT_B}]


class CaseConstructionTests(unittest.TestCase):
    def test_perturbations_are_deterministic_and_change_the_claim(self):
        sentence = "Our method reduces the number of trainable parameters by 10000 times while matching quality."
        self.assertEqual(calibration.change_number(sentence),
                         "Our method reduces the number of trainable parameters by 30007 times while matching quality.")
        self.assertEqual(calibration.negate("It is faster than full fine-tuning."), "It is not faster than full fine-tuning.")
        self.assertIsNone(calibration.change_number("No digits here at all in this sentence."))

    def test_supported_sentences_come_only_from_what_the_judge_sees(self):
        long_abstract = ("An introductory sentence that is comfortably long enough to be a candidate. " * 12
                         + "The hidden claim reports an improvement of exactly 42 percent overall. ")
        visible = calibration.visible_sentences(long_abstract)
        self.assertTrue(visible)
        self.assertTrue(all("hidden claim" not in s for s in visible))  # past the 700-character window

    def test_cases_cover_each_construction_with_the_right_gold_label(self):
        cases = calibration.build_cases(PAPERS)
        kinds = {(c["id"].split(":")[0], c["kind"]): c for c in cases}
        lora = kinds[("a", "supported")]
        self.assertIn(lora["answer"], " ".join(ABSTRACT_A.split()))
        self.assertTrue(lora["gold"])
        self.assertIn("10000", lora["answer"])  # a sentence with a number is preferred
        self.assertFalse(kinds[("a", "number")]["gold"])
        self.assertFalse(kinds[("a", "unrelated")]["gold"])
        self.assertNotIn(kinds[("a", "unrelated")]["answer"], ABSTRACT_A)
        self.assertEqual(calibration.build_cases(PAPERS), cases)  # seeded: the same set every run


class AgreementMetricTests(unittest.TestCase):
    def rows(self, scores_by_kind):
        return [{"kind": kind, "gold": kind == "supported", "scores": scores} for kind, scores in scores_by_kind]

    def test_perfect_judge(self):
        summary = calibration.summarise(self.rows([("supported", [0.9, 0.95]), ("number", [0.1, 0.2]),
                                                   ("negation", [0.0, 0.1]), ("unrelated", [0.2, 0.2])]))
        self.assertEqual((summary["accuracy"], summary["kappa"], summary["false_pass"]), (1.0, 1.0, 0.0))
        self.assertEqual(summary["retest_agreement"], 1.0)

    def test_a_judge_that_passes_everything_has_no_agreement_beyond_chance(self):
        summary = calibration.summarise(self.rows([("supported", [0.9]), ("number", [0.9]), ("negation", [0.8])]))
        self.assertEqual(summary["false_pass"], 1.0)
        self.assertEqual(summary["kappa"], 0.0)

    def test_failed_judgements_are_excluded_and_counted_and_instability_is_measured(self):
        summary = calibration.summarise(self.rows([("supported", [0.9, 0.3]), ("number", [None, None]),
                                                   ("unrelated", [0.1, 0.1])]))
        self.assertEqual((summary["scored"], summary["judge_failures"]), (2, 1))
        self.assertEqual(summary["retest_agreement"], 0.5)  # the supported case flipped between runs
        self.assertAlmostEqual(summary["retest_max_drift"], 0.6)

    def test_report_renders_with_a_fake_judge(self):
        cases = calibration.build_cases(PAPERS)
        results = calibration.judge_cases(cases, lambda case: 1.0 if case["gold"] else 0.0, repeats=2)
        text = calibration.report(calibration.summarise(results), {"judge": "fake"}, cost=None)
        self.assertIn("Cohen's kappa **1.000**", text)


if __name__ == "__main__":
    unittest.main()
