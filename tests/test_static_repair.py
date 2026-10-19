"""The shared static-check-and-revise-once step of the paper graph."""
import unittest
from types import SimpleNamespace

from src.agents.coding_practice import PracticeCancelled, PracticeError, _static_repair

GOOD = "print('ok')\n"
BAD = "import torch\nprint(torch.zeros(1))\n"


def services(revise):
    return SimpleNamespace(revise_paper_draft=revise)


class StaticRepairTests(unittest.TestCase):
    def test_passing_draft_is_not_revised(self):
        def revise(*_):
            raise AssertionError("should not be called")
        self.assertEqual(_static_repair(services(revise), {}, {"code": GOOD}), ({"code": GOOD}, ""))

    def test_failing_draft_is_replaced_by_a_passing_revision(self):
        draft, issue = _static_repair(services(lambda *_: {"code": GOOD}), {}, {"code": BAD})
        self.assertEqual((draft["code"], issue), (GOOD, ""))

    def test_failed_revision_keeps_the_draft_and_its_issue(self):
        def revise(*_):
            raise RuntimeError("model timeout")
        draft, issue = _static_repair(services(revise), {}, {"code": BAD})
        self.assertEqual(draft["code"], BAD)
        self.assertIn("torch", issue)

    def test_oversized_revision_is_rejected(self):
        draft, issue = _static_repair(services(lambda *_: {"code": "x = 1\n" * 5000}), {}, {"code": BAD})
        self.assertEqual(draft["code"], BAD)
        self.assertIn("torch", issue)

    def test_cancellation_propagates(self):
        def revise(*_):
            raise PracticeCancelled("stop")
        with self.assertRaises(PracticeCancelled):
            _static_repair(services(revise), {}, {"code": BAD})

    def test_oversized_draft_fails_the_step(self):
        with self.assertRaises(PracticeError):
            _static_repair(services(lambda *_: {}), {}, {"code": "x = 1\n" * 5000})


if __name__ == "__main__":
    unittest.main()
