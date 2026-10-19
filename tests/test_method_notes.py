"""Coding Agent method knowledge base: valid notes, deterministic selection, and prompts that use them."""
import unittest
from unittest.mock import patch

from evaluation.runners import run_method_notes_eval
from src.agents import method_notes
from src.agents.coding_services import CodingServices
from evaluation._harness import load_jsonl


def ids(title, abstract="", method="", objective=""):
    return [note.id for note in method_notes.select(title, abstract, method, objective)]


class NoteFileTests(unittest.TestCase):
    def test_every_note_parses_with_all_sections(self):
        notes = method_notes.all_notes()
        self.assertGreaterEqual(len(notes), 10)
        self.assertEqual(len({n.id for n in notes}), len(notes))
        for note in notes:
            self.assertTrue(note.triggers, note.id)
            for section in method_notes.SECTIONS:
                self.assertTrue(note.sections[section].startswith("- "), (note.id, section))

    def test_render_includes_only_the_requested_sections_and_the_precedence_rule(self):
        block = method_notes.render(method_notes.by_ids(["lora"]), ["Common mistakes"])
        self.assertIn("follow the excerpt", block)
        self.assertIn("B to zero", block)
        self.assertNotIn("Self-checks", block)
        self.assertEqual(method_notes.render([], ["Common mistakes"]), "")


class SelectionTests(unittest.TestCase):
    def test_canonical_papers(self):
        self.assertEqual(ids("LoRA: Low-Rank Adaptation of Large Language Models"), ["lora"])
        self.assertEqual(ids("Denoising Diffusion Probabilistic Models"), ["diffusion"])
        self.assertEqual(ids("Direct Preference Optimization: Your Language Model is Secretly a Reward Model"),
                         ["preference_optimization"])
        self.assertEqual(ids("Attention Is All You Need", "a new network architecture, the Transformer, "
                                                           "based solely on attention mechanisms"), ["attention"])
        self.assertEqual(ids("Proximal Policy Optimization Algorithms"), ["policy_gradient"])

    def test_lookalikes_are_not_matched(self):
        self.assertEqual(ids("LoRaWAN networks for low-power IoT sensing"), [])
        self.assertNotIn("quantization", ids("Residual Quantized VQ-VAE tokens for motion generation"))
        self.assertEqual(ids("Sparse Bayesian Message Passing under Structural Uncertainty"), [])

    def test_one_weak_mention_in_the_abstract_is_not_enough(self):
        self.assertEqual(ids("A benchmark for tabular reasoning", "unlike knowledge distillation, we ..."), [])

    def test_the_method_name_counts_like_the_title(self):
        self.assertEqual(ids("Making agents faster", method="group relative policy optimization with a "
                                                            "length penalty"), ["policy_gradient"])

    def test_a_specific_family_ranks_before_the_generic_one(self):
        self.assertEqual(ids("LoRA for Vision Transformers"), ["lora", "attention"])


class SelectionQualityGate(unittest.TestCase):
    """Editing triggers must not quietly make selection worse on the labelled papers."""

    def test_precision_and_recall_stay_above_their_floors(self):
        result = run_method_notes_eval.evaluate(load_jsonl(run_method_notes_eval.DATASET))
        self.assertGreaterEqual(result["precision"], 0.90)
        self.assertGreaterEqual(result["recall"], 0.80)


class PromptTests(unittest.TestCase):
    def services(self):
        services = CodingServices.__new__(CodingServices)
        services.calls = []
        services._json = lambda system, user, **kwargs: services.calls.append((system, user)) or {}
        services._profile, services.personalized, services.user_id = "", False, None
        return services

    STATE = {"title": "LoRA: Low-Rank Adaptation", "evidence": [{"id": "E1", "text": "W0 is frozen."}],
             "manifest": {"method": "LoRA", "known_details": []}, "code": "x = 1\n"}

    def test_generation_gets_mistakes_and_self_checks(self):
        services = self.services()
        services.generate_paper({**self.STATE, "method_notes": ["lora"]})
        system, user = services.calls[0]
        self.assertIn("Common mistakes", user)
        self.assertIn("Self-checks", user)
        self.assertIn("follow the excerpt", system)

    def test_review_gets_mistakes_but_must_still_cite_an_excerpt(self):
        services = self.services()
        services.review_draft({**self.STATE, "method_notes": ["lora"]})
        system, user = services.calls[0]
        self.assertIn("Common mistakes", user)
        self.assertNotIn("Self-checks", user)
        self.assertIn("never a requirement", system)

    def test_method_check_gets_the_choices_this_family_leaves_open(self):
        services = self.services()
        with patch.object(CodingServices, "_code_note", return_value=""):
            services._assess({"title": "LoRA: Low-Rank Adaptation", "abstract": "", "evidence": self.STATE["evidence"]},
                             objective="")
        self.assertIn("Often left open", services.calls[0][1])

    def test_no_notes_no_block(self):
        services = self.services()
        services.generate_paper(self.STATE)
        self.assertNotIn("General implementation notes for this kind", services.calls[0][1])


if __name__ == "__main__":
    unittest.main()
