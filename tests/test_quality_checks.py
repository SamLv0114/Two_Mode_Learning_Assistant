"""Deterministic code checks: always-true self-checks and random 'gradients'."""
import unittest

from src.agents.coding_practice import quality_issues


class QualityCheckTests(unittest.TestCase):
    def test_self_comparisons_are_flagged(self):
        issues = quality_issues("import numpy as np\nW0 = np.zeros(2)\nassert np.allclose(W0, W0)\nok = (W0 == W0)\n")
        self.assertEqual(len(issues), 2)
        self.assertIn("compares W0 with itself", issues[0])

    def test_random_gradients_are_flagged(self):
        code = "import numpy as np\nrng = np.random.default_rng(0)\ngrad_A = np.random.randn(2, 2)\nself_grad = rng.normal(size=3)\n"
        issues = quality_issues(code)
        self.assertEqual(len(issues), 2)
        self.assertIn("grad_A is computed from fresh random numbers", issues[0])

    def test_random_numbers_hidden_inside_a_loss_or_gradient_formula_are_flagged(self):
        # What a model wrote when asked to remove grad = np.random.randn(...)
        code = ("import numpy as np\nout = np.ones((3, 5))\nA = np.ones((1, 4))\n"
                "loss = np.sum((out - np.random.randn(*out.shape)) ** 2)\n"
                "grad_B = 2 * (out - np.random.randn(*out.shape)) @ A.T\n")
        self.assertEqual(len(quality_issues(code)), 2)

    def test_real_checks_and_legitimate_noise_pass(self):
        code = ("import numpy as np\nW0 = np.ones(2)\nW0_before = W0.copy()\nx = np.random.randn(2)\n"
                "loss = float((W0 @ x) ** 2)\ngrad = 2 * (W0 @ x) * x\n"
                "grad += np.random.normal(scale=0.1, size=2)  # noisy SGD\nassert np.allclose(W0, W0_before)\n")
        self.assertEqual(quality_issues(code), [])

    def test_unparseable_code_is_left_to_the_syntax_check(self):
        self.assertEqual(quality_issues("def broken(:\n"), [])


if __name__ == "__main__":
    unittest.main()
