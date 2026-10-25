import unittest

import numpy as np

from albench.al.charts import _mean_std
from albench.al.metrics import aubc


class AubcTests(unittest.TestCase):
    def test_uses_realized_labeled_image_counts_when_available(self):
        rounds = [
            {"frac": 0.05, "n_labeled": 185, "score": 0.2},
            {"frac": 0.075, "n_labeled": 278, "score": 0.4},
            {"frac": 0.10, "n_labeled": 370, "score": 0.8},
        ]

        expected = (0.3 * 93 + 0.6 * 92) / 185
        self.assertAlmostEqual(aubc(rounds, "score"), expected)

    def test_uses_fractions_when_counts_are_absent(self):
        rounds = [
            {"frac": 0.05, "score": 0.2},
            {"frac": 0.075, "score": 0.4},
            {"frac": 0.10, "score": 0.8},
        ]

        self.assertAlmostEqual(aubc(rounds, "score"), 0.45)

    def test_rejects_mixed_realized_budget_schema(self):
        rounds = [
            {"frac": 0.05, "n_labeled": 185, "score": 0.2},
            {"frac": 0.075, "score": 0.4},
        ]

        with self.assertRaisesRegex(ValueError, "mixed budget schema"):
            aubc(rounds, "score")

    def test_rejects_duplicate_realized_budgets(self):
        rounds = [
            {"round": 0, "n_labeled": 185, "score": 0.2},
            {"round": 1, "n_labeled": 185, "score": 0.4},
        ]

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            aubc(rounds, "score")

    def test_rejects_decreasing_realized_budgets_by_round(self):
        rounds = [
            {"round": 0, "n_labeled": 185, "score": 0.2},
            {"round": 1, "n_labeled": 170, "score": 0.4},
        ]

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            aubc(rounds, "score")

    def test_rejects_non_finite_values(self):
        rounds = [
            {"n_labeled": 185, "score": 0.2},
            {"n_labeled": 278, "score": float("nan")},
        ]

        with self.assertRaisesRegex(ValueError, "finite"):
            aubc(rounds, "score")


class ReportingDispersionTests(unittest.TestCase):
    def test_multiple_seed_curves_use_sample_standard_deviation(self):
        mean, std = _mean_std([[1.0, 2.0], [3.0, 4.0]])

        np.testing.assert_allclose(mean, [2.0, 3.0])
        np.testing.assert_allclose(std, [np.sqrt(2.0), np.sqrt(2.0)])

    def test_one_seed_curve_has_undefined_standard_deviation(self):
        _, std = _mean_std([[1.0, 2.0]])

        self.assertTrue(np.isnan(std).all())


if __name__ == "__main__":
    unittest.main()
