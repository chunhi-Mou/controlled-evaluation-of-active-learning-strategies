import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from albench.al.loop import SELECTORS
from albench.al.select_ppal import select as ppal_select
from albench.al.select_rarity import (
    Prediction,
    _integer_quotas,
    bernoulli_entropy,
    composite_scores,
    embedding_diversity,
    image_indicators,
    rarity_weights,
    stratified_query,
    select,
)


def _pred(classes=(), confidence=(), boxes=()):
    return Prediction(
        np.asarray(classes, dtype=int),
        np.asarray(confidence, dtype=float),
        np.asarray(boxes, dtype=float).reshape(-1, 4),
    )


class RarityFormulaTests(unittest.TestCase):
    def test_entropy_and_image_indicators_follow_equations(self):
        self.assertAlmostEqual(float(bernoulli_entropy([0.5])[0]), 1.0)
        prediction = _pred(
            [0, 1], [0.5, 0.5],
            [[0, 0, 2, 2], [3, 4, 5, 6]],
        )
        d, info, cls_div, geometry = image_indicators(
            prediction, np.ones(2), gamma=2.0, nc=2)
        self.assertAlmostEqual(d, 0.5)       # Eq. (4)
        self.assertAlmostEqual(info, 0.25)  # Eq. (5)
        self.assertAlmostEqual(cls_div, 0.5)  # Eqs. (6)-(7)
        self.assertAlmostEqual(geometry, 2.5)  # Eqs. (8)-(9), self-pairs

    def test_rarity_weights_are_median_based_and_clipped(self):
        got = rarity_weights([100.0, 25.0, 0.0], 0.6, 0.8, 2.5)
        expected = np.clip(
            (np.median([100.0, 25.0, 0.0]) /
             (np.array([100.0, 25.0, 0.0]) + 1e-12)) ** 0.6,
            0.8, 2.5)
        np.testing.assert_allclose(got, expected)

    def test_composite_is_equal_weight_sum_and_deterministic(self):
        predictions = [
            _pred([0], [0.5], [[0, 0, 2, 2]]),
            _pred([1], [0.7], [[2, 2, 4, 4]]),
            _pred([], [], []),
        ]
        embeddings = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=float)
        params = dict(beta=0.6, alpha_min=0.8, alpha_max=2.5,
                      eps=1e-12, gamma=2.0, kmeans_iter=5)
        a = composite_scores(predictions, embeddings, 2, 2, params, 13)
        b = composite_scores(predictions, embeddings, 2, 2, params, 13)
        np.testing.assert_allclose(a[0], a[1].sum(axis=1))
        np.testing.assert_allclose(a[0], b[0])

    def test_integer_adjustment_and_no_detection_budget_guard(self):
        quotas = _integer_quotas(np.asarray([0.51, 0.31, 0.18]), 7)
        self.assertEqual(int(quotas.sum()), 7)
        paths = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        predictions = [
            _pred([0], [0.5], [[0, 0, 1, 1]]),
            _pred([1], [0.5], [[0, 0, 1, 1]]),
            _pred([], [], []),
            _pred([], [], []),
        ]
        picks, diag = stratified_query(
            paths, predictions, np.asarray([4, 3, 2, 1], dtype=float),
            np.ones(2), np.asarray([10, 1], dtype=float),
            budget=3, rho=0.5, beta=0.6)
        self.assertEqual(len(picks), 3)
        self.assertEqual(len(set(picks)), 3)
        self.assertEqual(int(diag["k_strat"]), 1)
        self.assertEqual(int(diag["k_free"]), 2)

    def test_ppal_registry_entry_is_unchanged(self):
        self.assertIs(SELECTORS["ppal"], ppal_select)

    def test_duplicate_embeddings_are_finite(self):
        predictions = [_pred([0], [0.5], [[0, 0, 1, 1]]) for _ in range(4)]
        embeddings = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], float)
        params = dict(beta=0.6, alpha_min=0.8, alpha_max=2.5,
                      eps=1e-12, gamma=2.0, kmeans_iter=5)
        scores, components, _, _ = composite_scores(
            predictions, embeddings, 1, 3, params, 13)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(np.isfinite(components)))

    def test_kmeans_preserves_duplicate_embedding_multiplicity(self):
        embeddings = np.asarray(
            [[1, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        captured = {}

        def fake_kmeans(data, k, **kwargs):
            captured["data"] = data.copy()
            return data[[0, 2]], np.zeros(len(data), dtype=int)

        with patch("scipy.cluster.vq.kmeans2", side_effect=fake_kmeans):
            got = embedding_diversity(embeddings, 2, 5, 13)
        self.assertEqual(captured["data"].shape[0], 4)
        np.testing.assert_allclose(captured["data"][0], captured["data"][1])
        self.assertTrue(np.all(np.isfinite(got)))

    def test_candidate_ground_truth_is_never_used(self):
        class DummyModel:
            pass

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            images = root / "images"
            labels = root / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            names = ["labeled", "a", "b", "c", "d"]
            for name in names:
                (images / f"{name}.jpg").write_bytes(b"")
                (labels / f"{name}.txt").write_text(
                    "0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            labeled = [str(images / "labeled.jpg")]
            candidates = [str(images / f"{name}.jpg")
                          for name in ("a", "b", "c", "d")]
            predictions = [
                _pred([0], [0.4], [[0, 0, 1, 1]]),
                _pred([1], [0.5], [[0, 0, 1, 1]]),
                _pred([0, 1], [0.6, 0.3], [[0, 0, 1, 1], [2, 2, 3, 3]]),
                _pred([], [], []),
            ]
            embeddings = np.asarray([[1, 0], [0, 1], [1, 1], [-1, 0]], float)
            cfg = {"classes": {"nc": 2}, "al": {"rarity_aware": {
                "beta": 0.6, "alpha_min": 0.8, "alpha_max": 2.5,
                "eps": 1e-12, "gamma": 2.0, "kmeans_iter": 3,
                "rho": 0.5,
            }}}
            prediction_fn = lambda model, pool, cfg: predictions
            embedding_fn = lambda model, pool, cfg: embeddings
            import random
            with self.assertWarns(RuntimeWarning):
                first = select(DummyModel(), candidates, labeled, 2, cfg,
                               random.Random(13), prediction_fn=prediction_fn,
                               embedding_fn=embedding_fn)
            # Change every candidate GT class. Predictions/embeddings stay fixed.
            for name in ("a", "b", "c", "d"):
                (labels / f"{name}.txt").write_text(
                    "1 0.1 0.1 0.2 0.2\n1 0.8 0.8 0.1 0.1\n",
                    encoding="utf-8")
            with self.assertWarns(RuntimeWarning):
                second = select(DummyModel(), candidates, labeled, 2, cfg,
                                random.Random(13), prediction_fn=prediction_fn,
                                embedding_fn=embedding_fn)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
