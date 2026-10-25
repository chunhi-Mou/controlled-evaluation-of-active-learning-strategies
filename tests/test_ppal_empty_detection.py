import random
import unittest

import numpy as np

from albench.al.ccms import kmedoids
from albench.al.select_ppal import _stage2_with_empty_fallback
from albench.al.ppal_stage2 import acquisition_diag


class PPALEmptyDetectionTests(unittest.TestCase):
    def test_zero_distance_ties_do_not_empty_kmedoids_clusters(self):
        picks = kmedoids(np.zeros((6, 6)), 3, 5, random.Random(13))

        self.assertEqual(len(picks), 3)
        self.assertEqual(len(set(picks)), 3)

    def test_ccms_excludes_empty_detections_and_seeded_fills_budget(self):
        candidates = ["a", "b", "c", "d"]
        feats = [np.ones((1, 2)), np.zeros((0, 2)),
                 np.array([[0.0, 1.0]]), np.zeros((0, 2))]
        labels = [np.array([0]), np.array([], int),
                  np.array([1]), np.array([], int)]
        scores = [np.array([0.9]), np.array([]),
                  np.array([0.8]), np.array([])]

        selected, D, picks, stats = _stage2_with_empty_fallback(
            candidates, feats, labels, scores, 3, 0.05, 5,
            random.Random(42))

        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)
        self.assertTrue({"a", "c"}.issubset(selected))
        self.assertEqual(D.shape, (2, 2))
        self.assertEqual(len(picks), 2)
        self.assertEqual(stats["n_no_detection"], 2)
        self.assertEqual(stats["n_random_fill"], 1)
        # Golden output: adding other strategy registries must not alter the
        # numerical PPAL path for the same inputs and seed.
        self.assertEqual(selected, ["a", "b", "c"])

    def test_all_empty_detections_fall_back_without_ccms(self):
        candidates = ["a", "b", "c"]
        empty_feat = [np.zeros((0, 2)) for _ in candidates]
        empty_label = [np.array([], int) for _ in candidates]
        empty_score = [np.array([]) for _ in candidates]

        selected, D, picks, stats = _stage2_with_empty_fallback(
            candidates, empty_feat, empty_label, empty_score, 2, 0.05, 5,
            random.Random(13))

        self.assertEqual(len(selected), 2)
        self.assertEqual(D.shape, (0, 0))
        self.assertEqual(picks, [])
        self.assertEqual(stats["n_ccms_candidates"], 0)
        self.assertEqual(stats["n_random_fill"], 2)

        diag = acquisition_diag(D, picks, np.zeros(3), pool_n=3)
        self.assertTrue(np.isnan(diag["coverage_picked"]))
        self.assertEqual(int(diag["n_cand"]), 0)


if __name__ == "__main__":
    unittest.main()
