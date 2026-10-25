import unittest

import numpy as np

from albench.al.ccms import _directional_kl, image_distance_matrix
from albench.al.ppal_difficulty import (
    QualityAssignerProxy,
    QualityEMA,
    quality_sums_from_assignments,
)


class PPALPaperFidelityTests(unittest.TestCase):
    def test_all_positive_assignments_use_the_assigned_gt_class(self):
        import torch

        # Both positives are assigned class 0 even though class 1 is argmax.
        scores = torch.tensor([[[0.01, 0.99], [0.04, 0.80], [0.9, 0.1]]])
        boxes = torch.tensor([[[0., 0., 2., 2.], [0., 0., 1., 1.],
                               [5., 5., 6., 6.]]])
        labels = torch.tensor([[0, 0, 1]])
        targets = torch.tensor([[[0., 0., 2., 2.], [0., 0., 2., 2.],
                                 [0., 0., 0., 0.]]])
        foreground = torch.tensor([[True, True, False]])

        sums, counts = quality_sums_from_assignments(
            scores, boxes, labels, targets, foreground, nc=2, xi=0.6)
        expected = 0.01 ** 0.6 + 0.04 ** 0.6 * 0.25 ** 0.4
        self.assertAlmostEqual(float(sums[0]), expected, places=6)
        self.assertEqual(float(counts[0]), 2.0)
        self.assertEqual(float(counts[1]), 0.0)

    def test_assigner_proxy_returns_the_original_tuple_unchanged(self):
        import torch

        output = (
            torch.tensor([[0]]),
            torch.tensor([[[0., 0., 1., 1.]]]),
            torch.ones((1, 1, 2)),
            torch.tensor([[True]]),
            torch.tensor([[0]]),
        )

        class FakeAssigner:
            def __call__(self, *args):
                return output

        state = QualityEMA(2, base_momentum=0.0)
        proxy = QualityAssignerProxy(FakeAssigner(), state, xi=0.6)
        got = proxy(
            torch.tensor([[[0.02, 0.98]]]),
            torch.tensor([[[0., 0., 1., 1.]]]),
            torch.zeros((1, 2)), torch.zeros((1, 1, 1)),
            torch.tensor([[[0., 0., 1., 1.]]]), torch.ones((1, 1, 1)))
        self.assertIs(got, output)
        self.assertAlmostEqual(
            state.value[0], 0.02 ** 0.6 / (1.0 + 1e-5), places=7)

    def test_kl_direction_nearest_weight_and_symmetry(self):
        source = np.asarray([[0.8, 0.2], [0.6, 0.4]])
        target = np.asarray([[0.3, 0.7]])
        source_scores = np.asarray([0.9, 0.1])
        target_scores = np.asarray([0.8])
        got = _directional_kl(
            source, target, source_scores, target_scores, 0.05)
        q = target[0]
        kls = np.sum(q * (np.log(q) - np.log(source)), axis=1)
        expected = (0.9 * kls[0] + 0.1 * kls[1]) / (1.0 + 1e-5)
        self.assertAlmostEqual(got, expected, places=10)

        feats = [source, target]
        labels = [np.asarray([0, 1]), np.asarray([1])]
        scores = [source_scores, target_scores]
        distance = image_distance_matrix(feats, labels, scores, 0.05)
        np.testing.assert_allclose(distance, distance.T)
        np.testing.assert_allclose(np.diag(distance), 0.0)

    def test_finite_kl_above_two_is_not_zeroed(self):
        feats = [np.asarray([[0.999, 0.001]]),
                 np.asarray([[0.001, 0.999]])]
        scores = [np.asarray([0.9]), np.asarray([0.9])]
        labels = [np.asarray([0]), np.asarray([1])]
        distance = image_distance_matrix(feats, labels, scores, 0.05)
        self.assertTrue(np.isfinite(distance[0, 1]))
        self.assertGreater(distance[0, 1], 2.0)


if __name__ == "__main__":
    unittest.main()
