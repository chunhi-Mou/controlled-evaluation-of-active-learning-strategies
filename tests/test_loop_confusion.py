import unittest

import numpy as np

from albench.al.loop import _is_final_round, _validated_confusion


class TestLoopConfusion(unittest.TestCase):
    def setUp(self):
        self.al = {"schedule": {
            "init_frac": 0.05,
            "step_frac": 0.025,
            "cap_frac": 0.20,
        }}

    def test_only_final_round_collects_confusion(self):
        self.assertEqual(
            [_is_final_round(r, self.al) for r in range(7)],
            [False, False, False, False, False, False, True],
        )

    def test_valid_confusion_is_returned(self):
        matrix = np.ones((9, 9), dtype=np.float64)
        checked = _validated_confusion(matrix, 8, "random", 13)
        np.testing.assert_array_equal(checked, matrix)

    def test_empty_confusion_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "confusion is empty"):
            _validated_confusion(np.zeros((9, 9)), 8, "random", 13)

    def test_wrong_shape_and_nonfinite_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "confusion shape"):
            _validated_confusion(None, 8, "random", 13)
        with self.assertRaisesRegex(RuntimeError, "confusion shape"):
            _validated_confusion(np.ones((8, 8)), 8, "random", 13)
        matrix = np.ones((9, 9))
        matrix[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "NaN/Inf"):
            _validated_confusion(matrix, 8, "random", 13)


if __name__ == "__main__":
    unittest.main()
