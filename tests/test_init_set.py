import random
import unittest

from albench.al.init_set import random_initial


class RandomInitialTests(unittest.TestCase):
    def test_is_reproducible_and_has_exact_size(self):
        items = [(f"image-{i}", {i % 3}) for i in range(20)]

        first = random_initial(items, 0.25, random.Random(42))
        second = random_initial(items, 0.25, random.Random(42))

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

    def test_ignores_ground_truth_classes(self):
        keys = [f"image-{i}" for i in range(20)]
        balanced = [(key, {i % 4}) for i, key in enumerate(keys)]
        imbalanced = [(key, {0}) for key in keys]

        self.assertEqual(
            random_initial(balanced, 0.25, random.Random(13)),
            random_initial(imbalanced, 0.25, random.Random(13)),
        )

    def test_rejects_duplicate_image_keys(self):
        with self.assertRaisesRegex(ValueError, "unique image keys"):
            random_initial([("same", {0}), ("same", {1})], 0.5,
                           random.Random(13))


if __name__ == "__main__":
    unittest.main()
