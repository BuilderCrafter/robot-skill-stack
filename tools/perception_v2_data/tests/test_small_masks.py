"""Regression tests for the converter's pinned-loader fidelity check."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_dataset import mask_polygon, resample


class SmallMaskTests(unittest.TestCase):
    def test_resample_keeps_original_vertices(self):
        poly = np.array([[1., 3.], [2., 9.], [8., 10.], [12., 4.], [7., 1.]], np.float32)
        dense = resample(poly)
        self.assertEqual(dense.shape, (1000, 2))
        for vertex in poly:
            self.assertTrue(np.any(np.all(dense == vertex, axis=1)))

    def test_float32_like_pinned_loader(self):
        self.assertEqual(resample(np.array([[0., 0.], [4., 0.], [3., 4.]])).dtype, np.float32)

    def test_already_dense_polygon_unchanged(self):
        poly = np.random.default_rng(6).random((1000, 2)).astype(np.float32)
        np.testing.assert_array_equal(resample(poly), poly)

    def test_small_circles_at_camera_resolution(self):
        for radius in [5, 6, 8, 12]:
            for center in [(300, 275), (98, 177), (540, 352)]:
                with self.subTest(radius=radius, center=center):
                    mask = np.zeros((480, 640), np.uint8)
                    cv2.circle(mask, center, radius, 1, -1)
                    _, score, training_score = mask_polygon(mask)
                    self.assertEqual(score, 1.)
                    self.assertGreaterEqual(training_score, .90)

    def test_small_rotated_box(self):
        mask = np.zeros((480, 640), np.uint8)
        cv2.fillPoly(mask, [np.array([[300, 265], [312, 270], [308, 285], [296, 279]])], 1)
        _, raw, loaded = mask_polygon(mask)
        self.assertEqual(raw, 1.)
        self.assertGreaterEqual(loaded, .9)

    def test_quality_threshold_not_relaxed(self):
        mask = np.zeros((480, 640), np.uint8); mask[10:12, 11:13] = 1
        with self.assertRaises(ValueError):
            mask_polygon(mask)


if __name__ == '__main__':
    unittest.main(verbosity=2)
