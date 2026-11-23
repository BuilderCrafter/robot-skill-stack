"""Integer pixel sets for camera-boundary evidence."""
import unittest

import numpy as np

from robot_skill_stack.world.perception_multiview.observability import _unique_pixels


class FinalRefinementTests(unittest.TestCase):
    def test_scalar_pixel_codes_preserve_negative_and_out_of_image_pairs(self):
        raw = np.array([[-1, 1], [9, 0], [0, -1], [0, -1], [19, -1], [-11, 2]])
        actual = _unique_pixels(raw)
        actual = actual[np.lexsort((actual[:, 1], actual[:, 0]))]
        np.testing.assert_array_equal(actual, np.unique(raw, axis=0))

    def test_pixel_extent_overflow_uses_exact_pair_fallback(self):
        bound = 2**61
        raw = np.array([[-bound, bound], [bound, -bound], [-bound, bound]], dtype=np.int64)
        np.testing.assert_array_equal(_unique_pixels(raw), np.unique(raw, axis=0))

if __name__ == '__main__':
    unittest.main()
