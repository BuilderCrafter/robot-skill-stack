"""Integer pixel sets for camera-boundary evidence."""
import unittest

import numpy as np

from robot_skill_stack.world.common.depth_edges import unique_pixels


class FinalRefinementTests(unittest.TestCase):
    def test_scalar_pixel_codes_preserve_negative_and_out_of_image_pairs(self):
        raw = np.array([[-1, 1], [9, 0], [0, -1], [0, -1], [19, -1], [-11, 2]])
        actual = unique_pixels(raw)
        expected = np.unique(raw, axis=0)
        expected = expected[np.lexsort((expected[:, 0], expected[:, 1]))]
        np.testing.assert_array_equal(actual, expected)

if __name__ == '__main__':
    unittest.main()
