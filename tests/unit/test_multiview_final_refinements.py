"""Integer pixel sets and geometric axis metadata."""
import unittest

import numpy as np

from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from robot_skill_stack.world.perception_multiview.fitting import (
    PrimitiveFit, orientation_metadata,
)
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

    def test_cylinder_axis_metadata_preserves_existing_evidence_and_geometry(self):
        geometry = PrimitiveGeometry('cylinder', .99, radius=.025, length=.08,
            axis=np.array([1., 0., 0.]), metadata=dict(parameter_observable=True))
        fit = PrimitiveFit(geometry, np.array([.45, 0., .025]), np.array([.08, .05, .05]))
        before = geometry.axis.copy()
        metadata = orientation_metadata(fit, [])
        self.assertTrue(metadata['axis_observable'])
        self.assertFalse(metadata['rotation_about_axis_observable'])
        self.assertIsNone(metadata['yaw_hypothesis_rad'])
        np.testing.assert_array_equal(geometry.axis, before)
        self.assertEqual(geometry.metadata, dict(parameter_observable=True))
        geometry.metadata['parameter_observable'] = False
        self.assertFalse(orientation_metadata(fit, [])['axis_observable'])


if __name__ == '__main__':
    unittest.main()
