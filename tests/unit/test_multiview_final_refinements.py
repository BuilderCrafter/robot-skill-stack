"""Guarded retry verdicts, integer pixel sets and geometric axis metadata."""
from unittest.mock import patch
import unittest

import numpy as np

from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from robot_skill_stack.world.perception_multiview.fitting import (
    PrimitiveFit, SurfaceFitter, orientation_metadata,
)
from robot_skill_stack.world.perception_multiview.observability import _unique_pixels
from robot_skill_stack.world.perception_multiview.provider import _FitTimer
from robot_skill_stack.world.perception_multiview import segmentation
from tests.support.multiview_research import analytic_frames
from tests.support.primitive_scene import Solid
from tests.unit.test_multiview_segmentation_readiness import box_surface


class FinalRefinementTests(unittest.TestCase):
    def metric_calls(self, points, fitter, **kwargs):
        radii, original = [], segmentation._metric_components
        def recorded(cloud, **options):
            radii.append(options.get('radius', .004))
            return original(cloud, **options)
        with patch.object(segmentation, '_metric_components', recorded):
            result = segmentation.refine_cluster(points, fitter, **kwargs)
        return result, radii

    def test_complete_single_component_has_one_graph(self):
        _, radii = self.metric_calls(box_surface(), _FitTimer(SurfaceFitter()))
        self.assertEqual(radii, [.004])

    def test_explicit_missing_band_ambiguity_is_not_retried(self):
        points = box_surface()
        points = points[abs(points[:, 0]-.45) > .004]
        result, radii = self.metric_calls(points, _FitTimer(SurfaceFitter()))
        self.assertEqual(radii, [.004])
        self.assertEqual(result[0][1].geometry.metadata['reason'], 'ambiguous_single_or_multiple_surfaces')

    def test_occupied_bridge_keeps_supplied_parent_without_retry(self):
        points = box_surface()
        points = points[abs(points[:, 0]-.45) > .004]
        parent = SurfaceFitter().fit(points)
        frames = analytic_frames([Solid('cube', center=(.45, 0., .025), size=(.05, .05, .05))])
        result, radii = self.metric_calls(points, _FitTimer(SurfaceFitter()), parent_fit=parent, frames=frames)
        self.assertEqual(radii, [.004])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0][1], parent)

    def test_less_than_minimum_two_child_budget_does_no_retry_or_fits(self):
        points = box_surface()
        points = points[abs(points[:, 0]-.45) > .004]
        fitter = _FitTimer(SurfaceFitter(), max_calls=4)
        _, radii = self.metric_calls(points, fitter)
        self.assertEqual(radii, [.004])
        self.assertEqual(fitter.calls, 0)

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
