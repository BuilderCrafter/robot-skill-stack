"""Current measured side intervals separate face-grid yaw and side identity."""
from copy import deepcopy
from itertools import product
import unittest

import numpy as np

from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from robot_skill_stack.world.perception_multiview.fitting import PrimitiveFit
from scripts.probe_multiview_final_geometry import orientation_proposal


def evidence_fit(measured, published, *, yaw=.27, observable=True, padding=.0001):
    geometry = PrimitiveGeometry('cube', .99, box_size=np.asarray(published), yaw=yaw,
        metadata=dict(parameter_observable=observable, geometric_observability=dict(observable=observable),
                      parameter_uncertainty=dict(surface_padding_m=padding)))
    center = np.array([.45, .015, published[2]/2])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    corners = np.array(list(product((-1., 1.), repeat=3)))*np.asarray(measured)/2
    points = np.tile(corners, (5, 1)) @ rotation.T+center
    return PrimitiveFit(geometry, center, abs(rotation) @ published), points


class OrientationEvidenceProposalTests(unittest.TestCase):
    def test_asymmetric_outer_bounds_do_not_invent_square_major_axis(self):
        fit, points = evidence_fit((.05, .05, .05), (.0502, .0508, .05))
        result = orientation_proposal(fit, points)
        self.assertTrue(result['face_grid_observable'])
        self.assertEqual(result['face_grid_period_degrees'], 90)
        self.assertFalse(result['major_axis_observable'])
        self.assertIsNone(result['major_axis_yaw_rad'])
        self.assertEqual(result['reason'], 'overlapping_current_side_intervals')

    def test_distinct_rectangle_major_axis_uses_180_degree_period(self):
        fit, points = evidence_fit((.075, .04, .05), (.076, .042, .05))
        result = orientation_proposal(fit, points)
        self.assertTrue(result['major_axis_observable'])
        self.assertEqual(result['major_axis_period_degrees'], 180)
        self.assertAlmostEqual(result['major_axis_yaw_rad'], .27)

    def test_equivalent_swapped_local_axes_map_to_physical_major_axis(self):
        fit, points = evidence_fit((.04, .075, .05), (.042, .076, .05), yaw=.27-np.pi/2)
        result = orientation_proposal(fit, points)
        self.assertTrue(result['major_axis_observable'])
        self.assertAlmostEqual(result['major_axis_yaw_rad'], .27)

    def test_small_side_difference_can_be_resolved_with_tight_current_evidence(self):
        fit, points = evidence_fit((.0506, .05, .05), (.0507, .0501, .05), padding=.00001)
        result = orientation_proposal(fit, points)
        self.assertTrue(result['major_axis_observable'])
        self.assertEqual(result['reason'], 'distinct_current_side_intervals')

    def test_incomplete_footprint_has_only_a_numerical_hypothesis(self):
        fit, points = evidence_fit((.09, .0434, .05), (.0901, .0435, .05), observable=False)
        old = deepcopy(fit.geometry.metadata)
        result = orientation_proposal(fit, points)
        self.assertFalse(result['face_grid_observable'])
        self.assertFalse(result['major_axis_observable'])
        self.assertEqual(result['yaw_hypothesis_rad'], fit.geometry.yaw)
        self.assertEqual(fit.geometry.metadata, old)

    def test_missing_current_points_cannot_reuse_old_evidence(self):
        fit, _ = evidence_fit((.075, .04, .05), (.076, .042, .05))
        result = orientation_proposal(fit, [])
        self.assertFalse(result['face_grid_observable'])
        self.assertFalse(result['major_axis_observable'])
        self.assertEqual(result['reason'], 'insufficient_current_points')

    def test_featureless_sphere_has_no_geometric_heading(self):
        fit = PrimitiveFit(PrimitiveGeometry('sphere', .99, radius=.025),
                           np.array([.45, 0., .025]), np.full(3, .05))
        result = orientation_proposal(fit, [])
        self.assertFalse(result['face_grid_observable'])
        self.assertFalse(result['major_axis_observable'])
        self.assertIsNone(result['yaw_hypothesis_rad'])


if __name__ == '__main__':
    unittest.main()
