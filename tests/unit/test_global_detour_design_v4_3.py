"""Independent fixture geometry, interval bounds and predeclared selection."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.audit_rejected_ik_v4_2 import IndependentUrdfFK
from scripts.design_global_detour_v4_3 import (
    ALLOWANCE_M, PROTOCOL, aabb_distance, box_vertices, continuous_aabb_scores,
    densify, displacement_weights, intersection_proof, ordered_candidates, relative_weights,
)


class GlobalDetourDesignTests(unittest.TestCase):
    def test_grid_and_witness_library_are_finite_before_candidate_search(self):
        protocol = json.loads(PROTOCOL.read_text())
        grid = protocol['candidate_grid_integer_steps']
        count = np.prod([grid[name][1]-grid[name][0]+1 for name in ('x_inclusive', 'y_inclusive', 'z_inclusive')])
        self.assertEqual(count*len(protocol['candidate_dimensions_m']), 14858)
        self.assertEqual(len(protocol['witness_priority']), 7)
        self.assertEqual(protocol['candidate_RRT_calls_allowed_during_design'], 0)
        self.assertEqual(protocol['detour_acceptance']['minimum_rrt_route_waypoints'], 3)
        self.assertEqual(protocol['detour_acceptance']['minimum_maximum_L2_deviation_from_direct_joint_segment_rad'], .05)

    def test_densification_preserves_exact_nine_dof_endpoints_and_bounds_angles(self):
        first = np.r_[np.zeros(7), .0021, .0023]
        second = first+np.r_[[.017, -.011, 0., 0., 0., 0., 0.], 0., 0.]
        rows = densify([first, second], .005)
        np.testing.assert_array_equal(rows[0], first)
        np.testing.assert_array_equal(rows[-1], second)
        self.assertLessEqual(np.abs(np.diff(rows[:, :7], axis=0)).max(), .005)
        np.testing.assert_array_equal(rows[:, 7:], np.tile(first[7:], (len(rows), 1)))
        with self.assertRaises(ValueError):
            densify(np.zeros((2, 7)), .005)

    def test_aabb_separation_is_conservative_for_nonoverlap_and_not_overlap_proof(self):
        lower, upper = np.array([-.5]*3), np.array([.5]*3)
        self.assertAlmostEqual(aabb_distance(lower, upper, np.array([2., 0., 0.]), np.array([.5]*3)), 1.)
        self.assertEqual(aabb_distance(lower, upper, np.zeros(3), np.array([.25]*3)), 0.)

    def test_interval_bound_subtracts_whole_body_displacement_not_tcp_distance(self):
        low = np.array([[[0., 0., 0.]], [[.1, 0., 0.]]])
        high = low+.2
        score = continuous_aabb_scores(low, high, np.array([.1]), np.array([[1., 0., 0.]]), np.array([.2]*3))
        self.assertAlmostEqual(score[0], .5-ALLOWANCE_M)
        # A physically clear TCP says nothing about a wider body hull: overlap
        # of the enclosing hull bounds must remain uncertified.
        blocked = continuous_aabb_scores(low, high, np.array([.1]), np.array([[.2, 0., 0.]]), np.array([.2]*3))
        self.assertLess(blocked[0], 0.)

    def test_selection_rank_is_fixed_geometric_objective_and_lexicographic_tie(self):
        centers = np.array([[.5, .1, .3], [.4, .1, .3]])
        sizes = np.array([[.04, .04, .08], [.06, .06, .12]])
        rows = ordered_candidates(centers, sizes, [np.array([.02, .02]), np.array([.03, .005])], .01)
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(rows[0]['score'], .03)
        np.testing.assert_array_equal(rows[1]['center'], [.4, .1, .3])

    def test_positive_volume_witness_distinguishes_overlap_from_tangency(self):
        part = dict(body='/Franka/panda_rightfinger')
        solid = box_vertices([0., 0., 0.], [1., 1., 1.])
        overlap = intersection_proof([[solid]], [part], np.array([.8, 0., 0.]), np.ones(3), .01, times=[.2])
        self.assertIsNotNone(overlap)
        self.assertGreater(overlap['positive_ball_radius_lower_bound_m'], .09)
        self.assertEqual(overlap['robot_body'], '/Franka/panda_rightfinger')
        self.assertEqual(overlap['planned_time'], .2)
        tangent = intersection_proof([[solid]], [part], np.array([1., 0., 0.]), np.ones(3), .001)
        self.assertIsNone(tangent)

    def test_displacement_envelope_and_common_ancestor_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'robot.urdf'
            path.write_text('<robot><link name="root"/><link name="arm"/><link name="tip"/>'
                '<joint name="q1" type="revolute"><parent link="root"/><child link="arm"/>'
                '<origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>'
                '<joint name="q2" type="fixed"><parent link="arm"/><child link="tip"/>'
                '<origin xyz="1 0 0"/></joint></robot>')
            fk = IndependentUrdfFK(path)
            weights = displacement_weights(fk, ['arm', 'tip'], [.2, .1], ('q1',), .005)
            self.assertAlmostEqual(weights[0, 0], .2)
            self.assertAlmostEqual(weights[1, 0], 1.1)
            # Both shapes rotate by the same root joint. Their relative solid
            # distance remains invariant even though the individual radius
            # envelopes differ; the common ancestor contribution must cancel.
            np.testing.assert_array_equal(relative_weights(weights[0], weights[1]), [0.])
            for angle in (.01, .3, 1.):
                before = fk.poses(('q1',), [0.])['tip'][:3, 3]
                after = fk.poses(('q1',), [angle])['tip'][:3, 3]
                self.assertLessEqual(np.linalg.norm(after-before), weights[1, 0]*abs(angle))


if __name__ == '__main__':
    unittest.main()
