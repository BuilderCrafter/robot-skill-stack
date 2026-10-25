"""Independent FK, intersection witnesses and honest rejection attribution."""
import itertools
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.audit_rejected_ik_v4_2 import (
    IndependentUrdfFK, axis_rotation, classify_candidate, classify_relation, intersection_ball, physical_relations,
)


def cube(low, high):
    return np.array(list(itertools.product(*zip(low, high))), float)


class RejectedIkAuditTests(unittest.TestCase):
    def test_independent_fk_composes_origin_before_motion_and_keeps_both_fingers(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'robot.urdf'
            path.write_text('<robot><link name="base_link"/><link name="arm"/>'
                '<link name="left"/><link name="right"/>'
                '<joint name="a" type="revolute"><parent link="base_link"/><child link="arm"/>'
                '<origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/><axis xyz="0 0 1"/></joint>'
                '<joint name="l" type="prismatic"><parent link="arm"/><child link="left"/>'
                '<origin xyz=".2 0 0"/><axis xyz="0 1 0"/></joint>'
                '<joint name="r" type="prismatic"><parent link="arm"/><child link="right"/>'
                '<origin xyz=".2 0 0"/><axis xyz="0 -1 0"/><mimic joint="l"/></joint></robot>')
            fk = IndependentUrdfFK(path)
            poses = fk.poses(('a', 'l', 'r'), [np.pi/2, .003, .004])
            np.testing.assert_allclose(poses['left'][:3, 3], [.8, -.003, 0], atol=1e-15)
            np.testing.assert_allclose(poses['right'][:3, 3], [.8, .004, 0], atol=1e-15)
            with self.assertRaises(ValueError):
                fk.poses(('a', 'l'), [0., .003])

    def test_axis_rotation_rejects_zero_axes(self):
        with self.assertRaises(ValueError):
            axis_rotation([0., 0, 0], .1)
        np.testing.assert_allclose(axis_rotation([0., 0, 1], np.pi/2)@[1., 0, 0], [0., 1., 0.], atol=1e-15)

    def test_positive_inradius_proves_solid_overlap_but_tangency_does_not(self):
        first = cube([0., 0, 0], [1., 1, 1])
        overlap = intersection_ball([first, first+[.5, 0, 0]])
        self.assertEqual(overlap['status'], 'VERIFIED_POSITIVE_INTERSECTION_BALL')
        self.assertAlmostEqual(overlap['radius_m'], .25, places=8)
        self.assertLessEqual(overlap['maximum_constraint_residual_m'], 0)
        tangent = intersection_ball([first, first+[1., 0, 0]])
        self.assertNotEqual(tangent['status'], 'VERIFIED_POSITIVE_INTERSECTION_BALL')
        separate = intersection_ball([first, first+[1.2, 0, 0]])
        self.assertNotEqual(separate['status'], 'VERIFIED_POSITIVE_INTERSECTION_BALL')

    def test_positive_plane_penetration_ball_and_small_overlap_are_retained(self):
        part = cube([-.1, -.1, -.05], [.1, .1, .1])
        witness = intersection_ball([part], [[0., 0, 1., 0.]])
        self.assertGreater(witness['radius_m'], .02)
        self.assertEqual(classify_relation(dict(lower_bound_m=-.05), witness), 'SOLID_INTERSECTION_WITNESS')
        small = dict(status='VERIFIED_POSITIVE_INTERSECTION_BALL', radius_m=1e-6)
        self.assertEqual(classify_relation(dict(lower_bound_m=0.), small),
                         'SMALL_NUMERIC_INTERSECTION_WITHIN_FRAME_ALLOWANCE')

    def test_free_model_rejection_is_conspicuous_and_route_failure_is_not_explained(self):
        relations = [dict(classification='SEPARATED_BEYOND_FRAME_ALLOWANCE'),
                     dict(classification='QUALIFIED_NONPENETRATING_MOUNT')]
        self.assertEqual(classify_candidate('invalid_goal', 'Configuration intersects the frozen planning world', relations),
                         'PHYSICALLY_CLEAR_ENDPOINT_REJECTED_BY_CONSERVATIVE_MODEL')
        self.assertEqual(classify_candidate('trajectory_failed', 'Configuration intersects the frozen planning world', relations),
                         'ENDPOINT_ONLY_CANNOT_DIAGNOSE_FAILED_ROUTE_OR_TRAJECTORY')
        self.assertEqual(classify_candidate('invalid_goal', 'IK forward-kinematics full-pose validation failed', relations),
                         'PHYSICALLY_CLEAR_ENDPOINT_OTHER_REJECTION_REASON')

    def test_mount_qualification_never_accepts_negative_signed_clearance(self):
        self.assertEqual(classify_relation(dict(lower_bound_m=0.), None, mounted_support=True),
                         'QUALIFIED_NONPENETRATING_MOUNT')
        self.assertNotEqual(classify_relation(dict(lower_bound_m=-1e-8), None, mounted_support=True),
                            'QUALIFIED_NONPENETRATING_MOUNT')

    def test_actual_all_pair_audit_includes_fingers_without_any_lula_flag(self):
        vertices = cube([-.01, -.01, -.01], [.01, .01, .01])
        names = ('panda_leftfinger', 'panda_rightfinger', 'panda_link6')
        parts = [dict(body=name, vertices=vertices) for name in names]
        poses = {name: np.eye(4) for name in names}
        for index, name in enumerate(names):
            poses[name][0, 3] = index*.1
        rows = physical_relations(parts, poses, [], frozenset())
        self.assertEqual(len(rows), 3)
        self.assertTrue(any(set(row['bodies']) == set(names[:2]) for row in rows))
        self.assertTrue(all(row['lower_bound_m'] > .079 for row in rows))


if __name__ == '__main__':
    unittest.main()
