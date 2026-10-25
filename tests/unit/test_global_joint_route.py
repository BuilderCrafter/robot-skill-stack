import unittest

import numpy as np

from robot_skill_stack.manipulation.motion.joint_route import (
    JointRoute, PlanningFailure, check_joint_limits, interpolate_route,
    map_arm_joints, simulation_progress,
)


ARM = tuple(f"panda_joint{i}" for i in range(1, 8))


class JointRouteTests(unittest.TestCase):
    def test_order_follows_planner_and_excludes_fingers(self):
        articulation = ("panda_finger_joint1", *reversed(ARM), "panda_finger_joint2")
        self.assertEqual(map_arm_joints(ARM, articulation), (7, 6, 5, 4, 3, 2, 1))
        self.assertEqual(map_arm_joints(tuple(reversed(ARM)), articulation), tuple(range(1, 8)))

    def test_ambiguous_missing_or_finger_planner_joint_rejected(self):
        for names, articulation in ((ARM[:-1], ARM), (ARM, ARM[:-1]),
                (ARM, (*ARM, ARM[0])), ((*ARM[:-1], "panda_finger_joint1"), ARM)):
            with self.subTest(names=names, articulation=articulation), self.assertRaises(PlanningFailure):
                map_arm_joints(names, articulation)

    def test_every_route_segment_checked_including_short_and_duplicate(self):
        route = np.array([[0.]*7, [.031, 0, 0, 0, 0, 0, 0], [.031]*7, [.031]*7])
        dense = interpolate_route(route)
        self.assertTrue(np.all(np.max(np.abs(np.diff(dense, axis=0)), axis=1) <= .01+1e-12))
        for point in route:
            self.assertTrue(np.any(np.all(dense == point, axis=1)))
        self.assertTrue(np.array_equal(dense[0], route[0]))
        self.assertTrue(np.array_equal(dense[-1], route[-1]))

    def test_nonfinite_malformed_and_excessive_routes_fail_closed(self):
        for route in ([], [[0.]*6], [[float("nan")]*7], [0.]*7):
            with self.subTest(route=route), self.assertRaises(PlanningFailure):
                interpolate_route(route)
        with self.assertRaises(PlanningFailure) as caught:
            interpolate_route([[0.]*7, [1.]*7], max_samples=10)
        self.assertEqual(caught.exception.code, "validation_budget_exhausted")

    def test_joint_limits_apply_to_all_seven_coordinates(self):
        check_joint_limits([[0.]*7], [-1.]*7, [1.]*7)
        for index in range(7):
            q = np.zeros(7); q[index] = 1.01
            with self.assertRaises(PlanningFailure):
                check_joint_limits([q], [-1.]*7, [1.]*7)
        with self.assertRaises(PlanningFailure):
            check_joint_limits([[0.]*7], [-1.]*6, [1.]*7)

    def test_immutable_route_preserves_seven_joint_geometry(self):
        values = np.array([[0.]*7, [1., 0, 0, 0, 0, 0, 0]])
        route = JointRoute(ARM, values)
        values[:] = 9
        self.assertEqual(route.length_rad, 1.)
        self.assertEqual(route.waypoints[0], (0.,)*7)

    def test_simulation_reset_or_stall_invalidates_execution(self):
        self.assertTrue(simulation_progress(1., 1.1, 60, 66,
            maximum_pause_seconds=2., stalled_wall_seconds=0.))
        for now, index, stalled in ((0., 0, 0.), (1., 59, 0.), (1., 60, 2.)):
            with self.assertRaises(PlanningFailure):
                simulation_progress(1., now, 60, index,
                    maximum_pause_seconds=2., stalled_wall_seconds=stalled)


if __name__ == "__main__":
    unittest.main()
