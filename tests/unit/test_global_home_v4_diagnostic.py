"""A causal timing probe must retain geometry, limits and collision validation."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.integrations.isaac.manipulation.motion.lula_global import LulaPlanResult
from scripts.diagnose_global_home_v4 import time_scaled_planner_factory


class PolynomialTrajectory:
    start_time, end_time = 0., 1.

    @property
    def trajectory(self):
        return self

    def eval(self, time, order):
        return np.full(7, (time**3, 3*time**2, 6*time, 6.)[order])

    def get_joint_targets(self, time):
        return self.eval(time, 0), self.eval(time, 1)


class GlobalHomeDiagnosticTests(unittest.TestCase):
    def fixture(self, *, duration=1.):
        route = np.array([[0.]*7, [1.]*7])
        limits = [np.full(7, value) for value in (-2., 2., 4., 8., 10.)]
        result = LulaPlanResult(tuple(f'panda_joint{i}' for i in range(1, 8)),
            route, PolynomialTrajectory(), duration, *limits,
            dict(trajectory_validation={'nominal': True}))
        validation = Mock(return_value={'scaled': True})
        original_plan = Mock(return_value=result)
        planner = SimpleNamespace(plan=original_plan, _last_diagnostics={},
            _config={'urdf_path': 'unused'}, _validate_trajectory=validation)
        backend = SimpleNamespace(robot=SimpleNamespace(get_world_pose=lambda:
            (np.array([100., 0., 0.]), np.array([1., 0., 0., 0.]))))
        return result, planner, backend, original_plan, validation

    def test_only_timing_changes_and_original_collision_context_is_reused(self):
        original, planner, backend, planned, validate = self.fixture()
        capture = {}
        exclusions = frozenset({frozenset({'link1', 'link2'})})
        with patch('robot_skill_stack.integrations.isaac.manipulation.motion.lula_global.adjacent_self_collision_pairs',
                   return_value=exclusions):
            factory = time_scaled_planner_factory(lambda *args, **kwargs: planner, 2., capture)
        snapshot = SimpleNamespace(meters_per_unit=.01)
        selected = factory(backend)
        scaled = selected.plan('target', np.zeros(7), snapshot, .025, .10)
        planned.assert_called_once()
        self.assertIs(scaled.route, original.route)
        self.assertIs(scaled.velocity_limits, original.velocity_limits)
        self.assertEqual(scaled.duration, 2.)
        args = validate.call_args.args
        self.assertIs(args[0], scaled.trajectory)
        self.assertIs(args[1], snapshot)
        np.testing.assert_array_equal(args[2], [1., 0., 0.])
        self.assertIs(args[4], exclusions)
        for actual, expected in zip(args[5:], (original.lower_limits, original.upper_limits,
                original.velocity_limits, original.acceleration_limits, original.jerk_limits)):
            self.assertIs(actual, expected)
        proof = capture['home_time_scaling']
        self.assertEqual(proof['maximum_identity_errors_for_orders0_through3'], [0.]*4)
        self.assertTrue(proof['hard_limits_unchanged'])
        self.assertEqual(scaled.diagnostics['trajectory_validation'], {'scaled': True})

    def test_no_motion_home_cannot_be_presented_as_moving_timing_probe(self):
        _, planner, backend, _, validate = self.fixture(duration=0.)
        capture = {}
        factory = time_scaled_planner_factory(lambda *args, **kwargs: planner, 2., capture)
        selected = factory(backend)
        with self.assertRaisesRegex(RuntimeError, 'moving Home'):
            selected.plan('target', np.zeros(7), SimpleNamespace(meters_per_unit=1.), .025, .10)
        validate.assert_not_called()
        self.assertNotIn('home_time_scaling', capture)

    def test_preferred_home_configuration_is_forwarded_to_normal_planner_validation(self):
        _, planner, backend, planned, validate = self.fixture()
        capture = {}
        with patch('robot_skill_stack.integrations.isaac.manipulation.motion.lula_global.adjacent_self_collision_pairs',
                   return_value=frozenset()):
            factory = time_scaled_planner_factory(lambda *args, **kwargs: planner, 2., capture)
        selected = factory(backend)
        hint = np.linspace(-.2, .2, 7)
        selected.plan('target', np.zeros(7), SimpleNamespace(meters_per_unit=1.), .025, .10,
            preferred_goal_configuration=hint)
        self.assertIs(planned.call_args.kwargs['preferred_goal_configuration'], hint)
        validate.assert_called_once()
        self.assertIn('home_time_scaling', capture)


if __name__ == '__main__':
    unittest.main()
