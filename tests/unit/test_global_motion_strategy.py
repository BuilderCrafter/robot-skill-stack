"""Execution ownership, clocks, full-pose results and cleanup without Isaac."""
import asyncio
from types import SimpleNamespace
import unittest

import numpy as np

from robot_skill_stack.common.types import GripperState, Pose
from robot_skill_stack.integrations.isaac.manipulation.motion.global_rrt import GlobalRrtStrategy
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure


ARM = tuple(f"panda_joint{i}" for i in range(1, 8))


class Robot:
    dof_names = ("panda_finger_joint1", *reversed(ARM), "panda_finger_joint2")
    prim_path = "/Franka"
    def __init__(self):
        self.q = np.zeros(9)
        self.qd = np.zeros(9)
        self.q[[0, 8]] = .04
        self.actions, self.follow = [], True

    def get_joint_positions(self):
        return self.q.copy()

    def get_world_pose(self):
        return np.zeros(3), np.array([1., 0, 0, 0])

    def get_joint_velocities(self):
        return self.qd.copy()

    def apply_action(self, action):
        self.actions.append(action)
        if self.follow:
            self.q[action.joint_indices] = action.joint_positions


class Backend:
    position_tolerance, orientation_tolerance = .025, .10
    max_motion_steps = 1000
    is_ready = True
    def __init__(self):
        self.robot = Robot()
        self.world = SimpleNamespace(current_time=1., current_time_step_index=60)
        self.playing, self.freeze, self.updates = True, False, 0
        self.world.is_playing = lambda: self.playing
        self.feedback = GripperState.FREE
        self.on_update, self.pose_error = None, 0.

    def gripper_feedback(self):
        return SimpleNamespace(state=self.feedback)

    async def _next_update(self):
        await asyncio.sleep(0)
        self.updates += 1
        if not self.freeze:
            self.world.current_time += .05
            self.world.current_time_step_index += 1
        if self.on_update:
            self.on_update(self)

    def get_end_effector_pose(self):
        return Pose([float(self.robot.q[1])+self.pose_error, 0, 0], [1., 0, 0, 0])

    @staticmethod
    def _quat_error(first, second):
        return 2.*np.arccos(np.clip(abs(np.dot(first, second)), -1., 1.))


class Planner:
    joint_names = ARM
    def __init__(self):
        self.closed, self.error, self.close_error = False, None, False
        self.no_motion, self.calls = False, []
        self.diagnostics = {"rrt_seconds": .01}

    def plan(self, target, start, snapshot, **kwargs):
        self.calls.append((target, start.copy(), kwargs))
        if self.error:
            raise self.error
        duration = 0. if self.no_motion else 1.
        diagnostics = dict(self.diagnostics, already_at_goal=self.no_motion)
        trajectory = SimpleNamespace(get_joint_targets=lambda t: (np.full(7, .1*t), np.full(7, .1)))
        return SimpleNamespace(joint_names=ARM, route=np.array([np.zeros(7), np.full(7, .1)]),
            trajectory=trajectory, duration=duration, diagnostics=diagnostics)

    def close(self):
        self.closed = True
        if self.close_error:
            raise RuntimeError("cleanup test failure")


class SolverScope:
    def __init__(self):
        self.closed, self.closes = False, 0

    @property
    def diagnostics(self):
        return dict(requested_position_iterations_during=255,
            original_session_opinion_restored=self.closed)

    def close(self):
        if not self.closed:
            self.closed = True
            self.closes += 1


class GlobalMotionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.backend, self.planner = Backend(), Planner()
        self.factory_calls = []
        self.snapshot_calls = 0
        self.solver_scopes = []
        def factory(backend, **kwargs):
            self.factory_calls.append(kwargs)
            return self.planner
        def snapshot(backend):
            self.snapshot_calls += 1
            return SimpleNamespace(meters_per_unit=1.)
        def scope(backend):
            lease = SolverScope()
            self.solver_scopes.append(lease)
            return lease
        self.strategy = GlobalRrtStrategy(self.backend, planner_factory=factory,
            world_capture=snapshot, action_factory=lambda **kwargs: SimpleNamespace(**kwargs),
            solver_scope_factory=scope)
        self.target = Pose([.1, 0, 0], [1., 0, 0, 0])

    async def test_joint_execution_uses_physics_time_and_preserves_gripper(self):
        result = await self.strategy.move_to_pose_async(self.target)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(self.snapshot_calls, 1)
        self.assertEqual(self.factory_calls, [dict(seed=123456, max_iterations=10000)])
        self.assertTrue(self.planner.closed)
        self.assertTrue(result.details["final_hold"])
        self.assertTrue(np.array_equal(self.backend.robot.q[[0, 8]], [.04, .04]))
        for action in self.backend.robot.actions:
            self.assertEqual(tuple(action.joint_indices), (7, 6, 5, 4, 3, 2, 1))
        self.assertTrue(np.all(self.backend.robot.actions[-1].joint_velocities == 0))
        self.assertTrue(np.allclose(self.backend.robot.actions[-1].joint_positions, .1))
        self.assertGreaterEqual(result.details["execution_sim_seconds"], 1.)
        self.assertTrue(result.details["execution_solver"]["original_session_opinion_restored"])
        self.assertEqual(self.planner.calls[0][0].orientation.tolist(), [1., 0, 0, 0])

    async def test_invalid_full_pose_rejected_before_snapshot_or_action(self):
        for pose in (Pose([.1, 0, 0]), Pose([.1, 0, 0], frame="robot"),
                     Pose([float("nan"), 0, 0], [1, 0, 0, 0])):
            result = await self.strategy.move_to_pose_async(pose)
            self.assertFalse(result.ok)
            self.assertEqual(result.details["failure_code"], "invalid_target")
        self.assertEqual(self.snapshot_calls, 0)
        self.assertEqual(self.backend.robot.actions, [])
        self.assertEqual(self.solver_scopes, [])

    async def test_payload_or_uncertain_grasp_rejected_without_planning(self):
        for state in (GripperState.HELD, GripperState.UNCERTAIN):
            self.backend.feedback = state
            result = await self.strategy.move_to_pose_async(self.target)
            self.assertEqual(result.details["failure_code"], "unsupported_payload")
        self.assertEqual(self.factory_calls, [])
        self.assertEqual(self.backend.robot.actions, [])

    async def test_planning_failures_never_apply_an_arm_command(self):
        for code in ("invalid_start", "invalid_goal", "ik_failed", "search_exhausted", "trajectory_failed"):
            self.planner.error = PlanningFailure(code, "deliberate rejection")
            result = await self.strategy.move_to_pose_async(self.target)
            self.assertFalse(result.ok)
            self.assertEqual(result.details["failure_code"], code)
            self.assertGreaterEqual(result.details["planning_and_validation_wall_seconds"], 0.)
            self.assertFalse(result.details["final_hold"])
            self.assertTrue(self.planner.closed)
        self.assertEqual(self.backend.robot.actions, [])

    async def test_snapshot_failure_is_explicit_without_fallback(self):
        def unsupported(backend):
            raise ValueError("unsupported physical mesh")
        self.strategy._world_capture = unsupported
        result = await self.strategy.move_to_pose_async(self.target)
        self.assertEqual(result.details["failure_code"], "unsupported_geometry")
        self.assertEqual(self.backend.robot.actions, [])

    async def test_cancellation_propagates_and_holds_last_arm_target(self):
        task = asyncio.create_task(self.strategy.move_to_pose_async(self.target))
        await asyncio.sleep(0)
        self.assertTrue(self.backend.robot.actions)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.strategy.diagnostics["state"], "cancelled")
        self.assertTrue(self.planner.closed)
        self.assertTrue(np.all(self.backend.robot.actions[-1].joint_velocities == 0))
        self.assertTrue(self.solver_scopes[0].closed)

    async def test_stop_reset_or_handle_invalidation_prevents_stale_hold(self):
        for mode in ("stop", "reset", "handle"):
            with self.subTest(mode=mode):
                self.setUp()
                def invalidate(backend):
                    if mode == "stop":
                        backend.playing = False
                    elif mode == "reset":
                        backend.world.current_time, backend.world.current_time_step_index = 0., 0
                    else:
                        backend.is_ready = False
                self.backend.on_update = invalidate
                result = await self.strategy.move_to_pose_async(self.target)
                self.assertEqual(result.details["failure_code"], "lifecycle_invalidated")
                self.assertFalse(result.details["final_hold"])
                self.assertEqual(len(self.backend.robot.actions), 1)
                self.assertTrue(self.planner.closed)
                self.assertTrue(self.solver_scopes[0].closed)

    async def test_tracking_failure_stops_and_preserves_hold(self):
        self.backend.robot.follow = False
        self.planner.plan = lambda *a, **k: SimpleNamespace(joint_names=ARM,
            route=np.array([np.zeros(7), np.ones(7)]), duration=1., diagnostics={},
            trajectory=SimpleNamespace(get_joint_targets=lambda t: (np.full(7, t), np.ones(7))))
        result = await self.strategy.move_to_pose_async(self.target)
        self.assertEqual(result.details["failure_code"], "tracking_failure")
        self.assertTrue(result.details["final_hold"])
        self.assertTrue(self.planner.closed)

    async def test_final_pose_failure_keeps_request_tolerances(self):
        self.backend.pose_error = 1.
        result = await self.strategy.move_to_pose_async(self.target)
        self.assertEqual(result.details["failure_code"], "final_pose_failure")
        self.assertEqual(result.details["position_tolerance_m"], .025)
        self.assertGreater(result.details["position_error_m"], .9)

    async def test_no_motion_home_still_checks_actual_full_pose(self):
        self.planner.no_motion = True
        result = await self.strategy.move_to_pose_async(Pose([0, 0, 0], [1., 0, 0, 0]))
        self.assertTrue(result.ok)
        self.assertEqual(result.details["commanded_updates"], 0)
        self.assertEqual(self.backend.robot.actions, [])
        self.assertTrue(self.planner.closed)
        self.assertEqual(self.solver_scopes, [])

    async def test_success_waits_for_measured_velocity_within_original_settle_window(self):
        self.backend.robot.qd[1:8] = .2
        def settle(backend):
            if backend.world.current_time >= 2.5:
                backend.robot.qd[:] = 0.
        self.backend.on_update = settle
        result = await self.strategy.move_to_pose_async(self.target)
        self.assertTrue(result.ok, result.message)
        self.assertGreaterEqual(result.details['execution_sim_seconds'], 1.5)
        self.assertEqual(result.details['terminal_velocity_limit_rad_s'], .01)
        self.assertEqual(result.details['measured_arm_velocity_rad_s'], 0.)
        self.assertEqual(self.strategy.settle_seconds, 3.)
        self.assertTrue(self.solver_scopes[0].closed)

    async def test_pose_reach_with_persistent_velocity_cannot_report_success(self):
        self.backend.robot.qd[1:8] = .2
        result = await self.strategy.move_to_pose_async(self.target)
        self.assertFalse(result.ok)
        self.assertEqual(result.details['failure_code'], 'final_pose_failure')
        self.assertGreater(result.details['measured_arm_velocity_rad_s'], .01)
        self.assertTrue(result.details['final_hold'])
        self.assertTrue(self.solver_scopes[0].closed)

    async def test_brief_low_velocity_at_a_turning_point_does_not_complete_motion(self):
        self.backend.robot.qd[1:8] = .2
        def oscillation(backend):
            elapsed = backend.world.current_time-1.
            backend.robot.qd[1:8] = 0. if 1.1 <= elapsed < 1.2 or elapsed >= 1.6 else .2
        self.backend.on_update = oscillation
        result = await self.strategy.move_to_pose_async(self.target)
        self.assertTrue(result.ok, result.message)
        self.assertGreaterEqual(result.details['execution_sim_seconds'], 1.85)
        self.assertEqual(result.details['terminal_settle_seconds'], .25)
        self.assertEqual(self.strategy.settle_seconds, 3.)

    async def test_invalid_measured_velocity_fails_before_first_command_and_restores_scope(self):
        self.backend.robot.qd[:] = np.nan
        result = await self.strategy.move_to_pose_async(self.target)
        self.assertEqual(result.details['failure_code'], 'execution_failure')
        self.assertEqual(self.backend.robot.actions, [])
        self.assertTrue(self.solver_scopes[0].closed)

    async def test_solver_activation_failure_rejects_before_first_command(self):
        def unavailable(backend):
            raise PlanningFailure('solver_configuration_failure', 'Required accuracy unavailable')
        self.strategy._solver_scope_factory = unavailable
        result = await self.strategy.move_to_pose_async(self.target)
        self.assertEqual(result.details['failure_code'], 'solver_configuration_failure')
        self.assertEqual(self.backend.robot.actions, [])
        self.assertTrue(self.planner.closed)

    async def test_close_during_execution_restores_scope_once_and_prevents_stale_hold(self):
        self.backend.on_update = lambda backend: self.strategy.close()
        result = await self.strategy.move_to_pose_async(self.target)
        self.assertEqual(result.details['failure_code'], 'lifecycle_invalidated')
        self.assertTrue(self.solver_scopes[0].closed)
        self.assertEqual(self.solver_scopes[0].closes, 1)
        self.assertFalse(result.details['final_hold'])

    async def test_failed_solver_cleanup_is_retained_and_blocks_new_commands_until_recovered(self):
        lease = SolverScope()
        attempts = 0
        def cleanup():
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise RuntimeError('Transient cleanup failure')
            lease.closed = True
        lease.close = cleanup
        original_factory = self.strategy._solver_scope_factory
        self.strategy._solver_scope_factory = lambda backend: lease
        first = await self.strategy.move_to_pose_async(self.target)
        self.assertEqual(first.details['failure_code'], 'cleanup_failure')
        self.assertIs(self.strategy._solver_scope, lease)
        before = len(self.backend.robot.actions)
        second = await self.strategy.move_to_pose_async(self.target)
        self.assertEqual(second.details['failure_code'], 'cleanup_failure')
        self.assertEqual(len(self.backend.robot.actions), before)
        self.assertIs(self.strategy._solver_scope, lease)
        self.strategy._solver_scope_factory = original_factory
        third = await self.strategy.move_to_pose_async(self.target)
        self.assertTrue(third.ok, third.message)
        self.assertEqual(attempts, 3)
        self.assertIsNone(self.strategy._solver_scope)

    async def test_repeated_close_retries_failed_solver_cleanup(self):
        lease = SolverScope()
        attempts = 0
        def cleanup():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError('Transient cleanup failure')
            lease.closed = True
        lease.close = cleanup
        self.strategy._solver_scope = lease
        with self.assertRaises(RuntimeError):
            self.strategy.close()
        self.assertTrue(self.strategy.diagnostics['closed'])
        self.assertIs(self.strategy._solver_scope, lease)
        self.strategy.close()
        self.strategy.close()
        self.assertEqual(attempts, 2)
        self.assertIsNone(self.strategy._solver_scope)

    async def test_cleanup_failure_cannot_report_success(self):
        self.planner.close_error = True
        result = await self.strategy.move_to_pose_async(self.target)
        self.assertFalse(result.ok)
        self.assertEqual(result.details["failure_code"], "cleanup_failure")

    async def test_closed_strategy_rejects_without_actions(self):
        self.strategy.close()
        self.strategy.close()
        result = await self.strategy.move_to_pose_async(self.target)
        self.assertFalse(result.ok)
        self.assertEqual(self.backend.robot.actions, [])

    async def test_shutdown_invalidates_and_releases_planner_even_if_hold_fails(self):
        self.strategy._active_planner = self.planner
        self.strategy._last_action = SimpleNamespace(joint_indices=np.arange(1, 8))
        def failed_action(action):
            raise RuntimeError("invalidated articulation during shutdown")
        self.backend.robot.apply_action = failed_action
        with self.assertRaises(RuntimeError):
            self.strategy.close()
        self.assertTrue(self.strategy.diagnostics["closed"])
        self.assertTrue(self.planner.closed)
        self.strategy.close()


if __name__ == "__main__":
    unittest.main()
