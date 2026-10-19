"""Exercise the adapter's command loops with SDK boundary doubles, not Isaac."""
import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from robot_skill_stack.common.types import Pose, GripperState
from robot_skill_stack.manipulation.backend import BackendResult


class ArticulationPolicyDouble:
    """Only the public SDK adapter API used by our feedback policy."""
    def __init__(self, robot, motion_policy, default_physics_dt):
        self.robot, self.policy, self.dt = robot, motion_policy, default_physics_dt

    def get_active_joints_subset(self):
        return self.robot.active

    def get_watched_joints_subset(self):
        return self.robot.watched

    def get_motion_policy(self):
        return self.policy

    def get_default_physics_dt(self):
        return self.dt


def load_backend_module():
    names = ('isaacsim', 'isaacsim.core', 'isaacsim.core.utils', 'isaacsim.core.utils.numpy',
             'isaacsim.core.utils.numpy.rotations', 'isaacsim.robot_motion',
             'isaacsim.robot_motion.motion_generation',
             'isaacsim.robot_motion.motion_generation.lula',
             'isaacsim.robot_motion.motion_generation.lula.motion_policies')
    modules = {name: ModuleType(name) for name in names}
    modules[names[4]].rot_matrices_to_quats = Mock()
    modules[names[-1]].RmpFlow = Mock()
    modules['isaacsim.core.utils.types'] = ModuleType('isaacsim.core.utils.types')
    modules['isaacsim.core.utils.types'].ArticulationAction = NS
    generation = modules['isaacsim.robot_motion.motion_generation']
    generation.ArticulationMotionPolicy = ArticulationPolicyDouble
    for name in ('ArticulationKinematicsSolver', 'LulaKinematicsSolver', 'interface_config_loader',
                 'MotionPolicyController'):
        setattr(generation, name, Mock())
    path = Path(__file__).resolve().parents[2]/'robot_skill_stack/integrations/isaac/manipulation/franka_backend.py'
    spec = importlib.util.spec_from_file_location('franka_commands_under_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', modules):
        spec.loader.exec_module(module)
    return module


def load_backend():
    return load_backend_module().IsaacFrankaBackend


class FrankaFingerLimitTests(unittest.TestCase):
    def test_constructor_reads_named_finger_limits_from_articulation(self):
        module = load_backend_module()
        module.interface_config_loader.load_supported_motion_policy_config.return_value = {}
        module.interface_config_loader.load_supported_lula_kinematics_solver_config.return_value = {}
        module.ArticulationKinematicsSolver.return_value.compute_end_effector_pose.return_value = (
            np.zeros(3), np.eye(3),
        )
        module.rot_matrices_to_quats.return_value = np.array([1., 0., 0., 0.])
        # Isaac Sim 5.1 SingleArticulation exposes structured dof_properties;
        # get_dof_limits() belongs to its underlying view, not Franka/Robot.
        properties = np.zeros(4, dtype=[
            ('type', int), ('hasLimits', bool), ('lower', float), ('upper', float),
            ('driveMode', int), ('maxVelocity', float), ('maxEffort', float),
            ('stiffness', float), ('damping', float),
        ])
        properties['lower'] = [-1., 0., -2., 0.]
        properties['upper'] = [1., .03, 2., .04]
        robot = NS(get_dof_index=Mock(side_effect={'panda_finger_joint1': 3,
                                                   'panda_finger_joint2': 1}.__getitem__),
                   dof_properties=properties,
                   get_world_pose=lambda: (np.zeros(3), np.array([1., 0., 0., 0.])),
                   gripper=NS(joint_opened_positions=np.array([.05, .05]),
                              get_joint_positions=lambda: [.04, .03]))
        backend = module.IsaacFrankaBackend(NS(get_physics_dt=lambda: 1/60), robot)
        # Home is a copy of the measured startup TCP, not an invented joint/Cartesian target.
        np.testing.assert_array_equal(backend.named_poses['home'].position, np.zeros(3))
        module.ArticulationKinematicsSolver.return_value.compute_end_effector_pose.return_value[0][:] = 1
        module.rot_matrices_to_quats.return_value[:] = 0
        np.testing.assert_array_equal(backend.named_poses['home'].position, np.zeros(3))
        np.testing.assert_array_equal(backend.named_poses['home'].orientation, [1., 0., 0., 0.])
        np.testing.assert_array_equal(backend._finger_limits, [[0., .04], [0., .03]])
        self.assertEqual(robot.get_dof_index.call_args_list,
                         [unittest.mock.call('panda_finger_joint1'),
                          unittest.mock.call('panda_finger_joint2')])
        self.assertEqual(backend.gripper_feedback().state, GripperState.FREE)
        properties['lower'][:] = 0
        properties['upper'][:] = 0  # Cached bounds are independent of the SDK buffer.
        np.testing.assert_array_equal(backend._finger_limits, [[0., .04], [0., .03]])


class FrankaFrameFeedbackTests(unittest.TestCase):
    def adapter(self):
        positions = np.zeros(7)
        active = NS(get_joint_positions=Mock(return_value=positions),
                    get_joint_velocities=Mock(side_effect=AssertionError('instantaneous velocity used')),
                    make_articulation_action=Mock(side_effect=lambda *targets: targets))
        watched = NS(get_joint_positions=Mock(return_value=np.array([])),
                     get_joint_velocities=Mock(return_value=np.array([])))
        world = NS(current_time=1., get_physics_dt=lambda: 1/60)
        policy = NS(compute_joint_targets=Mock(return_value=('positions', 'velocities')))
        cls = load_backend_module().FrankaFrameVelocityPolicy
        adapter = cls(NS(active=active, watched=watched), policy, world)
        return adapter, active, policy, world

    def test_contact_velocity_spikes_do_not_seed_rmpflow_rollout(self):
        adapter, active, policy, world = self.adapter()
        # Second native close: q7 barely changes while raw qd7 is 0.18..0.79 rad/s.
        active.get_joint_positions.return_value[6] = 1.214380264
        self.assertEqual(adapter.get_next_articulation_action(), ('positions', 'velocities'))
        np.testing.assert_array_equal(policy.compute_joint_targets.call_args.args[1], np.zeros(7))
        world.current_time += 1/60
        active.get_joint_positions.return_value[6] += .000002
        adapter.get_next_articulation_action()
        self.assertAlmostEqual(policy.compute_joint_targets.call_args.args[1][6], .00012)
        active.get_joint_velocities.assert_not_called()
        self.assertEqual(policy.compute_joint_targets.call_args.args[4], 1/60)
        np.testing.assert_array_equal(policy.compute_joint_targets.call_args.args[2], [])

    def test_real_motion_is_not_clipped_and_gripper_gap_uses_elapsed_simulation_time(self):
        adapter, active, policy, world = self.adapter()
        adapter.get_next_articulation_action()
        world.current_time += 1/60
        active.get_joint_positions.return_value[:] += .05
        adapter.get_next_articulation_action()
        np.testing.assert_allclose(adapter.joint_velocity_feedback, np.full(7, 3.))
        # No forward calls during gripper closure: divide by 1.5 s, not one frame.
        world.current_time += 1.5
        active.get_joint_positions.return_value[:] += .003
        adapter.get_next_articulation_action(physics_dt=.01)
        np.testing.assert_allclose(adapter.joint_velocity_feedback, np.full(7, .002))
        self.assertEqual(policy.compute_joint_targets.call_args.args[4], .01)

    def test_rmpflow_mutation_does_not_corrupt_measurement_history_or_trace(self):
        adapter, active, policy, world = self.adapter()
        def rollout(q, qd, watched_q, watched_qd, dt):
            q[:] += 100
            qd[:] += 200
            return q, qd
        policy.compute_joint_targets.side_effect = rollout
        adapter.get_next_articulation_action()
        world.current_time += 1/60
        active.get_joint_positions.return_value[:] += .01
        adapter.get_next_articulation_action()
        np.testing.assert_allclose(adapter.joint_velocity_feedback, np.full(7, .6))
        np.testing.assert_allclose(active.get_joint_positions.return_value, np.full(7, .01))
        targets = active.make_articulation_action.call_args.args
        np.testing.assert_allclose(targets[0], np.full(7, 100.01))
        np.testing.assert_allclose(targets[1], np.full(7, 200.6))

    def test_same_or_rewound_simulation_time_restarts_measurement_without_dividing_by_zero(self):
        adapter, active, policy, world = self.adapter()
        for now in (1., 1., 0.):
            world.current_time = now
            active.get_joint_positions.return_value[:] += .01
            adapter.get_next_articulation_action()
            np.testing.assert_array_equal(adapter.joint_velocity_feedback, np.zeros(7))
        world.current_time += 1/60
        active.get_joint_positions.return_value[:] += .01
        adapter.get_next_articulation_action()
        np.testing.assert_allclose(adapter.joint_velocity_feedback, np.full(7, .6))


class FrankaCommandTests(unittest.IsolatedAsyncioTestCase):
    def adapter(self, playing=True, reaches=True):
        cls = load_backend()
        backend = cls.__new__(cls)  # Supply SDK boundaries without creating a native controller.
        state = NS(updates=0, position=Pose([0, 0, 0]))
        backend.world = NS(is_playing=lambda: playing, current_time=0.,
                           step=Mock(side_effect=AssertionError('blocking world.step used')))
        backend._gripper_command = None
        backend._finger_limits = np.array([[0., .04], [0., .04]])
        backend.robot = NS(handles_initialized=True, apply_action=Mock(),
                           get_world_pose=lambda: (np.zeros(3), np.array([1., 0., 0., 0.])),
                           gripper=NS(forward=Mock(return_value='gripper_action'),
                                      joint_opened_positions=np.array([.05, .05]),
                                      get_joint_positions=lambda: [.04, .04] if backend._gripper_command == 'open'
                                      else [.025, .025]))
        action = NS(joint_positions=np.arange(7, dtype=float)/10,
                    joint_velocities=np.full(7, .18), joint_indices=np.arange(7))
        backend.motion_controller = NS(reset=Mock(), forward=Mock(return_value=action),
                                       get_motion_policy=lambda: NS(set_robot_base_pose=Mock()))
        backend.position_tolerance, backend.orientation_tolerance = .025, .10
        backend.max_motion_steps = 3
        backend.grasp_min_width, backend.grasp_max_width = .005, .075
        target = Pose([.4, 0, .2])
        backend.named_poses = {'home': target}
        backend.check_reachability = lambda pose: True
        backend.get_end_effector_pose = lambda: state.position
        async def next_update():
            await asyncio.sleep(0)
            state.updates += 1
            if playing:
                backend.world.current_time += 1/60
            if reaches and state.updates >= 2:
                state.position = target
        backend._next_update = next_update
        return backend, state, target

    async def test_motion_yields_before_observing_progress_without_blocking_step(self):
        backend, state, target = self.adapter()
        result = await backend.move_to_pose_async(target)
        self.assertTrue(result.ok)
        self.assertEqual(state.updates, 2)
        self.assertEqual(backend.robot.apply_action.call_count, 3)  # Two motions, one hold; no extra update.
        backend.world.step.assert_not_called()
        self.assertEqual(result.details['position_tolerance_m'], .025)
        self.assertEqual(result.details['orientation_tolerance_rad'], .10)
        self.assert_arm_hold(backend)

    async def test_motion_reset_restores_robot_base_pose(self):
        backend, state, target = self.adapter()
        policy = NS(set_robot_base_pose=Mock())
        backend.motion_controller.get_motion_policy = lambda: policy
        await backend.move_to_pose_async(target)
        backend.motion_controller.reset.assert_called_once_with()
        call = policy.set_robot_base_pose.call_args.kwargs
        np.testing.assert_array_equal(call['robot_position'], np.zeros(3))
        np.testing.assert_array_equal(call['robot_orientation'], [1, 0, 0, 0])

    def assert_arm_hold(self, backend):
        motion = backend.motion_controller.forward.return_value
        hold = backend.robot.apply_action.call_args.args[0]
        np.testing.assert_array_equal(hold.joint_positions, motion.joint_positions)
        np.testing.assert_array_equal(hold.joint_indices, np.arange(7))
        np.testing.assert_array_equal(hold.joint_velocities, np.zeros(7))
        np.testing.assert_array_equal(motion.joint_velocities, np.full(7, .18))  # Original action unchanged.

    async def test_motion_timeout_and_paused_world_keep_the_update_budget(self):
        for playing in (False, True):
            with self.subTest(playing=playing):
                backend, state, target = self.adapter(playing=playing, reaches=False)
                result = await backend.move_to_pose_async(target)
                self.assertTrue(result.timed_out)
                self.assertEqual(state.updates, 3)
                self.assertEqual(backend.robot.apply_action.call_count, 4 if playing else 0)
                if playing:
                    self.assert_arm_hold(backend)
                backend.world.step.assert_not_called()

    async def test_cancelled_motion_holds_only_arm_without_extra_updates_or_gripper_commands(self):
        backend, state, target = self.adapter()
        waiting = asyncio.Event()
        async def wait():
            waiting.set()
            await asyncio.Event().wait()
        backend._next_update = wait
        task = asyncio.create_task(backend.move_to_pose_async(target))
        await waiting.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(state.updates, 0)
        self.assertEqual(backend.robot.apply_action.call_count, 2)
        backend.robot.gripper.forward.assert_not_called()
        self.assert_arm_hold(backend)

    async def test_rejected_target_does_not_issue_motion_or_hold(self):
        for target in (Pose([.4, 0, .2], frame='robot'), Pose([.4, 0, .2])):
            backend, state, _ = self.adapter()
            backend.check_reachability = lambda pose: False
            result = await backend.move_to_pose_async(target)
            self.assertFalse(result.ok)
            self.assertEqual(state.updates, 0)
            backend.robot.apply_action.assert_not_called()

    async def test_gripper_commands_yield_existing_durations_and_keep_verification(self):
        backend, state, _ = self.adapter()
        self.assertTrue((await backend.open_gripper_async()).ok)
        self.assertEqual(state.updates, 60)
        self.assertTrue((await backend.close_gripper_async()).ok)
        self.assertEqual(state.updates, 150)
        self.assertEqual(backend.gripper_feedback().state.value, 'held')
        backend.world.step.assert_not_called()

    async def test_gripper_commands_follow_arm_hold_without_overwriting_arm_targets(self):
        backend, state, target = self.adapter()
        await backend.move_to_pose_async(target)
        self.assert_arm_hold(backend)
        hold_call_count = backend.robot.apply_action.call_count
        finger_action = NS(joint_positions=np.array([.04, .04]), joint_velocities=None,
                           joint_indices=np.array([7, 8]))
        backend.robot.gripper.forward.return_value = finger_action
        await backend.open_gripper_async()
        await backend.close_gripper_async()
        calls = backend.robot.apply_action.call_args_list[hold_call_count:]
        self.assertEqual(len(calls), 150)
        self.assertTrue(all(call.args[0] is finger_action for call in calls))
        self.assertEqual(state.updates, 152)

    async def test_readiness_and_home_use_the_adapter_contract(self):
        backend, state, _ = self.adapter()
        self.assertTrue(backend.is_ready)
        backend.robot.handles_initialized = False
        self.assertFalse(backend.is_ready)
        backend.robot.handles_initialized = True
        result = await backend.home_async()
        self.assertTrue(result.ok)
        self.assertEqual(result.details['pose_name'], 'home')
        self.assertEqual(state.updates, 2)
        self.assertFalse((await backend.home_async('missing')).ok)

    async def test_named_home_forwards_exact_pose_guard_tolerances_and_failure(self):
        for timed_out in (False, True):
            with self.subTest(timed_out=timed_out):
                backend, _, _ = self.adapter()
                pose = Pose([.4, .1, .2])
                backend.named_poses['test_pose'] = pose
                failure = BackendResult(False, 'controller failed', timed_out=timed_out, details={'steps': 3})
                backend.move_to_pose_async = AsyncMock(return_value=failure)
                guard = Mock()
                result = await backend.home_async('test_pose', grasp_check=guard)
                self.assertIs(result, failure)
                self.assertEqual(result.message, 'controller failed')
                self.assertEqual(result.details, {'steps': 3, 'pose_name': 'test_pose'})
                backend.move_to_pose_async.assert_awaited_once_with(
                    pose, position_tolerance=.025, orientation_tolerance=.10, grasp_check=guard)

    async def test_unknown_named_pose_issues_no_command(self):
        backend, state, _ = self.adapter()
        backend.move_to_pose_async = AsyncMock()
        result = await backend.home_async('missing')
        self.assertFalse(result.ok)
        self.assertEqual(result.details['available_poses'], ['home'])
        self.assertEqual(state.updates, 0)
        backend.move_to_pose_async.assert_not_called()
        backend.robot.apply_action.assert_not_called()

    async def test_position_feedback_distinguishes_empty_held_opening_and_invalid(self):
        backend, _, _ = self.adapter()
        for command, positions, expected in (
            ('close', [.025, .025], GripperState.HELD),
            ('close', [.0375, .0375], GripperState.HELD),
            ('close', [0., 0.], GripperState.FREE),
            ('close', [.04, .04], GripperState.FREE),
            ('open', [.025, .025], GripperState.UNCERTAIN),
            ('close', [np.nan, .025], GripperState.UNCERTAIN),
            ('close', [-1., -1.], GripperState.UNCERTAIN),
            ('close', [.2, .2], GripperState.UNCERTAIN),
            ('close', [.025], GripperState.UNCERTAIN),
        ):
            with self.subTest(command=command, positions=positions):
                backend._gripper_command = command
                backend.robot.gripper.get_joint_positions = lambda: positions
                self.assertEqual(backend.gripper_feedback().state, expected)
        backend.robot.gripper.get_joint_positions = Mock(side_effect=RuntimeError('sensor unavailable'))
        self.assertEqual(backend.gripper_feedback().state, GripperState.UNCERTAIN)

    async def test_stationary_inspection_is_stable_bounded_and_never_commands_motion(self):
        for playing, invalid, expected, updates in (
            (True, False, GripperState.HELD, 2),
            (True, True, GripperState.UNCERTAIN, 12),
            (False, False, GripperState.UNCERTAIN, 12),
        ):
            with self.subTest(playing=playing, invalid=invalid):
                backend, state, _ = self.adapter(playing=playing)
                backend._gripper_command = 'close'
                if invalid:
                    backend.robot.gripper.get_joint_positions = lambda: [np.nan, .025]
                feedback = await backend.inspect_gripper_async()
                self.assertEqual(feedback.state, expected)
                self.assertEqual(state.updates, updates)
                backend.robot.apply_action.assert_not_called()
                backend.robot.gripper.forward.assert_not_called()

    async def test_changing_finger_width_does_not_prematurely_confirm_a_grasp(self):
        backend, state, _ = self.adapter()
        backend._gripper_command = 'close'
        # Closing fingers pass through the held-width range before settling.
        backend.robot.gripper.get_joint_positions = lambda: [.035-.003*min(state.updates, 4)]*2
        feedback = await backend.inspect_gripper_async()
        self.assertEqual(feedback.state, GripperState.HELD)
        self.assertEqual(state.updates, 6)
        self.assertAlmostEqual(feedback.width, .046)

    async def test_motion_guard_stops_before_next_action_and_applies_existing_arm_hold(self):
        backend, state, target = self.adapter(reaches=False)
        failure = BackendResult(False, 'Object lost', details={'grasp_failure': True, 'grasp_state': 'free'})
        check = Mock(side_effect=[None, None, failure])
        result = await backend.move_to_pose_async(target, grasp_check=check)
        self.assertIs(result, failure)
        self.assertEqual(state.updates, 3)
        self.assertEqual(check.call_count, 3)
        self.assertEqual(backend.robot.apply_action.call_count, 4)  # Three motions and the existing hold.
        self.assert_arm_hold(backend)

    async def test_open_command_duration_is_not_treated_as_verified_release(self):
        backend, state, _ = self.adapter()
        backend.robot.gripper.get_joint_positions = lambda: [.025, .025]
        result = await backend.open_gripper_async()
        self.assertFalse(result.ok)
        self.assertEqual(state.updates, 60)

    async def test_open_verification_uses_physical_limits_when_sdk_target_exceeds_travel(self):
        backend, state, _ = self.adapter()
        # Native failure: measured 0.0399999991 m per finger, despite an SDK
        # opening request beyond the USD's 0.04 m physical travel.
        backend.robot.gripper.get_joint_positions = lambda: [0.03999999910593033]*2
        result = await backend.open_gripper_async()
        self.assertTrue(result.ok, result.details)
        self.assertEqual(state.updates, 60)
        self.assertEqual(result.details['grasp_state'], 'free')
        self.assertEqual(result.details['finger_upper_limits_m'], [.04, .04])
        self.assertEqual(result.details['nominal_open_positions_m'], [.05, .05])
        feedback = await backend.inspect_gripper_async()
        self.assertEqual(feedback.state, GripperState.FREE)


if __name__ == '__main__':
    unittest.main()
