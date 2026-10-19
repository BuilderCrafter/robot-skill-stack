"""Exercise the adapter's command loops with SDK boundary doubles, not Isaac."""
import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.common.types import Pose


def load_backend():
    names = ('isaacsim', 'isaacsim.core', 'isaacsim.core.utils', 'isaacsim.core.utils.numpy',
             'isaacsim.core.utils.numpy.rotations', 'isaacsim.robot', 'isaacsim.robot.manipulators',
             'isaacsim.robot.manipulators.examples', 'isaacsim.robot.manipulators.examples.franka',
             'isaacsim.robot.manipulators.examples.franka.controllers',
             'isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller',
             'isaacsim.robot_motion', 'isaacsim.robot_motion.motion_generation')
    modules = {name: ModuleType(name) for name in names}
    modules[names[4]].rot_matrices_to_quats = Mock()
    modules[names[10]].RMPFlowController = Mock()
    modules['isaacsim.core.utils.types'] = ModuleType('isaacsim.core.utils.types')
    modules['isaacsim.core.utils.types'].ArticulationAction = NS
    for name in ('ArticulationKinematicsSolver', 'LulaKinematicsSolver', 'interface_config_loader'):
        setattr(modules[names[-1]], name, Mock())
    path = Path(__file__).resolve().parents[2]/'robot_skill_stack/integrations/isaac/manipulation/franka_backend.py'
    spec = importlib.util.spec_from_file_location('franka_commands_under_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', modules):
        spec.loader.exec_module(module)
    return module.IsaacFrankaBackend


class FrankaCommandTests(unittest.IsolatedAsyncioTestCase):
    def adapter(self, playing=True, reaches=True):
        cls = load_backend()
        backend = cls.__new__(cls)  # Supply SDK boundaries without creating a native controller.
        state = NS(updates=0, position=Pose([0, 0, 0]))
        backend.world = NS(is_playing=lambda: playing,
                           step=Mock(side_effect=AssertionError('blocking world.step used')))
        backend.robot = NS(handles_initialized=True, apply_action=Mock(),
                           gripper=NS(forward=Mock(return_value='gripper_action'),
                                      get_joint_positions=lambda: [.025, .025]))
        action = NS(joint_positions=np.arange(7, dtype=float)/10,
                    joint_velocities=np.full(7, .18), joint_indices=np.arange(7))
        backend.motion_controller = NS(reset=Mock(), forward=Mock(return_value=action))
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
        self.assertTrue(backend.verify_grasp().ok)
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
        self.assertEqual(result.details['requested_speed'], .35)
        self.assertEqual(state.updates, 2)
        self.assertFalse((await backend.home_async('missing')).ok)


if __name__ == '__main__':
    unittest.main()
