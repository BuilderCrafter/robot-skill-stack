"""Compare direct execution with the frozen V5 loop, using identical SDK inputs."""
import ast
import asyncio
from dataclasses import asdict
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.backend import BackendResult
from tests.unit import test_franka_async_backend as fixtures


BASELINE = 'e8f6d6471196441cf162fc9828fb3ff15a993e99'
ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = 'robot_skill_stack/integrations/isaac/manipulation/franka_backend.py'


def frozen_move():
    """Use the original implementation as an oracle, without importing Isaac."""
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:{BACKEND_PATH}'], cwd=ROOT)
    backend = next(node for node in ast.parse(source).body
                   if isinstance(node, ast.ClassDef) and node.name == 'IsaacFrankaBackend')
    method = next(node for node in backend.body if isinstance(node, ast.AsyncFunctionDef)
                  and node.name == 'move_to_pose_async')
    namespace = dict(np=np, Pose=Pose, BackendResult=BackendResult, ArticulationAction=NS)
    exec(compile(ast.Module(body=[method], type_ignores=[]), f'{BASELINE}:{BACKEND_PATH}', 'exec'), namespace)
    return namespace['move_to_pose_async']


def canonical(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, BackendResult):
        return canonical(asdict(value))
    return value


class DirectV5EquivalenceTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_move = staticmethod(frozen_move())

    async def capture(self, scenario, *, frozen):
        backend, state, target = fixtures.FrankaCommandTests().adapter(reaches=False)
        events = []
        requests = dict(position_tolerance=None, orientation_tolerance=None, grasp_check=None)
        if scenario == 'frame':
            target = Pose(target.position, frame='robot')
        elif scenario == 'no_orientation':
            target = Pose(target.position, orientation=None)
        elif scenario == 'tolerances':
            requests.update(position_tolerance=.003, orientation_tolerance=.02)

        ticks = 0
        action_count = 0
        applied = 0
        entered, release = asyncio.Event(), asyncio.Event()

        def reachable(pose):
            events.append(('ik', canonical(pose.position), canonical(pose.orientation)))
            if scenario == 'ik_exception':
                raise RuntimeError('inverse kinematics failed')
            return scenario != 'unreachable'

        def playing():
            result = scenario != 'paused' and not (scenario == 'pause_resume' and ticks == 0)
            events.append(('playing', result))
            return result

        def reset():
            events.append(('reset',))

        def base(**kwargs):
            events.append(('base', canonical(kwargs)))

        def forward(**kwargs):
            nonlocal action_count
            events.append(('forward', canonical(kwargs)))
            action_count += 1
            if scenario == 'forward_exception':
                raise RuntimeError('controller failed')
            return NS(joint_positions=np.arange(7, dtype=float)/10+action_count/100,
                      joint_velocities=np.full(7, .18+action_count/100),
                      joint_indices=np.arange(7))

        def apply(action):
            nonlocal applied
            applied += 1
            events.append(('apply', canonical(action.joint_positions),
                           canonical(action.joint_velocities), canonical(action.joint_indices)))
            if scenario == 'apply_exception' and applied == 1:
                raise RuntimeError('articulation failed')

        async def update():
            nonlocal ticks
            events.append(('yield',))
            if scenario == 'cancel':
                entered.set()
                await release.wait()
            await asyncio.sleep(0)
            ticks += 1
            if scenario == 'yield_exception':
                raise RuntimeError('Kit update failed')
            if scenario not in ('timeout', 'paused') and ticks >= 2:
                state.position = target

        def current():
            events.append(('feedback',))
            if scenario == 'feedback_exception':
                raise RuntimeError('TCP feedback failed')
            return state.position

        guard_result = BackendResult(False, 'held object lost', details={'grasp_state': 'free'})

        def guard():
            events.append(('grasp_check',))
            return guard_result if scenario == 'grasp_failure' else None

        if scenario in ('grasp_failure', 'grasp_success'):
            requests['grasp_check'] = guard
        backend.check_reachability = reachable
        backend.world.is_playing = playing
        backend.motion_controller = NS(reset=reset, forward=forward,
            get_motion_policy=lambda: NS(set_robot_base_pose=base))
        backend.robot.apply_action = apply
        backend._next_update = update
        backend.get_end_effector_pose = current
        call = self.old_move(backend, target, **requests) if frozen else backend.move_to_pose_async(target, **requests)
        task = asyncio.create_task(call)
        if scenario == 'cancel':
            await entered.wait()
            task.cancel()
        try:
            result = ('result', canonical(await task))
        except BaseException as exc:
            result = ('exception', type(exc).__name__, str(exc))
        self.assertFalse(backend._motion_active)
        return result, events

    async def compare(self, *scenarios):
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                old = await self.capture(scenario, frozen=True)
                actual = await self.capture(scenario, frozen=False)
                self.assertEqual(actual, old)

    async def test_success_orientation_optional_and_explicit_tolerances_match_v5(self):
        await self.compare('success', 'no_orientation', 'tolerances', 'grasp_success')

    async def test_frame_and_ik_rejections_and_ik_exception_match_v5(self):
        await self.compare('frame', 'unreachable', 'ik_exception')

    async def test_timeout_paused_update_budget_and_pause_resume_match_v5(self):
        await self.compare('timeout', 'paused', 'pause_resume')

    async def test_grasp_failure_and_exact_final_hold_match_v5(self):
        await self.compare('grasp_failure')

    async def test_controller_apply_feedback_and_update_exceptions_match_v5(self):
        await self.compare('forward_exception', 'apply_exception', 'feedback_exception', 'yield_exception')

    async def test_task_cancellation_and_final_hold_match_v5(self):
        await self.compare('cancel')


class StrategyDelegationTests(unittest.IsolatedAsyncioTestCase):
    def adapter(self):
        return fixtures.FrankaCommandTests().adapter()

    async def test_constructor_injects_alternative_after_sdk_and_home_initialization(self):
        module = fixtures.load_backend_module()
        module.interface_config_loader.load_supported_motion_policy_config.return_value = {}
        module.interface_config_loader.load_supported_lula_kinematics_solver_config.return_value = {}
        module.ArticulationKinematicsSolver.return_value.compute_end_effector_pose.return_value = (
            np.array([.4, 0., .5]), np.eye(3))
        module.rot_matrices_to_quats.return_value = np.array([1., 0., 0., 0.])
        properties = np.zeros(9, dtype=[('lower', float), ('upper', float)])
        properties['upper'][7:] = .04
        robot = NS(get_dof_index=lambda name: {'panda_finger_joint1': 7,
                    'panda_finger_joint2': 8}[name], dof_properties=properties,
                    get_world_pose=lambda: (np.zeros(3), np.array([1., 0., 0., 0.])))
        calls, constructed, results = [], [], []

        class Alternative:
            async def move_to_pose_async(self, target, **kwargs):
                calls.append((target, kwargs))
                value = BackendResult(True, 'test strategy completed', details={'steps': len(calls)})
                results.append(value)
                return value

        alternative = Alternative()

        def factory(backend):
            self.assertIs(backend.robot, robot)
            self.assertIs(backend.motion_controller, module.MotionPolicyController.return_value)
            self.assertIs(backend.lula_solver, module.LulaKinematicsSolver.return_value)
            self.assertIs(backend.kinematics_solver, module.ArticulationKinematicsSolver.return_value)
            np.testing.assert_array_equal(backend.named_poses['home'].position, [.4, 0., .5])
            self.assertFalse(backend._motion_active)
            constructed.append(backend)
            return alternative

        backend = module.IsaacFrankaBackend(NS(get_physics_dt=lambda: 1/60), robot,
                                            strategy_factory=factory)
        self.assertEqual(constructed, [backend])
        self.assertIs(backend.motion_strategy, alternative)
        target, guard = Pose([.5, .1, .3]), Mock()
        moved = await backend.move_to_pose_async(target, .006, .02, guard)
        home = await backend.home_async(grasp_check=guard)
        self.assertIs(moved, results[0])
        self.assertIs(home, results[1])
        self.assertIs(calls[0][0], target)
        self.assertEqual(calls[0][1], dict(position_tolerance=.006,
                                        orientation_tolerance=.02, grasp_check=guard))
        self.assertIs(calls[1][0], backend.named_poses['home'])
        self.assertEqual(calls[1][1], dict(position_tolerance=.025,
                                        orientation_tolerance=.10, grasp_check=guard))
        module.MotionPolicyController.assert_called_once()
        module.LulaKinematicsSolver.assert_called_once()
        module.ArticulationKinematicsSolver.assert_called_once()
        module.MotionPolicyController.return_value.forward.assert_not_called()

    async def test_fake_strategy_receives_original_request_and_result_identity(self):
        backend, state, target = self.adapter()
        guard = Mock()
        expected = BackendResult(False, 'future planner rejected route', details={'planning': 'failed'})
        calls = []

        class Alternative:
            async def move_to_pose_async(self, target, position_tolerance=None,
                                         orientation_tolerance=None, grasp_check=None):
                calls.append((target, position_tolerance, orientation_tolerance, grasp_check))
                return expected

        backend._motion_strategy = Alternative()
        result = await backend.move_to_pose_async(target, .007, .03, guard)
        self.assertIs(result, expected)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], target)
        self.assertEqual(calls[0][1:3], (.007, .03))
        self.assertIs(calls[0][3], guard)
        backend.motion_controller.forward.assert_not_called()
        backend.robot.apply_action.assert_not_called()
        self.assertEqual(state.updates, 0)

    async def test_home_uses_injected_strategy_with_existing_pose_and_tolerances(self):
        backend, _, _ = self.adapter()
        guard = Mock()
        calls = []
        expected = BackendResult(True, 'alternative reached pose', details={'steps': 2})

        class Alternative:
            async def move_to_pose_async(self, target, **kwargs):
                calls.append((target, kwargs))
                return expected

        backend._motion_strategy = Alternative()
        result = await backend.home_async(grasp_check=guard)
        self.assertIs(result, expected)
        self.assertIs(calls[0][0], backend.named_poses['home'])
        self.assertEqual(calls[0][1], dict(position_tolerance=.025,
                                        orientation_tolerance=.10, grasp_check=guard))
        self.assertEqual(result.message, "Robot reached named pose 'home'.")
        self.assertEqual(result.details, {'steps': 2, 'pose_name': 'home'})

    async def test_only_one_backend_motion_can_enter_strategy_and_cancel_releases_owner(self):
        backend, _, target = self.adapter()
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        class Alternative:
            async def move_to_pose_async(self, requested, **kwargs):
                calls.append(requested)
                entered.set()
                await release.wait()
                return BackendResult(True)

        backend._motion_strategy = Alternative()
        task = asyncio.create_task(backend.move_to_pose_async(target))
        await entered.wait()
        rejected = await backend.move_to_pose_async(Pose([.3, .1, .2]))
        self.assertFalse(rejected.ok)
        self.assertIn('already in progress', rejected.message)
        self.assertEqual(calls, [target])
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(backend._motion_active)
        release.set()
        self.assertTrue((await backend.move_to_pose_async(target)).ok)

    async def test_strategy_exception_releases_owner_for_next_motion(self):
        backend, _, target = self.adapter()

        class FailsOnce:
            calls = 0

            async def move_to_pose_async(self, requested, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError('strategy execution failed')
                return BackendResult(False, 'normal failure', timed_out=True)

        backend._motion_strategy = FailsOnce()
        with self.assertRaisesRegex(RuntimeError, 'strategy execution failed'):
            await backend.move_to_pose_async(target)
        self.assertFalse(backend._motion_active)
        result = await backend.move_to_pose_async(target)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.message, 'normal failure')

    async def test_direct_stores_optional_mapping_provider_without_consulting_it(self):
        backend, _, target = self.adapter()
        module = fixtures.load_backend_module()
        provider = Mock(side_effect=AssertionError('Direct queried occupancy'))
        for mapping in (None, provider):
            with self.subTest(mapping=mapping):
                strategy = module.DirectRmpflowStrategy(backend, mapping_source_provider=mapping)
                self.assertIs(strategy.mapping_source_provider, mapping)
                backend._motion_strategy = strategy
                self.assertTrue((await backend.move_to_pose_async(target)).ok)
                with self.assertRaises(AttributeError):
                    strategy.mapping_source_provider = lambda: None
        provider.assert_not_called()

    async def test_selected_strategy_is_readonly(self):
        backend, _, _ = self.adapter()
        self.assertIs(backend.motion_strategy, backend._motion_strategy)
        with self.assertRaises(AttributeError):
            backend.motion_strategy = object()


if __name__ == '__main__':
    unittest.main()
