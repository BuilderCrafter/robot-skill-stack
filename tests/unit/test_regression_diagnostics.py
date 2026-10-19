"""Exercise scenario assertions/reporting with doubles; never start native Isaac."""
import ast
import asyncio
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import tempfile
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.backend import BackendResult
from robot_skill_stack.manipulation.skills.place import PlaceSkill
from robot_skill_stack.runtime.skill import FailureCode, SkillResult, SkillStatus
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.world_model import WorldModel
from tests.support.results import write_result
from tests.unit.test_placement import FakeBackend

ROOT = Path(__file__).resolve().parents[2]


def scenario(name, folder='tests/regression'):
    # Load the actual scenario functions/constants without its native entry-point
    # imports or SimulationApp construction. Dependencies below are explicit doubles.
    path = ROOT/folder/f'{name}.py'
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or
             (isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets))]
    module = ModuleType(name)
    module.__dict__.update(np=np, Path=Path, Pose=Pose, FailureCode=FailureCode, time=time,
                           ExitStack=ExitStack, contextmanager=contextmanager, patch=patch,
                           write_result=write_result, simulation_app=object())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


class RegressionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.result = self.out/'result.json'

    def benchmark(self, failed='place', placement_offset=0.):
        module = scenario('test_benchmark')
        model = WorldModel()
        obj = WorldObject('cube', 'cube', Pose([.45, 0, .025]), [.05]*3, visible=True)
        model.register(obj)
        world, updater = NS(current_time=0., current_time_step_index=0), NS(update_count=0, last_error=None)
        bundle = NS(world=world, world_model=model, world_updater=updater,
                    objects={'cube': NS(get_world_pose=lambda: (obj.pose.position, None))})
        attempts = []
        async def execute(name, **kwargs):
            # Model updates inside commands, independently of task submission.
            world.current_time += 1/60
            world.current_time_step_index += 1
            updater.update_count += 1
            if name == 'home':
                return SkillResult(SkillStatus.SUCCESS)
            if name == 'pick':
                attempts.append(name)
                if len(attempts) == 1 and failed == 'pick':
                    return SkillResult(SkillStatus.FAILED, 'primary pick failure', FailureCode.GRASP_FAILED,
                                       {'evidence': 'pick feedback'})
                model.set_held('cube')
                return SkillResult(SkillStatus.SUCCESS, details={'evidence': 'picked'})
            if len(attempts) == 1 and failed == 'place':
                return SkillResult(SkillStatus.FAILED, 'primary place failure', FailureCode.PLACE_FAILED,
                                   {'visible': False, 'retreat_ok': True})
            obj.pose = kwargs['target'].translated([placement_offset, 0, 0])
            model.set_held(None)
            return SkillResult(SkillStatus.SUCCESS)
        bundle.runtime = NS(execute_async=AsyncMock(side_effect=execute))
        bundle.backend = NS(open_gripper_async=AsyncMock(side_effect=AssertionError('unexpected cleanup release')),
                            move_to_pose_async=AsyncMock(), close_gripper_async=AsyncMock(),
                            home_async=AsyncMock(), get_end_effector_pose=Mock(return_value=Pose([.45, 0, .1])))
        def bridge(coroutine, app):
            # Scheduling another standalone Kit task allows an extra physics
            # tick. A direct await of the next skill must avoid that handoff.
            world.current_time += 1/60
            world.current_time_step_index += 1
            updater.update_count += 1
            return asyncio.run(coroutine)
        def set_pose(cube, position, world, **kwargs):
            obj.pose = Pose(position)
            obj.last_seen = time.monotonic()
            return obj.pose.position.copy()
        module.run_kit_coroutine = bridge
        module.open_playground = Mock()
        module.build_perception_runtime = Mock(return_value=bundle)
        module.settle = Mock()
        module.choose_object = Mock(return_value=obj)
        module.set_cube_pose = set_pose
        module.object_snapshot = lambda item: {'position': item.pose.position.tolist(), 'visible': item.visible}
        return module, bundle

    def test_pick_place_handoff_does_not_submit_a_second_kit_task(self):
        module, bundle = self.benchmark(failed=None)
        bridge = Mock(wraps=module.run_kit_coroutine)
        module.run_kit_coroutine = bridge
        execute = bundle.runtime.execute_async.side_effect
        handoffs = []
        async def observe_handoff(name, **kwargs):
            if name == 'place':
                handoffs[-1].append(bundle.world.current_time_step_index)
            result = await execute(name, **kwargs)
            if name == 'pick':
                handoffs.append([bundle.world.current_time_step_index])
            return result
        bundle.runtime.execute_async.side_effect = observe_handoff
        module.main(self.result, self.out)
        self.assertEqual(bridge.call_count, 8)  # Home and one Pick/Place task per trial.
        self.assertEqual(len(handoffs), 4)
        self.assertTrue(all(finished == started for finished, started in handoffs))
        calls = bundle.runtime.execute_async.await_args_list
        self.assertEqual([call.args[0] for call in calls], ['home', 'pick', 'place']*4)
        for index, (_, target) in enumerate(module.MANIPULATION_TRIALS):
            self.assertEqual(calls[index*3+1].kwargs, {'object_id': 'cube'})
            self.assertEqual(calls[index*3+2].kwargs['mode'], 'stable')
            np.testing.assert_array_equal(calls[index*3+2].kwargs['target'].position, target)

    def test_command_trace_preserves_dispatch_and_records_physics_and_geometry(self):
        module, bundle = self.benchmark()
        target = Pose([.48, -.08, .025], [1, 0, 0, 0])
        result = BackendResult(False, 'orientation timeout', timed_out=True, details={'steps': 1000})
        async def move(*args, **kwargs):
            bundle.world.current_time += 1000/60
            bundle.world.current_time_step_index += 1000
            bundle.backend.get_end_effector_pose.return_value = target
            return result
        original = bundle.backend.move_to_pose_async
        original.side_effect = move
        records = []
        with module.trace_backend_commands(bundle, records):
            actual = asyncio.run(bundle.backend.move_to_pose_async(target, speed=.15, orientation_tolerance=.1))
        self.assertIs(actual, result)
        self.assertIs(bundle.backend.move_to_pose_async, original)
        original.assert_awaited_once_with(target, speed=.15, orientation_tolerance=.1)
        row = records[0]
        self.assertEqual(row['target']['position'], target.position.tolist())
        self.assertEqual(row['after']['physics_step']-row['before']['physics_step'], 1000)
        self.assertEqual(row['result']['details']['steps'], 1000)
        self.assertEqual(row['after']['tcp_position'], target.position.tolist())
        self.assertEqual(row['after']['cube_position'], [.45, 0, .025])
        self.assertTrue(row['result']['timed_out'])
        bundle.backend.open_gripper_async.assert_not_awaited()
        json.dumps(records, allow_nan=False)

    def test_command_trace_preserves_exception_or_cancellation_even_if_diagnostic_read_fails(self):
        for failure in (ValueError('primary command error'), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                module, bundle = self.benchmark()
                original = bundle.backend.close_gripper_async
                original.side_effect = failure
                bundle.backend.get_end_effector_pose.side_effect = RuntimeError('query failed')
                records = []
                with module.trace_backend_commands(bundle, records):
                    with self.assertRaises(type(failure)) as raised:
                        asyncio.run(bundle.backend.close_gripper_async(width=.04))
                self.assertIs(raised.exception, failure)
                self.assertIs(bundle.backend.close_gripper_async, original)
                original.assert_awaited_once_with(width=.04)
                self.assertIn('query failed', records[0]['after']['diagnostic_error'])
                self.assertIn(type(failure).__name__, records[0]['exception'])

    def test_failed_trial_persists_command_trace_and_restores_backend(self):
        module, bundle = self.benchmark()
        execute = bundle.runtime.execute_async.side_effect
        bundle.backend.close_gripper_async.return_value = BackendResult(True, 'closed')
        async def execute_with_command(name, **kwargs):
            if name == 'pick':
                await bundle.backend.close_gripper_async()
            return await execute(name, **kwargs)
        bundle.runtime.execute_async.side_effect = execute_with_command
        with self.assertRaisesRegex(AssertionError, 'benchmark stopped without cleanup commands'):
            module.main(self.result, self.out)
        rows = json.loads(self.result.read_text())['metrics']['manipulation_trials']
        for row in rows:
            self.assertEqual([r['command'] for r in row['backend_commands']], ['close_gripper_async'])
            self.assertTrue(row['backend_commands'][0]['result']['ok'])
        bundle.backend.open_gripper_async.assert_not_awaited()
        bundle.backend.close_gripper_async.assert_awaited()

    def test_setup_failure_preserves_prior_trials_and_current_physical_state(self):
        module, bundle = self.benchmark(failed=None)
        execute = bundle.runtime.execute_async.side_effect
        homes = 0
        async def execute_with_visibility_loss(name, **kwargs):
            nonlocal homes
            if name == 'home':
                homes += 1
                if homes == 4:
                    bundle.world_model.require('cube').visible = False
            return await execute(name, **kwargs)
        bundle.runtime.execute_async.side_effect = execute_with_visibility_loss
        with self.assertRaisesRegex(RuntimeError, 'not visible.*trial 4'):
            module.main(self.result, self.out)
        metrics = json.loads(self.result.read_text())['metrics']
        self.assertEqual(len(metrics['manipulation_trials']), 3)
        self.assertEqual(metrics['failed_setup']['trial'], 4)
        self.assertFalse(metrics['failed_setup']['world_state']['object']['visible'])
        self.assertEqual(metrics['failed_setup']['physical_state']['cube_position'], [.46, .02, .025])

    def test_failed_trial_stops_without_release_or_next_trial_and_retains_primary_failure(self):
        module, bundle = self.benchmark()
        with self.assertRaisesRegex(AssertionError, 'benchmark stopped without cleanup commands'):
            module.main(self.result, self.out)
        report = json.loads(self.result.read_text())
        row = report['metrics']['manipulation_trials'][0]
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(row['pick_status'], 'success')
        self.assertEqual(row['place_status'], 'failed')
        self.assertEqual(row['place_failure_code'], 'place_failed')
        self.assertEqual(row['failure_handling'], 'stopped_without_cleanup_commands')
        self.assertGreater(row['after_place']['world_updates'], row['before_pick']['world_updates'])
        self.assertIn('observation_age_s', row['after_pick'])
        bundle.backend.open_gripper_async.assert_not_awaited()
        self.assertEqual(bundle.world_model.held_object_id, 'cube')
        self.assertEqual([call.args[0] for call in bundle.runtime.execute_async.await_args_list],
                         ['home', 'pick', 'place'])
        self.assertEqual(report['metrics']['manipulation_completed'], 1)
        self.assertEqual(report['metrics']['manipulation_total'], 4)
        self.assertIn('primary place failure', report['error'])
        self.assertNotIn('control_samples', report['metrics'])

    def test_loaded_trace_is_opt_in_and_retained_on_failure_or_operator_interrupt(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                module, bundle = self.benchmark()
                contexts_closed = []
                @contextmanager
                def recorder(backend, scheduler, samples, cube):
                    self.assertIs(backend, bundle.backend)
                    self.assertIs(cube, bundle.objects['cube'])
                    try:
                        yield lambda stage: samples.append({'stage': stage})
                    finally:
                        contexts_closed.append(True)
                module.record_control = recorder
                module.PhysicsCallbackScope = Mock()
                if interrupted:
                    execute = bundle.runtime.execute_async.side_effect
                    async def stop(name, **kwargs):
                        if name == 'pick':
                            raise KeyboardInterrupt('operator stop')
                        return await execute(name, **kwargs)
                    bundle.runtime.execute_async.side_effect = stop
                with self.assertRaises(KeyboardInterrupt if interrupted else AssertionError):
                    module.main(self.result, self.out, control_trace=True)
                report = json.loads(self.result.read_text())
                self.assertEqual(report['status'], 'FAIL')
                self.assertEqual(report['metrics']['control_samples'][0]['stage'], 'trial_1:pick')
                self.assertEqual(len(contexts_closed), 1)
                bundle.backend.open_gripper_async.assert_not_called()
                if interrupted:
                    self.assertIn('KeyboardInterrupt: operator stop', report['error'])
                else:
                    self.assertIn('primary place failure', report['error'])

    def test_timeout_with_held_object_stops_without_changing_failure_or_held_state(self):
        module, bundle = self.benchmark(failed=None)
        execute = bundle.runtime.execute_async.side_effect
        async def timeout(name, **kwargs):
            if name == 'place':
                return SkillResult(SkillStatus.TIMEOUT, 'Pre-place motion failed.', FailureCode.TIMEOUT,
                                   {'orientation_error_rad': .1137, 'max_steps': 1000})
            return await execute(name, **kwargs)
        bundle.runtime.execute_async.side_effect = timeout
        with self.assertRaisesRegex(AssertionError, 'timeout.*Pre-place motion failed'):
            module.main(self.result, self.out)
        report = json.loads(self.result.read_text())
        row = report['metrics']['manipulation_trials'][0]
        self.assertEqual(row['place_status'], 'timeout')
        self.assertEqual(row['place_details']['max_steps'], 1000)
        self.assertEqual(row['after_place']['held_object_id'], 'cube')
        self.assertEqual(bundle.world_model.held_object_id, 'cube')
        bundle.backend.open_gripper_async.assert_not_called()
        self.assertEqual(bundle.runtime.execute_async.await_count, 3)

    def test_failed_pick_records_unrun_place_without_cleanup(self):
        module, bundle = self.benchmark(failed='pick')
        with self.assertRaisesRegex(AssertionError, 'benchmark stopped without cleanup commands'):
            module.main(self.result, self.out)
        row = json.loads(self.result.read_text())['metrics']['manipulation_trials'][0]
        self.assertEqual(row['pick_status'], 'failed')
        self.assertEqual(row['pick_failure_code'], 'grasp_failed')
        self.assertEqual(row['pick_details'], {'evidence': 'pick feedback'})
        self.assertIsNone(row['place_status'])
        self.assertIsNone(row['after_place'])
        bundle.backend.open_gripper_async.assert_not_awaited()

    def test_successful_skill_results_do_not_relax_true_placement_accuracy(self):
        module, _ = self.benchmark(failed=None, placement_offset=.03)
        with self.assertRaisesRegex(AssertionError, '25 mm'):
            module.main(self.result, self.out)
        report = json.loads(self.result.read_text())
        self.assertEqual(report['metrics']['manipulation_successes'], 4)
        self.assertGreater(report['metrics']['target_error_mm']['max'], 25.)
        self.assertEqual(report['status'], 'FAIL')

    def placement(self, unexpected_command=None, change_held=False, wrong_code=False):
        module = scenario('test_placement_feasibility')
        model = WorldModel()
        held = WorldObject('cube', 'cube', Pose([.45, 0, .025]), [.05]*3, visible=True)
        obstacle = WorldObject('obstacle', 'cube', Pose(module.OBSTACLE_POS), [.05]*3, visible=True)
        model.register(held); model.register(obstacle)
        backend = FakeBackend()
        async def release():
            held.pose = Pose(module.SAFE_TARGET)
            return BackendResult(True)
        backend.open_gripper_async = release
        place = PlaceSkill(backend, model)
        async def execute(name, **kwargs):
            if name == 'pick':
                model.set_held('cube')
                return SkillResult(SkillStatus.SUCCESS)
            if np.array_equal(kwargs['target'].position, obstacle.pose.position):
                if unexpected_command:
                    if unexpected_command == 'move_to_pose_async':
                        await backend.move_to_pose_async(Pose([.4, 0, .2]))
                    else:
                        await getattr(backend, unexpected_command)()
                    return SkillResult(SkillStatus.FAILED, failure_code=FailureCode.TARGET_OCCUPIED)
                if change_held:
                    model.set_held(None)
                    return SkillResult(SkillStatus.FAILED, failure_code=FailureCode.TARGET_OCCUPIED)
                if wrong_code:
                    return SkillResult(SkillStatus.FAILED, failure_code=FailureCode.PLACE_FAILED)
            return await place.execute_async(**kwargs)
        bundle = NS(backend=backend, world_model=model, world=object(), runtime=NS(execute_async=execute),
                    objects={'cube': NS(get_world_pose=lambda: (held.pose.position, None))})
        def bridge(coroutine, app):
            result = asyncio.run(coroutine)
            backend.ee = backend.ee.translated([.001, 0, 0])  # Residual dynamics during app updates.
            return result
        module.run_kit_coroutine = bridge
        module.open_playground = module.spawn_obstacle = module.settle = Mock()
        module.build_ground_truth_runtime = Mock(return_value=bundle)
        module.nearest_object = Mock(return_value=obstacle)
        return module

    def test_command_free_rejection_passes_despite_residual_physical_drift(self):
        module = self.placement()
        module.main(self.result, self.out)
        report = json.loads(self.result.read_text())
        self.assertEqual(report['status'], 'PASS')
        metrics = report['metrics']
        self.assertAlmostEqual(metrics['ee_motion_during_rejection_mm'], 1.)
        self.assertFalse(any(metrics['rejection_backend_command_counts'].values()))
        self.assertEqual(metrics['occupied_failure_code'], 'target_occupied')
        self.assertEqual(metrics['held_after_rejection'], 'cube')

    def test_any_backend_command_fails_rejection_even_with_expected_failure_code(self):
        for command in ('move_to_pose_async', 'open_gripper_async', 'close_gripper_async', 'home_async'):
            with self.subTest(command=command):
                module = self.placement(unexpected_command=command)
                with self.assertRaisesRegex(AssertionError, 'Backend commands issued'):
                    module.main(self.result, self.out)
                report = json.loads(self.result.read_text())
                self.assertEqual(report['status'], 'FAIL')
                self.assertEqual(report['metrics']['rejection_backend_command_counts'][command], 1)

    def test_rejection_still_requires_correct_code_and_held_state(self):
        for option, message in (('change_held', 'Held-object state'), ('wrong_code', 'Expected TARGET_OCCUPIED')):
            with self.subTest(option=option):
                module = self.placement(**{option: True})
                with self.assertRaisesRegex(AssertionError, message):
                    module.main(self.result, self.out)
                self.assertEqual(json.loads(self.result.read_text())['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
