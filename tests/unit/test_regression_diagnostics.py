"""Exercise scenario assertions/reporting with doubles; never start native Isaac."""
import ast
import asyncio
from contextlib import ExitStack
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


def scenario(name):
    # Load the actual scenario functions/constants without its native entry-point
    # imports or SimulationApp construction. Dependencies below are explicit doubles.
    path = ROOT/'tests/regression'/f'{name}.py'
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) or
             (isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets))]
    module = ModuleType(name)
    module.__dict__.update(np=np, Path=Path, Pose=Pose, FailureCode=FailureCode, time=time,
                           ExitStack=ExitStack, patch=patch, write_result=write_result, simulation_app=object())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


class RegressionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.result = self.out/'result.json'

    def benchmark(self, failed='place', cleanup='ok', placement_offset=0.):
        module = scenario('test_benchmark')
        model = WorldModel()
        obj = WorldObject('cube', 'cube', Pose([.45, 0, .025]), [.05]*3, visible=True)
        model.register(obj)
        world, updater = NS(current_time=0.), NS(update_count=0, last_error=None)
        bundle = NS(world=world, world_model=model, world_updater=updater,
                    objects={'cube': NS(get_world_pose=lambda: (obj.pose.position, None))})
        attempts = []
        async def execute(name, **kwargs):
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
        async def release():
            checkpoint = json.loads(self.result.read_text())
            row = checkpoint['metrics']['manipulation_trials'][0]
            self.assertEqual(checkpoint['status'], 'FAIL')
            self.assertEqual(row['place_failure_code'], 'place_failed')
            self.assertEqual(row['place_details'], {'visible': False, 'retreat_ok': True})
            if cleanup == 'exception':
                raise ValueError('release controller error')
            return BackendResult(cleanup == 'ok', 'release result', details={'feedback': cleanup})
        bundle.runtime = NS(execute_async=AsyncMock(side_effect=execute))
        bundle.backend = NS(open_gripper_async=AsyncMock(side_effect=release))
        def bridge(coroutine, app):
            world.current_time += 1/60
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

    def test_failed_trial_is_saved_before_async_cleanup_and_retained_in_final_failure(self):
        module, bundle = self.benchmark()
        with self.assertRaisesRegex(AssertionError, 'Only 3/4'):
            module.main(self.result, self.out)
        report = json.loads(self.result.read_text())
        row = report['metrics']['manipulation_trials'][0]
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(row['pick_status'], 'success')
        self.assertEqual(row['place_status'], 'failed')
        self.assertEqual(row['place_failure_code'], 'place_failed')
        self.assertTrue(row['cleanup']['ok'])
        self.assertGreater(row['after_place']['world_updates'], row['before_pick']['world_updates'])
        self.assertIn('observation_age_s', row['after_pick'])
        bundle.backend.open_gripper_async.assert_awaited_once()

    def test_cleanup_failure_never_replaces_primary_trial_failure(self):
        for cleanup in ('failure', 'exception'):
            with self.subTest(cleanup=cleanup):
                module, bundle = self.benchmark(cleanup=cleanup)
                with self.assertRaisesRegex(AssertionError, 'primary place failure.*cleanup also failed'):
                    module.main(self.result, self.out)
                report = json.loads(self.result.read_text())
                self.assertIn('primary place failure', report['error'])
                rows = report['metrics']['manipulation_trials']
                self.assertEqual(len(rows), 1)
                self.assertFalse(rows[0]['cleanup']['ok'])
                self.assertEqual(rows[0]['place_details'], {'visible': False, 'retreat_ok': True})
                self.assertEqual(bundle.world_model.held_object_id, 'cube')

    def test_failed_pick_records_unrun_place_without_cleanup(self):
        module, bundle = self.benchmark(failed='pick')
        with self.assertRaisesRegex(AssertionError, 'Only 3/4'):
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
