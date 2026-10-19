"""Place/release differ in landing verification, not motion failure handling."""
import unittest
from unittest.mock import AsyncMock

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.backend import BackendResult
from robot_skill_stack.manipulation.skills.place import PlaceSkill
from robot_skill_stack.runtime.skill import FailureCode, SkillStatus
from tests.unit.test_placement import FakeBackend, make_model


class PlaceModeTests(unittest.IsolatedAsyncioTestCase):
    target = Pose([.45, .30, .15])

    def stack(self, landing=None, visible=True):
        model, obj, _ = make_model()
        backend = FakeBackend()
        backend.move_to_pose_async = AsyncMock(wraps=backend.move_to_pose_async)
        async def release():
            backend.open_calls += 1
            obj.pose = Pose(self.target.position if landing is None else landing)
            obj.visible = visible
            return BackendResult(True)
        backend.open_gripper_async = AsyncMock(side_effect=release)
        return model, backend, PlaceSkill(backend, model)

    async def test_intentional_midair_release_succeeds_without_landing_accuracy_or_visibility(self):
        model, backend, skill = self.stack(landing=[.1, -.2, .025], visible=False)
        result = await skill.execute_async(self.target, mode='release')
        self.assertTrue(result.ok, result)
        self.assertEqual(result.message, "Released 'object_1'.")
        np.testing.assert_array_equal(backend.move_to_pose_async.await_args_list[1].args[0].position,
                                      self.target.position)
        backend.open_gripper_async.assert_awaited_once_with()
        self.assertIsNone(model.held_object_id)

    async def test_stable_requires_observed_final_position_near_target(self):
        for landing, visible, succeeds in ((self.target.position, True, True),
                                           ([.1, -.2, .025], True, False),
                                           (self.target.position, False, False)):
            with self.subTest(landing=landing, visible=visible):
                model, backend, skill = self.stack(landing, visible)
                result = await skill.execute_async(self.target, mode='stable')
                self.assertEqual(result.ok, succeeds, result)
                backend.open_gripper_async.assert_awaited_once()
                self.assertIsNone(model.held_object_id)  # Released even when verification fails.
                if not succeeds:
                    self.assertEqual(result.failure_code, FailureCode.PLACE_FAILED)

    async def test_default_and_legacy_default_alias_keep_stable_semantics(self):
        for kwargs in ({}, {'mode': 'default'}):
            with self.subTest(kwargs=kwargs):
                _, _, skill = self.stack(landing=[.1, -.2, .025])
                result = await skill.execute_async(self.target, **kwargs)
                self.assertEqual(result.failure_code, FailureCode.PLACE_FAILED)

    async def test_failed_preapproach_or_descent_never_releases_in_either_mode(self):
        for mode in ('stable', 'release'):
            for failed_move in (1, 2):
                with self.subTest(mode=mode, failed_move=failed_move):
                    model, backend, skill = self.stack()
                    async def move(*args, **kwargs):
                        if backend.move_to_pose_async.await_count == failed_move:
                            return BackendResult(False, 'orientation did not converge', timed_out=True,
                                                 details={'orientation_error_rad': .1137})
                        return BackendResult(True)
                    backend.move_to_pose_async.side_effect = move
                    result = await skill.execute_async(self.target, mode=mode)
                    self.assertEqual(result.status, SkillStatus.TIMEOUT)
                    self.assertEqual(result.failure_code, FailureCode.TIMEOUT)
                    self.assertEqual(result.details['orientation_error_rad'], .1137)
                    self.assertEqual(backend.move_to_pose_async.await_count, failed_move)
                    backend.open_gripper_async.assert_not_called()
                    self.assertEqual(model.held_object_id, 'object_1')

    async def test_open_failure_remains_failure_in_both_modes(self):
        for mode in ('stable', 'release'):
            with self.subTest(mode=mode):
                model, backend, skill = self.stack()
                backend.open_gripper_async.side_effect = None
                backend.open_gripper_async.return_value = BackendResult(False, 'release failed')
                result = await skill.execute_async(self.target, mode=mode)
                self.assertEqual(result.failure_code, FailureCode.PLACE_FAILED)
                self.assertEqual(model.held_object_id, 'object_1')
                self.assertEqual(backend.move_to_pose_async.await_count, 2)

    async def test_occupied_target_rejected_before_commands_in_both_modes(self):
        for mode in ('stable', 'release'):
            with self.subTest(mode=mode):
                model, backend, skill = self.stack()
                result = await skill.execute_async(model.require('object_2').pose, mode=mode)
                self.assertEqual(result.failure_code, FailureCode.TARGET_OCCUPIED)
                backend.move_to_pose_async.assert_not_called()
                backend.open_gripper_async.assert_not_called()
                self.assertEqual(model.held_object_id, 'object_1')
