"""Intentional contact is explicit until reactive manipulation geometry is modeled."""
import asyncio
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.backend import BackendResult
from robot_skill_stack.manipulation.held_state import motion_failure_code
from robot_skill_stack.manipulation.skills.pick import PickSkill
from robot_skill_stack.manipulation.skills.place import PlaceSkill
from robot_skill_stack.runtime.skill import FailureCode
from robot_skill_stack.world.model.world_model import WorldModel


class ReactiveContactPolicyTests(unittest.TestCase):
    def test_pick_and_place_reject_without_motion_planning_or_gripper_commands(self):
        async def run():
            backend = NS(supports_contact_motion=False,
                move_to_pose_async=Mock(side_effect=AssertionError('motion')),
                open_gripper_async=Mock(side_effect=AssertionError('open')),
                close_gripper_async=Mock(side_effect=AssertionError('close')))
            model, planner = WorldModel(), Mock(side_effect=AssertionError('planner'))
            pick = await PickSkill(backend, model, planner).execute_async('target')
            place = await PlaceSkill(backend, model).execute_async(Pose([.4, .2, .03]))
            for result in (pick, place):
                self.assertFalse(result.ok)
                self.assertEqual(result.failure_code, FailureCode.MOTION_UNSUPPORTED)
                self.assertIn('intended-contact', result.message)
            backend.move_to_pose_async.assert_not_called()
            backend.open_gripper_async.assert_not_called()
            backend.close_gripper_async.assert_not_called()
            self.assertIsNone(model.held_object_id)
        asyncio.run(run())

    def test_map_and_unsupported_failures_retain_specific_skill_codes(self):
        for details, expected in (({'world_state_unavailable': True}, FailureCode.WORLD_STATE_UNAVAILABLE),
                                  ({'unsupported_motion': True}, FailureCode.MOTION_UNSUPPORTED)):
            result = BackendResult(False, details=details)
            self.assertEqual(motion_failure_code(result, FailureCode.PATH_BLOCKED), expected)
        self.assertEqual(motion_failure_code(BackendResult(False), FailureCode.PATH_BLOCKED),
                         FailureCode.PATH_BLOCKED)
        self.assertEqual(motion_failure_code(BackendResult(False, timed_out=True), FailureCode.PATH_BLOCKED),
                         FailureCode.TIMEOUT)


if __name__ == '__main__':
    unittest.main()
