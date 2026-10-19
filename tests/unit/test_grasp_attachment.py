from __future__ import annotations

import asyncio
import unittest
import numpy as np

from robot_skill_stack.common.rotations import quat_matrix
from robot_skill_stack.common.types import Pose, GripperState
from robot_skill_stack.manipulation.backend import BackendResult, GripperFeedback, ManipulationBackend
from robot_skill_stack.manipulation.grasping import GraspPlan, GraspResult, SimplePrimitiveGraspPlanner
from robot_skill_stack.manipulation.grasping.primitive import _top_q
from robot_skill_stack.manipulation.skills.pick import PickSkill
from robot_skill_stack.manipulation.skills.place import PlaceSkill
from robot_skill_stack.runtime.registry import SkillRegistry
from robot_skill_stack.runtime.runtime import RobotRuntime
from robot_skill_stack.runtime.skill import FailureCode
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.world_model import WorldModel
from tests.unit.test_grasp_clearance import solid


class GeometryBackend(ManipulationBackend):
    is_ready = True
    grasp_min_width, grasp_max_width = .005, .075

    """Kinematic test double, not a physics simulation."""
    def __init__(self, obj):
        self.obj, self.ee = obj, Pose([.45, 0, .30], _top_q(0))
        self.closed, self.offset = False, None
        self.calls, self.bias, self.fail_release = [], 0., False

    def _move(self, target):
        actual = target.translated([0, 0, self.bias])
        if self.closed:
            self.obj.pose = Pose(actual.position+quat_matrix(actual.orientation) @ self.offset)
        elif np.linalg.norm(actual.position[:2]-self.obj.pose.position[:2]) < .001:
            top = self.obj.pose.position[2]+self.obj.size[2]/2
            if actual.position[2]+.030 < top-1e-9:
                raise AssertionError("TCP target would drive the palm through the object")
        self.ee = actual
        self.calls.append(("move", target))
        return BackendResult(True)

    def _open(self):
        self.calls.append(("open", None))
        if self.closed and self.fail_release:
            return BackendResult(False)
        self.closed = False
        return BackendResult(True)

    def _close(self):
        self.calls.append(("close", None))
        self.offset = quat_matrix(self.ee.orientation).T @ (self.obj.pose.position-self.ee.position)
        self.closed = True
        return BackendResult(True)

    async def move_to_pose_async(self, target, **kwargs):
        result = self._move(target)
        if kwargs.get('grasp_check'):
            for _ in range(3):
                failure = kwargs['grasp_check']()
                if failure is not None:
                    return failure
        return result
    async def open_gripper_async(self): return self._open()
    async def close_gripper_async(self, width=None): return self._close()
    def gripper_feedback(self):
        return GripperFeedback(GripperState.HELD if self.closed else GripperState.FREE,
                               .05 if self.closed else 0.)
    def get_end_effector_pose(self): return self.ee
    def check_reachability(self, target): return True
    async def home_async(self, name="home", *, grasp_check=None):
        return await self.move_to_pose_async(self.ee, grasp_check=grasp_check)


class AlternatePlanner:
    def plan(self, obj, *, lift_height=None, grasp_hint=None):
        grasp = Pose(obj.pose.position+[0, 0, .03], _top_q(.38))
        return GraspResult.success(GraspPlan(grasp.translated([0, 0, .1]), grasp,
                                           grasp.translated([0, 0, .12])), strategy="alternate")


class AttachmentTests(unittest.IsolatedAsyncioTestCase):
    def stack(self, obj=None, planner=None):
        obj = solid() if obj is None else obj
        wm = WorldModel(); wm.register(obj)
        backend = GeometryBackend(obj)
        grasp = planner or SimplePrimitiveGraspPlanner(support_plane_z=0., world_model=wm,
                                                       current_pose=backend.get_end_effector_pose)
        registry = SkillRegistry()
        registry.register(PickSkill(backend, wm, grasp))
        registry.register(PlaceSkill(backend, wm))
        return wm, backend, RobotRuntime(registry)

    async def pair(self, obj):
        target = Pose(obj.pose.position+[0, .25, 0])
        wm, backend, runtime = self.stack(obj)
        picked = await runtime.execute_async("pick", object_id=obj.object_id)
        self.assertTrue(picked.ok, picked)
        self.assertIsNotNone(wm.held_object_offset_in_ee)
        placed = await runtime.execute_async("place", target=target)
        self.assertTrue(placed.ok, placed)
        np.testing.assert_allclose(obj.pose.position, target.position, atol=1e-9)
        self.assertIsNone(wm.held_object_id)
        self.assertIsNone(wm.held_object_offset_in_ee)
        self.assertEqual(wm.association_hints()[0].object_id, obj.object_id)
        np.testing.assert_allclose(wm.association_hints()[0].expected_position, target.position)

    async def test_pick_place_keeps_object_center_target_for_all_shapes(self):
        for obj in (solid("cube"), solid("sphere", radius=.025), solid(), solid(horizontal=True, yaw=.4)):
            with self.subTest(shape=obj.geometry): await self.pair(obj)

    async def test_invalid_geometry_is_rejected_before_any_motion_or_open(self):
        for obj in (solid(length=.25), solid(radius=.05), solid("cube", size=(.04, .04, .01))):
            with self.subTest(geometry=obj.geometry):
                wm, backend, runtime = self.stack(obj)
                result = await runtime.execute_async("pick", object_id=obj.object_id)
                self.assertEqual(result.failure_code, FailureCode.NO_VALID_GRASP)
                self.assertFalse(backend.calls)
                self.assertIsNone(wm.held_object_id)

    async def test_obstacle_failure_does_not_start_descent(self):
        wm, backend, runtime = self.stack()
        wm.register(WorldObject("shelf", pose=Pose([.45, 0, .18]), size=[.2, .2, .01], visible=True))
        result = await runtime.execute_async("pick", object_id="target")
        self.assertEqual(result.failure_code, FailureCode.NO_VALID_GRASP)
        self.assertEqual(result.details["grasp_failure_reason"], "approach_blocked")
        self.assertFalse(backend.calls)

    async def test_actual_tcp_not_only_commanded_pose_sets_attachment(self):
        wm, backend, runtime = self.stack()
        backend.bias = .004
        result = await runtime.execute_async("pick", object_id="target")
        self.assertTrue(result.ok, result)
        np.testing.assert_allclose(wm.held_object_offset_in_ee, backend.offset, atol=1e-9)
        self.assertAlmostEqual(wm.held_object_offset_in_ee[2], .029)

    async def test_arbitrary_planner_is_still_replaceable_and_offset_generic(self):
        wm, backend, runtime = self.stack(solid("cube"), planner=AlternatePlanner())
        result = await runtime.execute_async("pick", object_id="target")
        self.assertTrue(result.ok, result)
        self.assertEqual(result.details["grasp_planner_details"]["strategy"], "alternate")
        target = Pose([.45, .25, .025])
        result = await runtime.execute_async("place", target=target)
        self.assertTrue(result.ok, result)
        np.testing.assert_allclose(backend.obj.pose.position, target.position, atol=1e-9)

    async def test_explicit_wrist_rotation_rotates_offset_instead_of_adding_world_z(self):
        wm, backend, _ = self.stack()
        wm.set_held("target", object_offset_in_ee=[.01, .02, .03])
        target = Pose([.45, .25, .05], _top_q(np.pi/2))
        _, release, _ = PlaceSkill(backend, wm)._poses(target)
        np.testing.assert_allclose(release.position+quat_matrix(release.orientation) @ wm.held_object_offset_in_ee,
                                   target.position, atol=1e-9)

    async def test_legacy_manually_held_object_keeps_old_no_offset_behavior(self):
        wm, backend, _ = self.stack()
        wm.set_held("target")
        target = Pose([.45, .25, .05])
        _, release, _ = PlaceSkill(backend, wm)._poses(target)
        np.testing.assert_allclose(release.position, target.position)

    async def test_failed_release_retains_attachment_and_clears_hint(self):
        wm, backend, runtime = self.stack()
        self.assertTrue((await runtime.execute_async("pick", object_id="target")).ok)
        offset = wm.held_object_offset_in_ee.copy()
        backend.fail_release = True
        self.assertFalse((await runtime.execute_async("place", target=Pose([.45, .25, .05]))).ok)
        self.assertEqual(wm.held_object_id, "target")
        np.testing.assert_equal(wm.held_object_offset_in_ee, offset)
        self.assertFalse(wm.association_hints())

    async def test_attachment_lifecycle_validates_copies_and_clears(self):
        wm, _, _ = self.stack()
        original = np.array([0., 0., .025])
        wm.set_held("target", object_offset_in_ee=original)
        original[2] = 9
        self.assertAlmostEqual(wm.held_object_offset_in_ee[2], .025)
        for value in ([1, 2], [0, np.nan, 1]):
            with self.assertRaises(ValueError): wm.set_held("target", object_offset_in_ee=value)
        wm.set_held(None)
        self.assertIsNone(wm.held_object_offset_in_ee)
        wm.set_held("target")
        self.assertIsNone(wm.held_object_offset_in_ee)


if __name__ == "__main__":
    unittest.main(verbosity=2)
