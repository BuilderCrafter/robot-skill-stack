from __future__ import annotations

import asyncio

import argparse
import sys

import numpy as np

from robot_skill_stack.manipulation.backend import BackendResult, GripperFeedback, ManipulationBackend
from robot_skill_stack.runtime.skill import FailureCode
from robot_skill_stack.common.types import Pose, GripperState
from robot_skill_stack.manipulation.grasping import SimplePrimitiveGraspPlanner
from tests.support.results import fail_result, write_result
from robot_skill_stack.manipulation.skills.pick import PickSkill
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.world_model import WorldModel


class FakeBackend(ManipulationBackend):
    is_ready = True
    grasp_min_width, grasp_max_width = .005, .075

    def __init__(self, obj, *, move_every_time=False):
        self.obj = obj
        self.move_every_time = move_every_time
        self.move_calls = 0
        self.closed = False
        self.ee = Pose(
            [0.4, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.0],
        )

    async def move_to_pose_async(
        self,
        target,
        position_tolerance=None,
        orientation_tolerance=None,
        grasp_check=None,
    ):
        self.move_calls += 1
        self.ee = target
        if self.move_every_time or self.move_calls == 1:
            self.obj.pose = Pose(
                self.obj.pose.position + np.array([0.0, 0.04, 0.0]),
                self.obj.pose.orientation,
            )
            self.obj.visible = True
        if grasp_check:
            for _ in range(3):
                failure = grasp_check()
                if failure is not None:
                    return failure
        return BackendResult(True)

    async def open_gripper_async(self):
        self.closed = False
        return BackendResult(True)

    async def close_gripper_async(self, width=None):
        self.closed = True
        return BackendResult(True)

    def gripper_feedback(self):
        return GripperFeedback(
            GripperState.HELD if self.closed else GripperState.FREE, .05 if self.closed else 0.,
            details={"estimated_width_m": 0.05},
        )

    def get_end_effector_pose(self):
        return self.ee

    def check_reachability(self, target):
        return True

    async def home_async(self, name="home", *, grasp_check=None):
        return await self.move_to_pose_async(self.ee, grasp_check=grasp_check)


def make_world():
    model = WorldModel()
    obj = WorldObject(
        object_id="object_1",
        class_name="cube",
        pose=Pose([0.4, 0.0, 0.025]),
        size=np.array([0.05, 0.05, 0.05]),
        visible=True,
    )
    model.register(obj)
    return model, obj


def main(result_path=None):
    metrics = {}

    model, obj = make_world()
    backend = FakeBackend(obj)
    skill = PickSkill(
        backend,
        model,
        SimplePrimitiveGraspPlanner(
            approach_height=0.10,
            default_lift_height=0.12,
            grasp_z_offset=0.0,
        ),
        movement_threshold=0.02,
        max_local_replans=2,
    )
    result = asyncio.run(skill.execute_async("object_1"))
    assert result.ok
    assert result.details["local_replans"] == 1
    assert model.held_object_id == "object_1"
    metrics["single_move_replanned"] = True

    model, obj = make_world()
    backend = FakeBackend(obj, move_every_time=True)
    skill = PickSkill(
        backend,
        model,
        SimplePrimitiveGraspPlanner(
            approach_height=0.10,
            default_lift_height=0.12,
            grasp_z_offset=0.0,
        ),
        movement_threshold=0.02,
        max_local_replans=1,
    )
    result = asyncio.run(skill.execute_async("object_1"))
    assert not result.ok
    assert result.failure_code == FailureCode.OBJECT_MOVED
    assert result.details["local_replans"] == 1
    metrics["persistent_motion_bounded"] = True

    print("=== PHASE A UNIT ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("PASS")
    print("====================")
    write_result(result_path, status="PASS", metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result")
    args = parser.parse_args()
    try:
        main(args.result)
    except Exception as exc:
        fail_result(args.result, exc)
        sys.exit(1)
