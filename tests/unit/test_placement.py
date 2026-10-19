from __future__ import annotations

import asyncio

import argparse
import sys

import numpy as np

from robot_skill_stack.common.types import Pose, GripperState
from robot_skill_stack.manipulation.backend import BackendResult, GripperFeedback, ManipulationBackend
from robot_skill_stack.manipulation.placement import (
    PlacementFailureReason,
    SimplePlacementPlanner,
)
from robot_skill_stack.manipulation.skills.place import PlaceSkill
from robot_skill_stack.runtime.skill import FailureCode
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.world_model import WorldModel
from tests.support.results import fail_result, write_result


class FakeBackend(ManipulationBackend):
    is_ready = True
    grasp_min_width, grasp_max_width = .005, .075

    def __init__(self):
        self.ee = Pose([0.45, 0.0, 0.20], [0.0, 0.0, 1.0, 0.0])
        self.move_calls = 0
        self.open_calls = 0
        self.closed = True

    async def move_to_pose_async(self, target, speed=0.5, position_tolerance=None, orientation_tolerance=None,
                                 grasp_check=None):
        self.move_calls += 1
        self.ee = target
        if grasp_check:
            for _ in range(3):
                failure = grasp_check()
                if failure is not None:
                    return failure
        return BackendResult(True)

    async def open_gripper_async(self):
        self.open_calls += 1
        self.closed = False
        return BackendResult(True)

    async def close_gripper_async(self, width=None):
        self.closed = True
        return BackendResult(True)

    def gripper_feedback(self):
        return GripperFeedback(GripperState.HELD if self.closed else GripperState.FREE,
                               .05 if self.closed else 0.)

    def get_end_effector_pose(self):
        return self.ee

    def check_reachability(self, target):
        return True

    async def home_async(self, name="home", *, grasp_check=None):
        return await self.move_to_pose_async(self.ee, grasp_check=grasp_check)


def make_model(obstacle_visible=True):
    model = WorldModel()
    held = WorldObject(
        "object_1",
        "cube",
        Pose([0.45, 0.0, 0.025]),
        np.array([0.05, 0.05, 0.05]),
        visible=True,
    )
    obstacle = WorldObject(
        "object_2",
        "cube",
        Pose([0.45, 0.15, 0.025]),
        np.array([0.05, 0.05, 0.05]),
        visible=obstacle_visible,
    )
    model.register(held)
    model.register(obstacle)
    model.set_held(held.object_id)
    return model, held, obstacle


def main(result_path=None):
    metrics = {}
    planner = SimplePlacementPlanner(clearance=0.005)
    model, held, obstacle = make_model()

    result = planner.evaluate(held, Pose(obstacle.pose.position.copy()), model)
    assert not result.ok
    assert result.failure_reason == PlacementFailureReason.TARGET_OCCUPIED
    assert result.details["blocking_object_id"] == obstacle.object_id
    metrics["occupied_target_rejected"] = True

    safe = planner.evaluate(held, Pose([0.45, 0.25, 0.025]), model)
    assert safe.ok
    assert safe.details["feasibility_checked"]
    metrics["free_target_accepted"] = True

    model, held, obstacle = make_model(obstacle_visible=False)
    assert planner.evaluate(held, Pose(obstacle.pose.position.copy()), model).ok
    metrics["invisible_object_not_treated_as_hard_obstacle"] = True

    model, held, obstacle = make_model()
    backend = FakeBackend()
    skill = PlaceSkill(backend, model, planner)
    result = asyncio.run(skill.execute_async(Pose(obstacle.pose.position.copy())))
    assert not result.ok
    assert result.failure_code == FailureCode.TARGET_OCCUPIED
    assert backend.move_calls == 0
    assert backend.open_calls == 0
    assert model.held_object_id == held.object_id
    metrics["place_rejects_before_motion"] = True

    print("=== PLACEMENT UNIT ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("PASS")
    print("======================")
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
