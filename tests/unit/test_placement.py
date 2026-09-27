from __future__ import annotations

import argparse
import sys

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.backend import BackendResult, ManipulationBackend
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
    def __init__(self):
        self.ee = Pose([0.45, 0.0, 0.20], [0.0, 0.0, 1.0, 0.0])
        self.move_calls = 0
        self.open_calls = 0

    def move_to_pose(self, target, speed=0.5, position_tolerance=None, orientation_tolerance=None):
        self.move_calls += 1
        self.ee = target
        return BackendResult(True)

    def open_gripper(self):
        self.open_calls += 1
        return BackendResult(True)

    def close_gripper(self, width=None):
        return BackendResult(True)

    def verify_grasp(self):
        return BackendResult(True)

    def get_end_effector_pose(self):
        return self.ee

    def check_reachability(self, target):
        return True

    def home(self, name="home"):
        return BackendResult(True)


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
    result = skill.execute(Pose(obstacle.pose.position.copy()))
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
