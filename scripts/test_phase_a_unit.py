from __future__ import annotations

import argparse
import sys

import numpy as np

from core.manipulation import BackendResult, ManipulationBackend
from core.skill import FailureCode
from core.types import Pose
from manipulation.grasp_planner import TopDownGraspPlanner
from scripts.perception_v1_common import fail_result, write_result
from skills.pick import PickSkill
from world_model.entities import WorldObject
from world_model.world_model import WorldModel


class FakeBackend(ManipulationBackend):
    def __init__(self, obj, *, move_every_time=False):
        self.obj = obj
        self.move_every_time = move_every_time
        self.move_calls = 0
        self.ee = Pose(
            [0.4, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.0],
        )

    def move_to_pose(
        self,
        target,
        speed=0.5,
        position_tolerance=None,
        orientation_tolerance=None,
    ):
        self.move_calls += 1
        self.ee = target
        if self.move_every_time or self.move_calls == 1:
            self.obj.pose = Pose(
                self.obj.pose.position + np.array([0.0, 0.04, 0.0]),
                self.obj.pose.orientation,
            )
            self.obj.visible = True
        return BackendResult(True)

    def open_gripper(self):
        return BackendResult(True)

    def close_gripper(self, width=None):
        return BackendResult(True)

    def verify_grasp(self):
        return BackendResult(
            True,
            details={"estimated_width_m": 0.05},
        )

    def get_end_effector_pose(self):
        return self.ee

    def check_reachability(self, target):
        return True

    def home(self, name="home"):
        return BackendResult(True)


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
        TopDownGraspPlanner(
            approach_height=0.10,
            default_lift_height=0.12,
            grasp_z_offset=0.0,
        ),
        movement_threshold=0.02,
        max_local_replans=2,
    )
    result = skill.execute("object_1")
    assert result.ok
    assert result.details["local_replans"] == 1
    assert model.held_object_id == "object_1"
    metrics["single_move_replanned"] = True

    model, obj = make_world()
    backend = FakeBackend(obj, move_every_time=True)
    skill = PickSkill(
        backend,
        model,
        TopDownGraspPlanner(
            approach_height=0.10,
            default_lift_height=0.12,
            grasp_z_offset=0.0,
        ),
        movement_threshold=0.02,
        max_local_replans=1,
    )
    result = skill.execute("object_1")
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
