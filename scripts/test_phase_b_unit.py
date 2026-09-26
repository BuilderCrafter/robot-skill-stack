from __future__ import annotations

import argparse
import sys

import numpy as np

from core.manipulation import BackendResult, ManipulationBackend
from core.skill import FailureCode
from core.types import Pose
from manipulation.grasping import (
    GraspFailureReason,
    GraspPlan,
    GraspResult,
    ParallelJawGripperSpec,
    TopDownGraspPlanner,
)
from scripts.perception_v1_common import fail_result, write_result
from skills.pick import PickSkill
from world_model.entities import WorldObject
from world_model.world_model import WorldModel


def world_object(
    *,
    size=(0.05, 0.05, 0.05),
    pose=True,
    graspable=True,
):
    return WorldObject(
        object_id="object_1",
        class_name="cube",
        pose=(
            Pose([0.45, 0.0, 0.025])
            if pose
            else None
        ),
        size=(
            None
            if size is None
            else np.asarray(size, dtype=float)
        ),
        graspable=graspable,
        visible=True,
    )


class FakeBackend(ManipulationBackend):
    def __init__(self):
        self.ee = Pose(
            [0.45, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.0],
        )

    def move_to_pose(
        self,
        target,
        speed=0.5,
        position_tolerance=None,
        orientation_tolerance=None,
    ):
        self.ee = target
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


class AlternatePlanner:
    def plan(
        self,
        obj,
        *,
        lift_height=None,
        grasp_hint=None,
    ):
        p = obj.pose.position.copy()
        grasp = Pose(
            p,
            [0.0, 0.0, 1.0, 0.0],
        )
        return GraspResult.success(
            GraspPlan(
                pre_grasp=grasp.translated([0, 0, 0.08]),
                grasp=grasp,
                lift=grasp.translated([0, 0, 0.10]),
            ),
            strategy="alternate_test_planner",
        )


def main(result_path=None):
    metrics = {}
    planner = TopDownGraspPlanner(
        gripper=ParallelJawGripperSpec(max_width=0.075),
    )

    result = planner.plan(world_object())
    assert result.ok
    assert result.plan is not None
    assert result.details["feasibility_checked"]
    assert abs(result.details["required_width_m"] - 0.05) < 1e-9
    metrics["small_cube_feasible"] = True

    result = planner.plan(
        world_object(size=(0.09, 0.09, 0.05))
    )
    assert not result.ok
    assert result.failure_reason == GraspFailureReason.OBJECT_TOO_LARGE
    assert result.plan is None
    metrics["oversized_cube_rejected"] = True

    result = planner.plan(world_object(size=None))
    assert not result.ok
    assert result.failure_reason == GraspFailureReason.SIZE_UNKNOWN
    metrics["unknown_size_rejected"] = True

    result = planner.plan(world_object(pose=False))
    assert not result.ok
    assert result.failure_reason == GraspFailureReason.OBJECT_POSE_UNKNOWN
    metrics["unknown_pose_rejected"] = True

    result = planner.plan(world_object(graspable=False))
    assert not result.ok
    assert result.failure_reason == GraspFailureReason.OBJECT_NOT_GRASPABLE
    metrics["non_graspable_rejected"] = True

    result = planner.plan(
        world_object(),
        grasp_hint="side",
    )
    assert not result.ok
    assert result.failure_reason == GraspFailureReason.UNSUPPORTED_HINT
    metrics["unsupported_hint_rejected"] = True

    model = WorldModel()
    obj = world_object(size=(0.09, 0.09, 0.05))
    model.register(obj)
    skill = PickSkill(
        FakeBackend(),
        model,
        planner,
    )
    result = skill.execute("object_1")
    assert not result.ok
    assert result.failure_code == FailureCode.NO_VALID_GRASP
    assert (
        result.details["grasp_failure_reason"]
        == GraspFailureReason.OBJECT_TOO_LARGE.value
    )
    metrics["pick_maps_feasibility_failure"] = True

    model = WorldModel()
    obj = world_object()
    model.register(obj)
    skill = PickSkill(
        FakeBackend(),
        model,
        AlternatePlanner(),
    )
    result = skill.execute("object_1")
    assert result.ok
    assert result.details["grasp_planner_details"]["strategy"] == (
        "alternate_test_planner"
    )
    metrics["planner_replaceability"] = True

    print("=== PHASE B UNIT ===")
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
