from __future__ import annotations

import argparse
import asyncio
import sys

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.backend import (
    BackendResult,
    ManipulationBackend,
)
from robot_skill_stack.manipulation.grasping import TopDownGraspPlanner
from robot_skill_stack.manipulation.skills.home import HomeSkill
from robot_skill_stack.manipulation.skills.move_to_pose import MoveToPoseSkill
from robot_skill_stack.manipulation.skills.pick import PickSkill
from robot_skill_stack.runtime.registry import SkillRegistry
from robot_skill_stack.runtime.runtime import RobotRuntime
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.world_model import WorldModel
from tests.support.results import write_result


class FakeAsyncBackend(ManipulationBackend):
    def __init__(self):
        self.ee = Pose(
            [0.45, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.0],
        )
        self.async_calls = 0

    def check_reachability(self, target):
        return True

    def get_end_effector_pose(self):
        return self.ee

    def move_to_pose(self, *args, **kwargs):
        raise AssertionError("sync move path was used")

    def open_gripper(self):
        raise AssertionError("sync gripper path was used")

    def close_gripper(self, width=None):
        raise AssertionError("sync gripper path was used")

    def home(self, name="home"):
        raise AssertionError("sync home path was used")

    def verify_grasp(self):
        return BackendResult(
            True,
            details={"estimated_width_m": 0.05},
        )

    async def move_to_pose_async(self, target, **kwargs):
        await asyncio.sleep(0)
        self.async_calls += 1
        self.ee = target
        return BackendResult(True)

    async def open_gripper_async(self):
        await asyncio.sleep(0)
        self.async_calls += 1
        return BackendResult(True)

    async def close_gripper_async(self, width=None):
        await asyncio.sleep(0)
        self.async_calls += 1
        return BackendResult(True)

    async def home_async(self, name="home"):
        await asyncio.sleep(0)
        self.async_calls += 1
        return BackendResult(
            True,
            f"Robot reached named pose '{name}'.",
        )


async def run():
    backend = FakeAsyncBackend()
    model = WorldModel()
    model.register(
        WorldObject(
            object_id="object_1",
            class_name="cube",
            pose=Pose([0.45, 0.0, 0.025]),
            size=np.array([0.05, 0.05, 0.05]),
            visible=True,
        )
    )

    registry = SkillRegistry()
    registry.register(MoveToPoseSkill(backend))
    registry.register(HomeSkill(backend))
    registry.register(
        PickSkill(
            backend,
            model,
            TopDownGraspPlanner(),
        )
    )
    runtime = RobotRuntime(registry)

    move = await runtime.execute_async(
        "move_to_pose",
        target=Pose(
            [0.45, 0.0, 0.15],
            [0.0, 0.0, 1.0, 0.0],
        ),
    )
    assert move.ok

    home = await runtime.execute_async("home")
    assert home.ok

    pick = await runtime.execute_async(
        "pick",
        object_id="object_1",
    )
    assert pick.ok
    assert model.held_object_id == "object_1"
    assert backend.async_calls >= 6

    return {
        "move_async": move.ok,
        "home_async": home.ok,
        "pick_async": pick.ok,
        "held_object": model.held_object_id,
        "async_backend_calls": backend.async_calls,
    }


def main(result_path=None):
    metrics = asyncio.run(run())
    print("=== ASYNC RUNTIME UNIT ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("PASS")
    print("==========================")
    write_result(result_path, status="PASS", metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result")
    args = parser.parse_args()
    try:
        main(args.result)
    except Exception as exc:
        write_result(
            args.result,
            status="FAIL",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
