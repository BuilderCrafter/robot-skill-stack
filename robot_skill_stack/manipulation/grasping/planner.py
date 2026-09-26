from __future__ import annotations

from typing import Protocol

from robot_skill_stack.manipulation.grasping.types import GraspResult
from robot_skill_stack.world.model.entities import WorldObject


class GraspPlanner(Protocol):
    def plan(
        self,
        obj: WorldObject,
        *,
        lift_height: float | None = None,
        grasp_hint: str | None = None,
    ) -> GraspResult:
        ...
