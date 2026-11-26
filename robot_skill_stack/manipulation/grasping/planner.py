from __future__ import annotations

from typing import Protocol

from robot_skill_stack.manipulation.grasping.types import GraspResult
from robot_skill_stack.world.model.entities import WorldObject


class GraspPlanner(Protocol):
    """Interchangeable object-to-grasp planning used by PickSkill."""

    def plan(
        self,
        obj: WorldObject,
        *,
        lift_height: float | None = None,
        grasp_hint: str | None = None,
    ) -> GraspResult:
        ...
