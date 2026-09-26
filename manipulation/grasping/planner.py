from __future__ import annotations

from typing import Protocol

from manipulation.grasping.types import GraspResult
from world_model.entities import WorldObject


class GraspPlanner(Protocol):
    def plan(
        self,
        obj: WorldObject,
        *,
        lift_height: float | None = None,
        grasp_hint: str | None = None,
    ) -> GraspResult:
        ...
