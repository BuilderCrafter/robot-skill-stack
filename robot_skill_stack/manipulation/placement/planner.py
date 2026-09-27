from __future__ import annotations

from typing import Protocol

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.placement.types import PlacementResult
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.world_model import WorldModel


class PlacementPlanner(Protocol):
    def evaluate(
        self,
        obj: WorldObject,
        target: Pose,
        world_model: WorldModel,
    ) -> PlacementResult:
        ...
