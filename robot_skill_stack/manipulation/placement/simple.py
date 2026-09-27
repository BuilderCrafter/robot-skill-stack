from __future__ import annotations

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.placement.types import (
    PlacementFailureReason,
    PlacementResult,
)
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.world_model import WorldModel


class SimplePlacementPlanner:
    """Cube-focused AABB target-occupancy check for stable table placement."""

    def __init__(self, *, clearance=0.005):
        if clearance < 0:
            raise ValueError("clearance must be >= 0")
        self.clearance = float(clearance)

    def evaluate(
        self,
        obj: WorldObject,
        target: Pose,
        world_model: WorldModel,
    ) -> PlacementResult:
        if obj.size is None:
            return PlacementResult.failure(
                PlacementFailureReason.OBJECT_SIZE_UNKNOWN,
                "Held object size is unknown; placement occupancy cannot be checked.",
                object_id=obj.object_id,
            )

        size = np.asarray(obj.size, dtype=float)
        checked = []
        ignored = []

        for other in world_model.objects():
            if other.object_id == obj.object_id:
                continue
            if not other.visible or other.pose is None or other.size is None:
                ignored.append(other.object_id)
                continue

            other_size = np.asarray(other.size, dtype=float)
            delta = np.abs(target.position - other.pose.position)
            required = (size + other_size) / 2.0 + self.clearance
            checked.append(other.object_id)

            if np.all(delta < required):
                return PlacementResult.failure(
                    PlacementFailureReason.TARGET_OCCUPIED,
                    f"Placement target is occupied by '{other.object_id}'.",
                    object_id=obj.object_id,
                    blocking_object_id=other.object_id,
                    target_position_m=target.position.tolist(),
                    blocking_position_m=other.pose.position.tolist(),
                    object_size_m=size.tolist(),
                    blocking_size_m=other_size.tolist(),
                    center_delta_m=delta.tolist(),
                    required_separation_m=required.tolist(),
                    clearance_m=self.clearance,
                    checked_object_ids=checked,
                    ignored_object_ids=ignored,
                )

        return PlacementResult.success(
            object_id=obj.object_id,
            target_position_m=target.position.tolist(),
            object_size_m=size.tolist(),
            clearance_m=self.clearance,
            checked_object_ids=checked,
            ignored_object_ids=ignored,
            feasibility_checked=True,
            strategy="simple_aabb",
        )
