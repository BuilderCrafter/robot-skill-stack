"""Motion execution stays behind the existing manipulation backend API.

Strategies execute a requested goal; they do not change skill contracts. A
future global planner will produce a joint-space route for a separate executor,
rather than forcing that route through Cartesian waypoint calls.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, TYPE_CHECKING

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.backend import BackendResult

if TYPE_CHECKING:
    from robot_skill_stack.world.spatial_mapping.runtime import MappingReadView


class MotionStrategy(Protocol):
    """One selected strategy executes on the backend's owning context.

    Cancellation propagates to the caller after normal arm cleanup. A carrying
    motion checks grasp state after each control update, before completion.
    """
    async def move_to_pose_async(
        self,
        target: Pose,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
        grasp_check: Callable[[], BackendResult | None] | None = None,
    ) -> BackendResult:
        ...


class MappingSource(Protocol):
    """Read-only access structurally provided by V5's mapping service.

    ``snapshot()`` obtains a constant-time view, retaining original source time,
    generation, health and voxel-age semantics. Dense exports are explicit
    work; strategies must not scan/copy them in control callbacks. An enabled
    source can be waiting, stale, failed or closed, so its presence implies no
    collision-clearance guarantee.
    """
    def snapshot(self) -> MappingReadView:
        ...

    @property
    def health(self) -> Mapping[str, object]:
        ...


# Late binding preserves existing backend-before-mapping initialization. A
# missing provider, or a provider returning None, explicitly means unavailable.
MappingSourceProvider = Callable[[], MappingSource | None]
