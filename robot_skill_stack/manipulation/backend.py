from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

from robot_skill_stack.common.types import Pose, GripperState


@dataclass
class BackendResult:
    ok: bool
    message: str = ""
    timed_out: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GripperFeedback:
    state: GripperState
    width: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ManipulationBackend(ABC):
    """Commands yield on the simulator thread; queries do not advance simulation.

    Carrying commands call grasp_check after each update, before reporting
    completion. A returned failure must stop the command using its normal hold.
    Neither motion completion nor a gripper command alone proves attachment.
    """

    # Parallel-jaw grasp capabilities consumed by the composition root.
    grasp_min_width: float
    grasp_max_width: float
    # Some strategies deliberately reject intended-contact tasks until modeled.
    supports_contact_motion: bool = True

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    async def move_to_pose_async(
        self,
        target: Pose,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
        grasp_check: Callable[[], BackendResult | None] | None = None,
    ) -> BackendResult:
        ...

    @abstractmethod
    async def open_gripper_async(self) -> BackendResult:
        ...

    @abstractmethod
    async def close_gripper_async(self, width: float | None = None) -> BackendResult:
        ...

    @abstractmethod
    def gripper_feedback(self) -> GripperFeedback:
        """Read positions: distinguish supported grasp, confirmed empty and unknown."""
        ...

    async def inspect_gripper_async(self) -> GripperFeedback:
        """Resolve stationary feedback; simulators override with bounded sampling."""
        return self.gripper_feedback()

    @abstractmethod
    def get_end_effector_pose(self) -> Pose:
        ...

    @abstractmethod
    def check_reachability(self, target: Pose) -> bool:
        ...

    @abstractmethod
    async def home_async(self, name: str = "home", *, grasp_check=None) -> BackendResult:
        """Move to an adapter-defined named pose; this is not a safety guarantee."""
        ...
