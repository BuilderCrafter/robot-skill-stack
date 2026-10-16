from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from robot_skill_stack.common.types import Pose


@dataclass
class BackendResult:
    ok: bool
    message: str = ""
    timed_out: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class ManipulationBackend(ABC):
    """Commands yield on the simulator thread; queries do not advance simulation."""

    # Parallel-jaw grasp capabilities consumed by the composition root.
    grasp_min_width: float
    grasp_max_width: float

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    async def move_to_pose_async(
        self,
        target: Pose,
        speed: float = 0.5,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
    ) -> BackendResult:
        ...

    @abstractmethod
    async def open_gripper_async(self) -> BackendResult:
        ...

    @abstractmethod
    async def close_gripper_async(self, width: float | None = None) -> BackendResult:
        ...

    @abstractmethod
    def verify_grasp(self) -> BackendResult:
        """Return whether gripper feedback indicates an object is currently held."""
        ...

    @abstractmethod
    def get_end_effector_pose(self) -> Pose:
        ...

    @abstractmethod
    def check_reachability(self, target: Pose) -> bool:
        ...

    @abstractmethod
    async def home_async(self, name: str = "home") -> BackendResult:
        ...
