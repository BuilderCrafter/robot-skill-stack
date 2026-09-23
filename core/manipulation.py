from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.types import Pose


@dataclass
class BackendResult:
    ok: bool
    message: str = ""
    timed_out: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class ManipulationBackend(ABC):
    @abstractmethod
    def move_to_pose(
        self,
        target: Pose,
        speed: float = 0.5,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
    ) -> BackendResult:
        ...

    @abstractmethod
    def open_gripper(self) -> BackendResult:
        ...

    @abstractmethod
    def close_gripper(self, width: float | None = None) -> BackendResult:
        ...

    @abstractmethod
    def get_end_effector_pose(self) -> Pose:
        ...

    @abstractmethod
    def check_reachability(self, target: Pose) -> bool:
        ...

    @abstractmethod
    def home(self, name: str = "home") -> BackendResult:
        ...