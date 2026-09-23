from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.types import Pose


@dataclass
class BackendResult:
    """
    Generic result returned by a robot backend.

    This is intentionally simulator-independent.
    """

    ok: bool

    message: str = ""

    timed_out: bool = False

    details: dict[str, Any] = field(
        default_factory=dict
    )


class ManipulationBackend(ABC):
    """
    Platform-independent robot manipulation interface.

    Skills depend on this interface.

    Isaac Sim, MoveIt, and a future real robot should
    implement this interface separately.
    """

    @abstractmethod
    def move_to_pose(
        self,
        target: Pose,
        speed: float = 0.5,
    ) -> BackendResult:
        """
        Move the end effector toward target.
        """
        ...


    @abstractmethod
    def open_gripper(
        self,
    ) -> BackendResult:
        """
        Open the robot gripper.
        """
        ...


    @abstractmethod
    def close_gripper(
        self,
        width: float | None = None,
    ) -> BackendResult:
        """
        Close the robot gripper.

        width=None means fully closed.
        """
        ...


    @abstractmethod
    def get_end_effector_pose(
        self,
    ) -> Pose:
        """
        Return the current end-effector pose.
        """
        ...


    @abstractmethod
    def check_reachability(
        self,
        target: Pose,
    ) -> bool:
        """
        Check whether the target pose has an IK solution.
        """
        ...


    @abstractmethod
    def home(
        self,
        name: str = "home",
    ) -> BackendResult:
        """
        Move the robot into a named safe pose.
        """
        ...