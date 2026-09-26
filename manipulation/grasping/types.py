from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.types import Pose


class GraspFailureReason(str, Enum):
    OBJECT_POSE_UNKNOWN = "object_pose_unknown"
    SIZE_UNKNOWN = "size_unknown"
    OBJECT_NOT_GRASPABLE = "object_not_graspable"
    OBJECT_TOO_LARGE = "object_too_large"
    UNSUPPORTED_HINT = "unsupported_hint"
    NO_FEASIBLE_GRASP = "no_feasible_grasp"


@dataclass(frozen=True)
class ParallelJawGripperSpec:
    max_width: float
    min_width: float = 0.0

    def __post_init__(self):
        if self.max_width <= 0:
            raise ValueError("max_width must be > 0")
        if self.min_width < 0:
            raise ValueError("min_width must be >= 0")
        if self.min_width >= self.max_width:
            raise ValueError("min_width must be < max_width")


@dataclass
class GraspPlan:
    pre_grasp: Pose
    grasp: Pose
    lift: Pose


@dataclass
class GraspResult:
    plan: GraspPlan | None = None
    failure_reason: GraspFailureReason | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.plan is not None and self.failure_reason is None

    @classmethod
    def success(cls, plan: GraspPlan, **details):
        return cls(plan=plan, details=details)

    @classmethod
    def failure(
        cls,
        reason: GraspFailureReason,
        message: str,
        **details,
    ):
        return cls(
            plan=None,
            failure_reason=reason,
            message=message,
            details=details,
        )
