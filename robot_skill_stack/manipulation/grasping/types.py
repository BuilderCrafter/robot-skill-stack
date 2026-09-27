from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import math

from robot_skill_stack.common.types import Pose


class GraspFailureReason(str, Enum):
    OBJECT_POSE_UNKNOWN = "object_pose_unknown"
    SIZE_UNKNOWN = "size_unknown"
    OBJECT_NOT_GRASPABLE = "object_not_graspable"
    OBJECT_TOO_LARGE = "object_too_large"
    UNSUPPORTED_HINT = "unsupported_hint"
    INVALID_GEOMETRY = "invalid_geometry"
    OBJECT_TOO_SMALL = "object_too_small"
    OBJECT_TOO_TALL = "object_too_tall"
    INSUFFICIENT_CLEARANCE = "insufficient_clearance"
    APPROACH_BLOCKED = "approach_blocked"
    UNSUPPORTED_ORIENTATION = "unsupported_orientation"
    NO_FEASIBLE_GRASP = "no_feasible_grasp"


@dataclass(frozen=True)
class ParallelJawGripperSpec:
    max_width: float
    min_width: float = 0.0

    # Envelope relative to the commanded TCP, NOT the panda_hand link origin.
    open_width: float | None = None
    tcp_to_palm: float = 0.030
    fingertip_offset: float = 0.012
    pad_above_tcp: float = 0.010
    pad_below_tcp: float = 0.010
    finger_depth: float = 0.024
    finger_thickness: float = 0.025
    palm_depth: float = 0.070
    palm_width: float = 0.170
    palm_height: float = 0.075

    @property
    def aperture(self) -> float:
        return self.max_width if self.open_width is None else self.open_width

    def __post_init__(self):
        for name in ("max_width", "tcp_to_palm", "fingertip_offset", "pad_above_tcp",
                     "pad_below_tcp", "finger_depth", "finger_thickness", "palm_depth",
                     "palm_width", "palm_height"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if not math.isfinite(self.min_width) or not 0 <= self.min_width < self.max_width:
            raise ValueError("min_width must be finite and in [0, max_width)")
        if not math.isfinite(self.aperture) or self.aperture < self.max_width:
            raise ValueError("open_width must be finite and >= max_width")
        if self.pad_below_tcp > self.fingertip_offset or self.pad_above_tcp > self.tcp_to_palm:
            raise ValueError("Contact pad must lie between fingertips and palm")


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
