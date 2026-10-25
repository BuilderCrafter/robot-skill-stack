"""Small SDK-independent contracts for seven-joint plans and execution checks."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class PlanningFailure(RuntimeError):
    """An explicit failure phase; callers must never turn this into direct motion."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = {} if details is None else dict(details)


@dataclass(frozen=True)
class JointRoute:
    """Ordered arm configurations, with no SDK objects or Cartesian conversion."""

    joint_names: tuple[str, ...]
    waypoints: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        names = tuple(self.joint_names)
        map_arm_joints(names, names)
        points = route_array(self.waypoints)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "waypoints", tuple(tuple(map(float, q)) for q in points))

    @property
    def length_rad(self) -> float:
        return float(np.linalg.norm(np.diff(np.asarray(self.waypoints), axis=0), axis=1).sum())


def map_arm_joints(active_names, articulation_names) -> tuple[int, ...]:
    """Retain planner order and exclude fingers even if articulation order differs."""
    names, dofs = tuple(active_names), tuple(articulation_names)
    required = {f"panda_joint{i}" for i in range(1, 8)}
    if len(names) != 7 or set(names) != required:
        raise PlanningFailure("joint_mapping_failure", "Planner must name exactly the seven Panda arm joints.")
    if len(dofs) != len(set(dofs)) or not required <= set(dofs):
        raise PlanningFailure("joint_mapping_failure", "Articulation arm names are missing or ambiguous.")
    return tuple(dofs.index(name) for name in names)


def route_array(route) -> np.ndarray:
    points = np.asarray(route, dtype=float)
    if points.ndim != 2 or points.shape[1] != 7 or len(points) < 1 or not np.isfinite(points).all():
        raise PlanningFailure("invalid_route", "Joint route must contain finite seven-joint configurations.")
    return points.copy()


def check_joint_limits(points, lower, upper, *, tolerance=1e-8) -> None:
    values = route_array(points)
    low, high = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (low.shape != (7,) or high.shape != (7,) or not np.isfinite(low).all()
            or not np.isfinite(high).all() or np.any(low >= high)):
        raise PlanningFailure("joint_mapping_failure", "Invalid seven-joint position limits.")
    if np.any(values < low-tolerance) or np.any(values > high+tolerance):
        raise PlanningFailure("joint_limit_failure", "A route or trajectory sample exceeds joint limits.")


def interpolate_route(route, *, max_joint_step=.01, max_samples=100_000) -> np.ndarray:
    """Sample every linear segment with bounded L-infinity joint displacement.

    This is a discrete resolution, not a continuous swept-volume certificate.
    Every original waypoint is retained, including both endpoints of the route.
    """
    points = route_array(route)
    if not math.isfinite(max_joint_step) or max_joint_step <= 0 or max_samples < 1:
        raise ValueError("Interpolation bounds must be finite and positive")
    segments = []
    total = 1
    for first, last in zip(points, points[1:]):
        count = max(1, int(math.ceil(float(np.max(np.abs(last-first)))/max_joint_step)))
        total += count
        if total > max_samples:
            raise PlanningFailure("validation_budget_exhausted", "Route exceeds bounded validation sample count.")
        segments.append(np.linspace(first, last, count+1)[1:])
    return np.concatenate((points[:1], *segments), axis=0)


def simulation_progress(previous_time: float, current_time: float, previous_index: int,
                        current_index: int, *, maximum_pause_seconds: float,
                        stalled_wall_seconds: float) -> bool:
    """Reject time/index resets; report whether physics advanced since the sample."""
    values = (previous_time, current_time, maximum_pause_seconds, stalled_wall_seconds)
    if not all(math.isfinite(value) for value in values):
        raise PlanningFailure("execution_failure", "Invalid simulation clock.")
    if current_time < previous_time-1e-9 or current_index < previous_index:
        raise PlanningFailure("lifecycle_invalidated", "Simulation reset invalidated the frozen trajectory.")
    advanced = current_time > previous_time+1e-12
    if not advanced and stalled_wall_seconds >= maximum_pause_seconds:
        raise PlanningFailure("execution_timeout", "Physics did not advance during trajectory execution.")
    return advanced
