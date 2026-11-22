"""SDK-independent joint-space routes, validation, and execution checks."""
from __future__ import annotations

import math

import numpy as np


class PlanningFailure(RuntimeError):
    """An explicit failure phase; callers must never turn this into direct motion."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = {} if details is None else dict(details)


def map_joints(active_names, articulation_names) -> tuple[int, ...]:
    """Map arbitrary planner joint names to articulation indices in planner order."""
    names, dofs = tuple(active_names), tuple(articulation_names)
    if (not names or len(names) != len(set(names)) or
            any(not isinstance(name, str) or not name for name in names)):
        raise PlanningFailure("joint_mapping_failure", "Planner joint names must be unique and nonempty.")
    if (len(dofs) != len(set(dofs)) or any(not isinstance(name, str) or not name for name in dofs)
            or not set(names) <= set(dofs)):
        raise PlanningFailure("joint_mapping_failure", "Articulation joint names are missing or ambiguous.")
    return tuple(dofs.index(name) for name in names)


def route_array(route, *, expected_dof: int | None = None) -> np.ndarray:
    try:
        points = np.asarray(route, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanningFailure("invalid_route", "Joint route must contain finite joint configurations.") from exc
    if (points.ndim != 2 or points.shape[1] < 1 or len(points) < 1 or
            (expected_dof is not None and points.shape[1] != expected_dof) or
            not np.isfinite(points).all()):
        raise PlanningFailure("invalid_route", "Joint route must contain finite, consistently sized joint configurations.")
    return points.copy()


def check_joint_limits(points, lower, upper, *, tolerance=1e-8) -> None:
    values = route_array(points)
    low, high = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (low.shape != (values.shape[1],) or high.shape != (values.shape[1],) or
            not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high)):
        raise PlanningFailure("joint_mapping_failure", "Invalid joint position limits.")
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
