"""Exact constant time scaling of an existing continuous joint trajectory."""
from __future__ import annotations

import math

import numpy as np


class TimeScaledTrajectory:
    """Retain the joint-space curve while slowing its time parameter.

    For an original interval [a,b], the new interval is [a,a+s*(b-a)].
    A new time t evaluates the original at a+(t-a)/s. The derivative of
    order n is the original derivative divided by s**n. This does not modify
    robot limits, path geometry, smoothing, or an already selected IK branch.

    Collision and execution validation remain the caller's responsibility.
    The ``trajectory.eval`` shape matches continuous trajectory adapters
    without depending on an Isaac or Lula type.
    """

    def __init__(self, trajectory, factor=2.):
        if (isinstance(factor, bool) or not isinstance(factor, (int, float))
                or not math.isfinite(factor) or not 1. <= factor <= 4.):
            raise ValueError('Trajectory time factor must be finite and between 1 and 4')
        first, last = float(trajectory.start_time), float(trajectory.end_time)
        if not math.isfinite(first) or not math.isfinite(last) or last < first:
            raise ValueError('The original trajectory must have a finite increasing time domain')
        if not callable(getattr(trajectory, 'get_joint_targets', None)):
            raise ValueError('The original trajectory must expose joint position and velocity targets')
        raw = getattr(trajectory, 'trajectory', trajectory)
        if not callable(getattr(raw, 'eval', None)):
            raise ValueError('The original trajectory must expose continuous derivative evaluation')
        combined = float(factor)*getattr(trajectory, 'time_scale_factor', 1.)
        if not math.isfinite(combined) or combined > 4.:
            raise ValueError('Combined trajectory time scaling must not exceed 4')
        self.base_trajectory = trajectory
        self.factor = float(factor)
        self.time_scale_factor = combined
        self.start_time = first
        self.end_time = first+self.factor*(last-first)
        if not math.isfinite(self.end_time):
            raise ValueError('Scaled trajectory time domain is not finite')
        self._base_end_time = last
        self._raw = raw

    @property
    def trajectory(self):
        return self

    def _base_time(self, time):
        stamp = float(time)
        if not math.isfinite(stamp) or stamp < self.start_time or stamp > self.end_time:
            raise ValueError('Requested trajectory time is outside its finite domain')
        if stamp == self.start_time:
            return self.start_time
        if stamp == self.end_time:
            return self._base_end_time
        return min(self._base_end_time, max(self.start_time,
            self.start_time+(stamp-self.start_time)/self.factor))

    def get_joint_targets(self, time):
        position, velocity = self.base_trajectory.get_joint_targets(self._base_time(time))
        return np.asarray(position, dtype=float).copy(), np.asarray(velocity, dtype=float)/self.factor

    def eval(self, time, derivative):
        if type(derivative) is not int or not 0 <= derivative <= 3:
            raise ValueError('Supported derivative orders are integers 0 through 3')
        value = self._raw.eval(self._base_time(time), derivative)
        return np.asarray(value, dtype=float)/self.factor**derivative

    def get_active_joints(self):
        return self.base_trajectory.get_active_joints()
