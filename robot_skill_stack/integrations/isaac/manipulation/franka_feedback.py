"""Control-frame velocity feedback for the NumPy Franka motion adapter."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy

if TYPE_CHECKING:
    from isaacsim.core.api import World
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot_motion.motion_generation.lula.motion_policies import RmpFlow


class FrankaFrameVelocityPolicy(ArticulationMotionPolicy):
    """Use measured frame displacement instead of contact-sensitive raw velocity.

    The first sample has no velocity history and uses zero. History survives
    Cartesian-goal resets; gripper-only gaps use their full simulation duration.
    Unchanged or rewound simulation time starts another zero-velocity sample.
    Joint positions, RMPflow configuration and returned targets stay unchanged.
    """
    def __init__(self, robot: Franka, motion_policy: RmpFlow, world: World) -> None:
        super().__init__(robot, motion_policy, default_physics_dt=world.get_physics_dt())
        self.world = world
        self._previous_positions: np.ndarray | None = None
        self._previous_time: float | None = None
        self.joint_velocity_feedback: np.ndarray | None = None

    def get_next_articulation_action(self, physics_dt: float | None = None) -> ArticulationAction:
        active = self.get_active_joints_subset()
        watched = self.get_watched_joints_subset()
        positions = np.asarray(active.get_joint_positions(), dtype=float).copy()
        now = float(self.world.current_time)
        if self._previous_time is not None and now > self._previous_time:
            velocities = (positions - self._previous_positions) / (now - self._previous_time)
        else:
            velocities = np.zeros_like(positions)
        # SDK RMPflow integration mutates its input arrays. Keep measurement
        # history and diagnostic feedback independent of that rollout.
        self._previous_positions = positions.copy()
        self._previous_time = now
        self.joint_velocity_feedback = velocities.copy()
        targets = self.get_motion_policy().compute_joint_targets(
            positions, velocities,
            watched.get_joint_positions(), watched.get_joint_velocities(),
            self.get_default_physics_dt() if physics_dt is None else physics_dt,
        )
        return active.make_articulation_action(*targets)
