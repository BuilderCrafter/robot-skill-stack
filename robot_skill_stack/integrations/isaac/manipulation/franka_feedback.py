"""Control-frame velocity feedback for the NumPy Franka motion adapter."""
import numpy as np

from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy


class FrankaFrameVelocityPolicy(ArticulationMotionPolicy):
    """Use measured frame displacement instead of contact-sensitive raw velocity.

    The first sample has no velocity history and uses zero. History survives
    Cartesian-goal resets; gripper-only gaps use their full simulation duration.
    Joint positions, RMPflow configuration and returned targets stay unchanged.
    """
    def __init__(self, robot, motion_policy, world):
        super().__init__(robot, motion_policy, default_physics_dt=world.get_physics_dt())
        self.world = world
        self._previous_positions = None
        self._previous_time = None
        self.joint_velocity_feedback = None

    def get_next_articulation_action(self, physics_dt=None):
        active = self.get_active_joints_subset()
        watched = self.get_watched_joints_subset()
        positions = np.asarray(active.get_joint_positions(), dtype=float).copy()
        now = float(self.world.current_time)
        if self._previous_time is not None and now > self._previous_time:
            velocities = (positions-self._previous_positions)/(now-self._previous_time)
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
