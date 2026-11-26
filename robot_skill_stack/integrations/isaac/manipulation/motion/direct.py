"""The validated direct Cartesian RMPflow loop; SDK resources belong to the backend."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from isaacsim.core.utils.types import ArticulationAction

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.backend import BackendResult

if TYPE_CHECKING:
    from ..franka_backend import IsaacFrankaBackend


class DirectRmpflowStrategy:
    """Execute Cartesian targets on Kit using the backend's controller and IK."""

    def __init__(self, backend: IsaacFrankaBackend) -> None:
        self._backend = backend

    async def move_to_pose_async(
        self,
        target: Pose,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
        grasp_check: Callable[[], BackendResult | None] | None = None,
    ) -> BackendResult:
        backend = self._backend
        if target.frame != "world":
            return BackendResult(False, f"Unsupported frame: {target.frame}")
        if not backend.check_reachability(target):
            return BackendResult(False, "Target pose has no IK solution.")

        pos_tol = backend.position_tolerance if position_tolerance is None else position_tolerance
        ori_tol = backend.orientation_tolerance if orientation_tolerance is None else orientation_tolerance

        backend._reset_motion_controller()
        last_pos_error = last_ori_error = None

        last_action = None
        try:
            for step in range(backend.max_motion_steps):
                if not backend.world.is_playing():
                    await backend._next_update()
                    continue

                action = backend.motion_controller.forward(
                    target_end_effector_position=target.position,
                    target_end_effector_orientation=target.orientation,
                )
                backend.robot.apply_action(action)
                last_action = action
                await backend._next_update()

                if grasp_check is not None:
                    grasp_failure = grasp_check()
                    if grasp_failure is not None:
                        return grasp_failure

                current = backend.get_end_effector_pose()
                last_pos_error = float(
                    np.linalg.norm(current.position - target.position)
                )

                if target.orientation is None:
                    last_ori_error = None
                    orientation_ok = True
                else:
                    last_ori_error = backend._quat_error(
                        current.orientation,
                        target.orientation,
                    )
                    orientation_ok = last_ori_error <= ori_tol

                if last_pos_error <= pos_tol and orientation_ok:
                    return BackendResult(
                        True,
                        "End effector reached target pose.",
                        details={
                            "steps": step + 1,
                            "position_error_m": last_pos_error,
                            "orientation_error_rad": last_ori_error,
                            "position_tolerance_m": pos_tol,
                            "orientation_tolerance_rad": ori_tol,
                        },
                    )

            return BackendResult(
                False,
                "Motion timed out before reaching target pose.",
                timed_out=True,
                details={
                    "max_steps": backend.max_motion_steps,
                    "position_error_m": last_pos_error,
                    "orientation_error_rad": last_ori_error,
                    "position_tolerance_m": pos_tol,
                    "orientation_tolerance_rad": ori_tol,
                },
            )
        finally:
            if last_action is not None:
                # Partial finger commands leave arm targets latched. Retain the
                # final arm position target, but stop requesting continued motion.
                backend.robot.apply_action(ArticulationAction(
                    joint_positions=last_action.joint_positions,
                    joint_velocities=np.zeros_like(last_action.joint_positions, dtype=float),
                    joint_indices=last_action.joint_indices,
                ))
