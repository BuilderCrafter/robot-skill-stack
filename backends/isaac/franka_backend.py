from __future__ import annotations

import numpy as np

from core.manipulation import (
    BackendResult,
    ManipulationBackend,
)
from core.types import Pose


# ---------------------------------------------------------------------
# Isaac imports
#
# IMPORTANT:
# This module must only be imported AFTER SimulationApp has been created.
# ---------------------------------------------------------------------

from isaacsim.core.utils.numpy.rotations import (
    rot_matrices_to_quats,
)

from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import (
    RMPFlowController,
)

from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    ArticulationMotionPolicy,
    LulaKinematicsSolver,
    RmpFlow,
    interface_config_loader,
)


class IsaacFrankaBackend(ManipulationBackend):
    """
    Isaac Sim implementation of ManipulationBackend
    for the Franka Panda robot.

    Responsibilities:
        - Cartesian end-effector motion
        - named joint-space motion
        - gripper control
        - FK / IK reachability
    """

    def __init__(
        self,
        world,
        robot,
        *,
        end_effector_frame: str = "right_gripper",
        position_tolerance: float = 0.01,
        orientation_tolerance: float = 0.05,
        joint_tolerance: float = 0.02,
        max_motion_steps: int = 1000,
        max_home_steps: int = 1000,
    ) -> None:

        self.world = world
        self.robot = robot

        self.end_effector_frame = end_effector_frame

        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.joint_tolerance = joint_tolerance

        self.max_motion_steps = max_motion_steps
        self.max_home_steps = max_home_steps

        # -------------------------------------------------------------
        # Cartesian RMPflow controller
        #
        # This is the already-working controller used by MoveToPose.
        # -------------------------------------------------------------

        self.motion_controller = RMPFlowController(
            name="franka_backend_rmpflow",
            robot_articulation=self.robot,
        )

        self.motion_controller.reset()

        # -------------------------------------------------------------
        # Lula kinematics
        #
        # Used for:
        #   - forward kinematics
        #   - inverse kinematics / reachability
        # -------------------------------------------------------------

        kinematics_config = (
            interface_config_loader
            .load_supported_lula_kinematics_solver_config(
                "Franka"
            )
        )

        self.lula_solver = LulaKinematicsSolver(
            **kinematics_config
        )

        self.kinematics_solver = (
            ArticulationKinematicsSolver(
                self.robot,
                self.lula_solver,
                self.end_effector_frame,
            )
        )

        # -------------------------------------------------------------
        # Dedicated RMPflow instance for C-space / named-pose motion.
        #
        # This is separate from the Cartesian RMPFlowController above.
        # -------------------------------------------------------------

        home_rmp_config = (
            interface_config_loader
            .load_supported_motion_policy_config(
                "Franka",
                "RMPflow",
            )
        )

        self.home_rmpflow = RmpFlow(
            **home_rmp_config
        )

        self.home_motion_policy = (
            ArticulationMotionPolicy(
                self.robot,
                self.home_rmpflow,
            )
        )

        # -------------------------------------------------------------
        # Determine which articulation joints RMPflow actively controls.
        #
        # For Franka this should correspond to panda_joint1..7.
        # We deliberately ask RMPflow instead of hardcoding ordering.
        # -------------------------------------------------------------

        self.home_active_joint_names = (
            self.home_rmpflow.get_active_joints()
        )

        self.home_active_joint_indices = np.array(
            [
                self.robot.get_dof_index(name)
                for name in self.home_active_joint_names
            ],
            dtype=np.int64,
        )

        if np.any(
            self.home_active_joint_indices < 0
        ):
            raise RuntimeError(
                "Could not resolve all RMPflow active joints "
                "in the Franka articulation."
            )

        # -------------------------------------------------------------
        # Capture V1 named pose: "home"
        #
        # IMPORTANT:
        # Construct this backend AFTER world.reset().
        #
        # This stores the robot's current post-reset configuration.
        # -------------------------------------------------------------

        full_joint_positions = np.asarray(
            self.robot.get_joint_positions(),
            dtype=float,
        )

        if full_joint_positions is None:
            raise RuntimeError(
                "Franka articulation is not initialized. "
                "Construct IsaacFrankaBackend after world.reset()."
            )

        home_joint_positions = (
            full_joint_positions[
                self.home_active_joint_indices
            ].copy()
        )

        self.named_joint_poses: dict[
            str,
            np.ndarray,
        ] = {
            "home": home_joint_positions,
        }

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _update_robot_base_pose(
        self,
    ) -> None:
        """
        Keep Lula kinematics synchronized with the robot's
        actual world-space base transform.
        """

        base_position, base_orientation = (
            self.robot.get_world_pose()
        )

        self.lula_solver.set_robot_base_pose(
            base_position,
            base_orientation,
        )

    @staticmethod
    def _quaternion_error(
        q1: np.ndarray,
        q2: np.ndarray,
    ) -> float:
        """
        Angular difference between two quaternions in radians.

        Quaternion convention:
            [w, x, y, z]
        """

        q1 = np.asarray(
            q1,
            dtype=float,
        )

        q2 = np.asarray(
            q2,
            dtype=float,
        )

        q1 = (
            q1
            / np.linalg.norm(q1)
        )

        q2 = (
            q2
            / np.linalg.norm(q2)
        )

        # q and -q represent the same orientation.
        dot = abs(
            float(
                np.dot(
                    q1,
                    q2,
                )
            )
        )

        dot = np.clip(
            dot,
            -1.0,
            1.0,
        )

        return float(
            2.0
            * np.arccos(dot)
        )

    # -----------------------------------------------------------------
    # End-effector state
    # -----------------------------------------------------------------

    def get_end_effector_pose(
        self,
    ) -> Pose:
        """
        Return the current Lula right_gripper pose.
        """

        self._update_robot_base_pose()

        position, rotation_matrix = (
            self.kinematics_solver
            .compute_end_effector_pose()
        )

        orientation = (
            rot_matrices_to_quats(
                np.asarray(
                    rotation_matrix,
                    dtype=float,
                )
            )
        )

        return Pose(
            position=np.asarray(
                position,
                dtype=float,
            ),
            orientation=np.asarray(
                orientation,
                dtype=float,
            ),
            frame="world",
        )

    # -----------------------------------------------------------------
    # Reachability
    # -----------------------------------------------------------------

    def check_reachability(
        self,
        target: Pose,
    ) -> bool:
        """
        Check whether Lula IK can find a solution.

        Does not move the robot.
        """

        self._update_robot_base_pose()

        _, success = (
            self.kinematics_solver
            .compute_inverse_kinematics(
                target_position=(
                    target.position
                ),
                target_orientation=(
                    target.orientation
                ),
            )
        )

        return bool(success)

    # -----------------------------------------------------------------
    # Cartesian motion
    # -----------------------------------------------------------------

    def move_to_pose(
        self,
        target: Pose,
        speed: float = 0.5,
    ) -> BackendResult:
        """
        Move the Franka end effector to a requested pose.

        Uses the Franka RMPFlowController.

        NOTE:
        V1 accepts a generic speed parameter but does not
        currently map it to RMPflow tuning.
        """

        if target.frame != "world":

            return BackendResult(
                ok=False,
                message=(
                    "IsaacFrankaBackend V1 supports "
                    "only world-frame targets."
                ),
                details={
                    "frame":
                        target.frame,
                },
            )

        # -------------------------------------------------------------
        # Pre-check reachability.
        # -------------------------------------------------------------

        if not self.check_reachability(
            target
        ):

            return BackendResult(
                ok=False,
                message=(
                    "Target pose has no IK solution."
                ),
                details={
                    "target_position":
                        target.position.tolist(),

                    "target_orientation":
                        (
                            None
                            if target.orientation is None
                            else target.orientation.tolist()
                        ),
                },
            )

        # -------------------------------------------------------------
        # Fresh Cartesian command.
        # -------------------------------------------------------------

        self.motion_controller.reset()

        last_position_error = None
        last_orientation_error = None

        for step in range(
            self.max_motion_steps
        ):

            if not self.world.is_playing():

                self.world.step(
                    render=True
                )

                continue

            action = (
                self.motion_controller.forward(
                    target_end_effector_position=(
                        target.position
                    ),
                    target_end_effector_orientation=(
                        target.orientation
                    ),
                )
            )

            self.robot.apply_action(
                action
            )

            self.world.step(
                render=True
            )

            # ---------------------------------------------------------
            # Verify actual pose.
            # ---------------------------------------------------------

            current_pose = (
                self.get_end_effector_pose()
            )

            position_error = float(
                np.linalg.norm(
                    current_pose.position
                    - target.position
                )
            )

            last_position_error = (
                position_error
            )

            if target.orientation is None:

                orientation_error = None
                orientation_ok = True

            else:

                orientation_error = (
                    self._quaternion_error(
                        current_pose.orientation,
                        target.orientation,
                    )
                )

                last_orientation_error = (
                    orientation_error
                )

                orientation_ok = (
                    orientation_error
                    <= self.orientation_tolerance
                )

            position_ok = (
                position_error
                <= self.position_tolerance
            )

            if (
                position_ok
                and orientation_ok
            ):

                return BackendResult(
                    ok=True,
                    message=(
                        "End effector reached target pose."
                    ),
                    details={
                        "steps":
                            step + 1,

                        "position_error_m":
                            position_error,

                        "orientation_error_rad":
                            orientation_error,

                        "requested_speed":
                            speed,
                    },
                )

        return BackendResult(
            ok=False,
            timed_out=True,
            message=(
                "Motion timed out before "
                "reaching target pose."
            ),
            details={
                "max_steps":
                    self.max_motion_steps,

                "position_error_m":
                    last_position_error,

                "orientation_error_rad":
                    last_orientation_error,

                "requested_speed":
                    speed,
            },
        )

    # -----------------------------------------------------------------
    # Gripper
    # -----------------------------------------------------------------

    def open_gripper(
        self,
    ) -> BackendResult:
        """
        Physically command the Franka gripper open.
        """

        action = (
            self.robot.gripper.forward(
                action="open"
            )
        )


        for _ in range(60):

            self.robot.apply_action(
                action
            )

            self.world.step(
                render=True
            )


        positions = (
            self.robot.gripper
            .get_joint_positions()
        )


        return BackendResult(
            ok=True,

            message="Gripper opened.",

            details={
                "joint_positions":
                    np.asarray(
                        positions
                    ).tolist(),
            },
        )

    def close_gripper(
        self,
        width: float | None = None,
    ) -> BackendResult:
        """
        Physically command the Franka gripper closed.

        V1 grasping uses width=None.

        Explicit partial widths are not implemented yet because
        contact-aware grasping should use controlled joint targets
        rather than teleporting finger state.
        """

        if width is not None:

            return BackendResult(
                ok=False,

                message=(
                    "Explicit gripper width is not "
                    "implemented in IsaacFrankaBackend V1."
                ),

                details={
                    "requested_width":
                        width,
                },
            )


        action = (
            self.robot.gripper.forward(
                action="close"
            )
        )


        # Keep applying the closing command while physics/contact
        # prevents the fingers from simply passing through the object.
        for _ in range(90):

            self.robot.apply_action(
                action
            )

            self.world.step(
                render=True
            )


        positions = (
            self.robot.gripper
            .get_joint_positions()
        )


        return BackendResult(
            ok=True,

            message="Gripper close command completed.",

            details={
                "joint_positions":
                    np.asarray(
                        positions
                    ).tolist(),
            },
        )

    # -----------------------------------------------------------------
    # Named joint-space motion
    # -----------------------------------------------------------------

    def home(
        self,
        name: str = "home",
    ) -> BackendResult:
        """
        Move Franka into a named joint-space configuration.

        V1:
            "home" is the post-world.reset() arm configuration
            captured when this backend was constructed.

        Execution:
            RMPflow C-space target with no Cartesian target.
        """

        # -------------------------------------------------------------
        # Validate requested pose.
        # -------------------------------------------------------------

        if (
            name
            not in self.named_joint_poses
        ):

            return BackendResult(
                ok=False,
                message=(
                    f"Unknown named robot pose: "
                    f"'{name}'."
                ),
                details={
                    "available_poses":
                        list(
                            self.named_joint_poses.keys()
                        ),
                },
            )

        target_positions = (
            self.named_joint_poses[
                name
            ].copy()
        )

        # -------------------------------------------------------------
        # Fresh C-space RMPflow command.
        # -------------------------------------------------------------

        self.home_rmpflow.reset()

        self.home_rmpflow.set_cspace_target(
            target_positions
        )

        # Important:
        #
        # RMPflow's C-space target normally acts as a null-space
        # preference while a Cartesian end-effector target exists.
        #
        # Explicitly clearing the end-effector target causes RMPflow
        # to drive directly toward the C-space target.
        self.home_rmpflow.set_end_effector_target(
            None,
            None,
        )

        physics_dt = (
            self.world
            .get_physics_context()
            .get_physics_dt()
        )

        last_error = None

        # -------------------------------------------------------------
        # Closed-loop home motion.
        # -------------------------------------------------------------

        for step in range(
            self.max_home_steps
        ):

            if not self.world.is_playing():

                self.world.step(
                    render=True
                )

                continue

            # ---------------------------------------------------------
            # Keep robot base synchronized.
            # ---------------------------------------------------------

            base_position, base_orientation = (
                self.robot.get_world_pose()
            )

            self.home_rmpflow.set_robot_base_pose(
                base_position,
                base_orientation,
            )

            # ---------------------------------------------------------
            # Compute next joint command.
            # ---------------------------------------------------------

            action = (
                self.home_motion_policy
                .get_next_articulation_action(
                    physics_dt
                )
            )

            self.robot.apply_action(
                action
            )

            self.world.step(
                render=True
            )

            # ---------------------------------------------------------
            # Measure actual C-space error.
            # ---------------------------------------------------------

            full_positions = np.asarray(
                self.robot.get_joint_positions(),
                dtype=float,
            )

            current_positions = (
                full_positions[
                    self.home_active_joint_indices
                ]
            )

            joint_errors = (
                current_positions
                - target_positions
            )

            max_error = float(
                np.max(
                    np.abs(
                        joint_errors
                    )
                )
            )

            last_error = (
                max_error
            )

            # ---------------------------------------------------------
            # Success.
            # ---------------------------------------------------------

            if (
                max_error
                <= self.joint_tolerance
            ):

                # Reset Cartesian RMPflow so next MoveToPose
                # starts with a fresh internal state.
                self.motion_controller.reset()

                return BackendResult(
                    ok=True,
                    message=(
                        f"Robot reached named pose "
                        f"'{name}'."
                    ),
                    details={
                        "steps":
                            step + 1,

                        "max_joint_error_rad":
                            max_error,

                        "target_joint_positions":
                            target_positions.tolist(),

                        "final_joint_positions":
                            current_positions.tolist(),
                    },
                )

        # -------------------------------------------------------------
        # Timeout.
        # -------------------------------------------------------------

        full_positions = np.asarray(
            self.robot.get_joint_positions(),
            dtype=float,
        )

        final_positions = (
            full_positions[
                self.home_active_joint_indices
            ]
        )

        return BackendResult(
            ok=False,
            timed_out=True,
            message=(
                f"Robot timed out while moving "
                f"to named pose '{name}'."
            ),
            details={
                "max_steps":
                    self.max_home_steps,

                "max_joint_error_rad":
                    last_error,

                "target_joint_positions":
                    target_positions.tolist(),

                "final_joint_positions":
                    final_positions.tolist(),
            },
        )