from __future__ import annotations

import numpy as np

from core.manipulation import BackendResult, ManipulationBackend
from core.types import Pose

from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import (
    RMPFlowController,
)
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)


class IsaacFrankaBackend(ManipulationBackend):
    def __init__(
        self,
        world,
        robot,
        *,
        end_effector_frame="right_gripper",
        position_tolerance=0.025,
        orientation_tolerance=0.10,
        joint_tolerance=0.02,     # kept for compatibility
        max_motion_steps=1000,
        max_home_steps=1000,      # kept for compatibility
    ):
        self.world = world
        self.robot = robot
        self.end_effector_frame = end_effector_frame
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.max_motion_steps = max_motion_steps

        self.motion_controller = RMPFlowController(
            name="franka_backend_rmpflow",
            robot_articulation=robot,
        )
        self.motion_controller.reset()

        cfg = interface_config_loader.load_supported_lula_kinematics_solver_config(
            "Franka"
        )
        self.lula_solver = LulaKinematicsSolver(**cfg)
        self.kinematics_solver = ArticulationKinematicsSolver(
            robot,
            self.lula_solver,
            end_effector_frame,
        )

        # V1 home = safe Cartesian pose at runtime initialization.
        home = self.get_end_effector_pose()
        self.named_poses = {
            "home": Pose(
                home.position.copy(),
                home.orientation.copy(),
                home.frame,
            )
        }

    def _update_base(self):
        pos, ori = self.robot.get_world_pose()
        self.lula_solver.set_robot_base_pose(pos, ori)

    @staticmethod
    def _quat_error(q1, q2):
        q1 = np.asarray(q1) / np.linalg.norm(q1)
        q2 = np.asarray(q2) / np.linalg.norm(q2)
        return float(2 * np.arccos(np.clip(abs(np.dot(q1, q2)), -1, 1)))

    def get_end_effector_pose(self) -> Pose:
        self._update_base()
        pos, rot = self.kinematics_solver.compute_end_effector_pose()
        quat = rot_matrices_to_quats(np.asarray(rot, dtype=float))
        return Pose(np.asarray(pos, dtype=float), np.asarray(quat), "world")

    def check_reachability(self, target: Pose) -> bool:
        self._update_base()
        _, success = self.kinematics_solver.compute_inverse_kinematics(
            target_position=target.position,
            target_orientation=target.orientation,
        )
        return bool(success)

    def move_to_pose(
        self,
        target: Pose,
        speed=0.5,
        position_tolerance=None,
        orientation_tolerance=None,
    ) -> BackendResult:
        if target.frame != "world":
            return BackendResult(False, f"Unsupported frame: {target.frame}")

        if not self.check_reachability(target):
            return BackendResult(False, "Target pose has no IK solution.")

        pos_tol = (
            self.position_tolerance
            if position_tolerance is None
            else position_tolerance
        )
        ori_tol = (
            self.orientation_tolerance
            if orientation_tolerance is None
            else orientation_tolerance
        )

        self.motion_controller.reset()
        last_pos_error = last_ori_error = None

        for step in range(self.max_motion_steps):
            if not self.world.is_playing():
                self.world.step(render=True)
                continue

            action = self.motion_controller.forward(
                target_end_effector_position=target.position,
                target_end_effector_orientation=target.orientation,
            )
            self.robot.apply_action(action)
            self.world.step(render=True)

            current = self.get_end_effector_pose()
            last_pos_error = float(
                np.linalg.norm(current.position - target.position)
            )

            if target.orientation is None:
                last_ori_error = None
                orientation_ok = True
            else:
                last_ori_error = self._quat_error(
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
                        "requested_speed": speed,
                        "position_tolerance_m": pos_tol,
                        "orientation_tolerance_rad": ori_tol,
                    },
                )

        return BackendResult(
            False,
            "Motion timed out before reaching target pose.",
            timed_out=True,
            details={
                "max_steps": self.max_motion_steps,
                "position_error_m": last_pos_error,
                "orientation_error_rad": last_ori_error,
                "position_tolerance_m": pos_tol,
                "orientation_tolerance_rad": ori_tol,
            },
        )

    def open_gripper(self) -> BackendResult:
        action = self.robot.gripper.forward(action="open")
        for _ in range(60):
            self.robot.apply_action(action)
            self.world.step(render=True)

        return BackendResult(
            True,
            "Gripper opened.",
            details={
                "joint_positions": np.asarray(
                    self.robot.gripper.get_joint_positions()
                ).tolist()
            },
        )

    def close_gripper(self, width=None) -> BackendResult:
        if width is not None:
            return BackendResult(
                False,
                "Explicit gripper width is not implemented in V1.",
            )

        action = self.robot.gripper.forward(action="close")
        for _ in range(90):
            self.robot.apply_action(action)
            self.world.step(render=True)

        return BackendResult(
            True,
            "Gripper close command completed.",
            details={
                "joint_positions": np.asarray(
                    self.robot.gripper.get_joint_positions()
                ).tolist()
            },
        )

    def home(self, name="home") -> BackendResult:
        if name not in self.named_poses:
            return BackendResult(
                False,
                f"Unknown named pose '{name}'.",
                details={"available_poses": list(self.named_poses)},
            )

        result = self.move_to_pose(
            self.named_poses[name],
            speed=0.35,
            position_tolerance=0.025,
            orientation_tolerance=0.10,
        )

        if result.ok:
            result.message = f"Robot reached named pose '{name}'."

        result.details["pose_name"] = name
        return result