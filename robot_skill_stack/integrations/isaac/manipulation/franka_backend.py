"""Franka controller ownership, measured gripper feedback and async commands."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
    MotionPolicyController,
)
from isaacsim.robot_motion.motion_generation.lula.motion_policies import RmpFlow

from robot_skill_stack.common.types import GripperState, Pose
from robot_skill_stack.manipulation.backend import BackendResult, GripperFeedback, ManipulationBackend
from robot_skill_stack.manipulation.motion import MotionStrategy
from robot_skill_stack.integrations.isaac.manipulation.franka_feedback import FrankaFrameVelocityPolicy
from robot_skill_stack.integrations.isaac.manipulation.motion.direct import DirectRmpflowStrategy

if TYPE_CHECKING:
    from isaacsim.core.api import World
    from isaacsim.robot.manipulators.examples.franka import Franka


class IsaacFrankaBackend(ManipulationBackend):
    """Own the SDK controllers; delegate arm commands to one selected strategy."""

    def __init__(
        self,
        world: World,
        robot: Franka,
        *,
        end_effector_frame: str = "right_gripper",
        position_tolerance: float = 0.025,
        orientation_tolerance: float = 0.10,
        max_motion_steps: int = 1000,
        grasp_min_width: float = 0.005,
        grasp_max_width: float = 0.075,
        strategy_factory: Callable[[IsaacFrankaBackend], MotionStrategy] | None = None,
    ) -> None:
        self.world = world
        self.robot = robot
        self.end_effector_frame = end_effector_frame
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.max_motion_steps = max_motion_steps
        self.grasp_min_width = float(grasp_min_width)
        self.grasp_max_width = float(grasp_max_width)
        self._gripper_command: str | None = None
        # SDK nominal opening/action targets can exceed the loaded USD travel.
        finger_indices = [robot.get_dof_index(f'panda_finger_joint{i}') for i in (1, 2)]
        properties = robot.dof_properties
        self._finger_limits = np.column_stack((
            properties['lower'][finger_indices], properties['upper'][finger_indices],
        ))

        rmpflow = RmpFlow(**interface_config_loader.load_supported_motion_policy_config('Franka', 'RMPflow'))
        self.motion_controller = MotionPolicyController(
            name="franka_backend_rmpflow",
            articulation_motion_policy=FrankaFrameVelocityPolicy(robot, rmpflow, world),
        )
        self._reset_motion_controller()

        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self.lula_solver = LulaKinematicsSolver(**kinematics_config)
        self.kinematics_solver = ArticulationKinematicsSolver(
            robot,
            self.lula_solver,
            end_effector_frame,
        )

        # Preserve the validated target: TCP after bootstrap reset and startup updates.
        # This is a startup reference, not a safe joint configuration or collision-free path.
        home = self.get_end_effector_pose()
        self.named_poses = {
            "home": Pose(
                home.position.copy(),
                home.orientation.copy(),
                home.frame,
            )
        }

        self._motion_active = False
        self._motion_strategy = (DirectRmpflowStrategy(self) if strategy_factory is None
                                 else strategy_factory(self))

    @property
    def motion_strategy(self) -> MotionStrategy:
        """The strategy selected once at construction; SDK ownership stays here."""
        return self._motion_strategy

    @property
    def supports_contact_motion(self) -> bool:
        return getattr(self._motion_strategy, 'supports_contact_motion', True)

    def close(self) -> None:
        close = getattr(self._motion_strategy, 'close', None)
        if close is not None:
            close()

    @property
    def is_ready(self) -> bool:
        return bool(self.robot.handles_initialized)

    def _update_base(self) -> None:
        position, orientation = self.robot.get_world_pose()
        self.lula_solver.set_robot_base_pose(position, orientation)

    def _reset_motion_controller(self) -> None:
        self.motion_controller.reset()
        # The SDK example controller also restores base pose after RMPflow reset.
        position, orientation = self.robot.get_world_pose()
        self.motion_controller.get_motion_policy().set_robot_base_pose(
            robot_position=position, robot_orientation=orientation,
        )

    @staticmethod
    def _quat_error(q1: np.ndarray, q2: np.ndarray) -> float:
        """Shortest orientation angle in radians; quaternion signs are equivalent."""
        q1 = np.asarray(q1) / np.linalg.norm(q1)
        q2 = np.asarray(q2) / np.linalg.norm(q2)
        return float(2 * np.arccos(np.clip(abs(np.dot(q1, q2)), -1, 1)))

    def _measure_gripper(self) -> tuple[np.ndarray, float | None]:
        """Read finger positions and estimate nonnegative total jaw width."""
        joints = np.asarray(self.robot.gripper.get_joint_positions(), dtype=float).reshape(-1)
        finite = bool(joints.size >= 2 and np.isfinite(joints).all())
        width = float(np.sum(np.clip(joints, 0.0, None))) if finite else None
        return joints, width

    def get_end_effector_pose(self) -> Pose:
        self._update_base()
        position, rotation = self.kinematics_solver.compute_end_effector_pose()
        orientation = rot_matrices_to_quats(np.asarray(rotation, dtype=float))
        return Pose(np.asarray(position, dtype=float), np.asarray(orientation), "world")

    def check_reachability(self, target: Pose) -> bool:
        self._update_base()
        _, success = self.kinematics_solver.compute_inverse_kinematics(
            target_position=target.position,
            target_orientation=target.orientation,
        )
        return bool(success)

    def gripper_feedback(self) -> GripperFeedback:
        """Classify measured travel using physical limits and the last command."""
        try:
            joints, width = self._measure_gripper()
        except Exception as exc:
            return GripperFeedback(GripperState.UNCERTAIN, details={'feedback_error': str(exc)})
        details = {
            "joint_positions": joints.tolist(),
            "estimated_width_m": width,
            "minimum_grasp_width_m": self.grasp_min_width,
            "maximum_grasp_width_m": self.grasp_max_width,
            "finger_lower_limits_m": self._finger_limits[:, 0].tolist(),
            "finger_upper_limits_m": self._finger_limits[:, 1].tolist(),
            "nominal_open_positions_m": np.asarray(self.robot.gripper.joint_opened_positions).tolist(),
        }
        lower, opened = self._finger_limits[:, 0], self._finger_limits[:, 1]
        tolerance = self.grasp_min_width / 2
        valid = (joints.shape == (2,) and width is not None and
                 np.all(joints >= lower - tolerance) and np.all(joints <= opened + tolerance))
        state = GripperState.UNCERTAIN
        if valid:
            if width < self.grasp_min_width:
                state = GripperState.FREE
            elif self._gripper_command == 'close' and width <= self.grasp_max_width:
                state = GripperState.HELD
            elif np.all(np.abs(joints - opened) <= tolerance):
                state = GripperState.FREE
        details['grasp_state'] = state.value
        return GripperFeedback(state, width, details)

    async def inspect_gripper_async(self) -> GripperFeedback:
        """Require three stable samples within 12 updates (normally 0.2 s)."""
        previous = None
        previous_time = None
        stable = 0
        for sample in range(13):
            feedback = self.gripper_feedback()
            now = float(self.world.current_time)
            same = (previous is not None and now > previous_time and feedback.state == previous.state and
                    feedback.width is not None and previous.width is not None and
                    abs(feedback.width - previous.width) <= .001)
            stable = stable + 1 if same else 1
            if feedback.state != GripperState.UNCERTAIN and stable >= 3:
                return feedback
            previous = feedback
            previous_time = now
            if sample < 12:
                await self._next_update()
        return GripperFeedback(GripperState.UNCERTAIN, feedback.width, feedback.details)

    async def _next_update(self) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()

    async def move_to_pose_async(
        self,
        target: Pose,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
        grasp_check: Callable[[], BackendResult | None] | None = None,
    ) -> BackendResult:
        # One arm-command owner on the cooperative Kit loop, including direct
        # backend callers that do not enter RobotRuntime's skill busy guard.
        if self._motion_active:
            return BackendResult(False, "Arm motion is already in progress.")
        self._motion_active = True
        try:
            return await self._motion_strategy.move_to_pose_async(
                target,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
                grasp_check=grasp_check,
            )
        finally:
            self._motion_active = False

    async def open_gripper_async(self) -> BackendResult:
        """Apply 60 opening updates, then verify travel from measured fingers."""
        self._gripper_command = 'open'
        action = self.robot.gripper.forward(action="open")
        for _ in range(60):
            self.robot.apply_action(action)
            await self._next_update()

        feedback = self.gripper_feedback()
        opened = (feedback.state == GripperState.FREE and feedback.width is not None and
                  feedback.width > self.grasp_max_width)
        return BackendResult(
            opened,
            "Gripper opened." if opened else "Measured finger positions do not confirm opening.",
            details=feedback.details,
        )

    async def close_gripper_async(self, width: float | None = None) -> BackendResult:
        """Apply 90 closing updates; command completion alone does not prove a grasp."""
        if width is not None:
            return BackendResult(
                False,
                "Explicit gripper width commands are unsupported.",
            )

        self._gripper_command = 'close'
        action = self.robot.gripper.forward(action="close")
        for _ in range(90):
            self.robot.apply_action(action)
            await self._next_update()

        joints, measured_width = self._measure_gripper()
        return BackendResult(
            True,
            "Gripper close command completed.",
            details={
                "joint_positions": joints.tolist(),
                "estimated_width_m": measured_width,
            },
        )

    async def home_async(self, name: str = "home", *,
                         grasp_check: Callable[[], BackendResult | None] | None = None) -> BackendResult:
        """Move to a named TCP pose; Home is the measured startup reference."""
        if name not in self.named_poses:
            return BackendResult(
                False,
                f"Unknown named pose '{name}'.",
                details={"available_poses": list(self.named_poses)},
            )

        result = await self.move_to_pose_async(
            self.named_poses[name],
            position_tolerance=0.025,
            orientation_tolerance=0.10,
            grasp_check=grasp_check,
        )
        if result.ok:
            result.message = f"Robot reached named pose '{name}'."
        result.details["pose_name"] = name
        return result
