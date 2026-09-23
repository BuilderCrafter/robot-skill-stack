from __future__ import annotations

from core.manipulation import ManipulationBackend
from core.skill import (
    BaseSkill,
    FailureCode,
    SkillResult,
    SkillSpec,
    SkillStatus,
)
from core.types import Pose


class MoveToPoseSkill(BaseSkill):
    """
    Semantic skill for moving the robot end effector to a requested pose.

    This skill knows nothing about Isaac Sim, RMPflow, Lula, or Franka.
    Those details belong to ManipulationBackend.
    """

    SPEC = SkillSpec(
        name="move_to_pose",

        description=(
            "Move the robot end effector to a requested pose."
        ),

        inputs=(
            "target",
            "speed",
        ),

        preconditions=(
            "target pose is valid",
            "target pose is reachable",
        ),

        effects=(
            "end effector reaches the requested pose",
        ),

        failures=(
            FailureCode.UNREACHABLE,
            FailureCode.PATH_BLOCKED,
            FailureCode.TIMEOUT,
            FailureCode.INTERNAL_ERROR,
        ),
    )


    def __init__(
        self,
        backend: ManipulationBackend,
    ) -> None:
        self.backend = backend


    def execute(
        self,
        target: Pose,
        speed: float = 0.5,
    ) -> SkillResult:
        """
        Execute the move skill.

        Parameters
        ----------
        target:
            Desired end-effector pose.

        speed:
            Requested normalized motion speed.

            The generic skill exposes this parameter even though
            individual backends may interpret it differently.
        """

        # -------------------------------------------------------------
        # Validate semantic input
        # -------------------------------------------------------------

        if not isinstance(
            target,
            Pose,
        ):
            return SkillResult(
                status=SkillStatus.FAILED,
                failure_code=FailureCode.INTERNAL_ERROR,
                message=(
                    "MoveToPoseSkill expected a Pose target."
                ),
            )


        if speed <= 0.0:
            return SkillResult(
                status=SkillStatus.FAILED,
                failure_code=FailureCode.INTERNAL_ERROR,
                message=(
                    "Move speed must be greater than zero."
                ),
            )


        # -------------------------------------------------------------
        # Precondition: target must be reachable
        # -------------------------------------------------------------

        reachable = (
            self.backend.check_reachability(
                target
            )
        )


        if not reachable:
            return SkillResult(
                status=SkillStatus.FAILED,

                failure_code=(
                    FailureCode.UNREACHABLE
                ),

                message=(
                    "Requested end-effector pose "
                    "is unreachable."
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

                    "frame":
                        target.frame,
                },
            )


        # -------------------------------------------------------------
        # Delegate physical execution to backend
        # -------------------------------------------------------------

        backend_result = (
            self.backend.move_to_pose(
                target=target,
                speed=speed,
            )
        )


        # -------------------------------------------------------------
        # Success
        # -------------------------------------------------------------

        if backend_result.ok:
            return SkillResult(
                status=SkillStatus.SUCCESS,

                message=(
                    "End effector reached target pose."
                ),

                details={
                    **backend_result.details,

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
        # Timeout
        # -------------------------------------------------------------

        if backend_result.timed_out:
            return SkillResult(
                status=SkillStatus.TIMEOUT,

                failure_code=(
                    FailureCode.TIMEOUT
                ),

                message=(
                    backend_result.message
                    or "MoveToPose timed out."
                ),

                details=backend_result.details,
            )


        # -------------------------------------------------------------
        # Generic execution failure
        #
        # For V1 we call this PATH_BLOCKED.
        # Later the backend can return richer failure information.
        # -------------------------------------------------------------

        return SkillResult(
            status=SkillStatus.FAILED,

            failure_code=(
                FailureCode.PATH_BLOCKED
            ),

            message=(
                backend_result.message
                or "MoveToPose execution failed."
            ),

            details=backend_result.details,
        )