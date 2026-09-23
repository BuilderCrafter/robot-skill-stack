from __future__ import annotations

from core.manipulation import (
    ManipulationBackend,
)

from core.skill import (
    BaseSkill,
    FailureCode,
    SkillResult,
    SkillSpec,
    SkillStatus,
)


class HomeSkill(BaseSkill):
    """
    Semantic skill that places the robot into a known safe pose.

    The actual definition of the named pose is owned by the backend.
    """

    SPEC = SkillSpec(
        name="home",

        description=(
            "Return the robot to a known safe "
            "joint configuration."
        ),

        inputs=(
            "optional named pose",
        ),

        preconditions=(),

        effects=(
            "robot reaches a known safe configuration",
        ),

        failures=(
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
        name: str = "home",
    ) -> SkillResult:

        # -------------------------------------------------------------
        # Validate request
        # -------------------------------------------------------------

        if not name:

            return SkillResult(
                status=SkillStatus.FAILED,

                failure_code=(
                    FailureCode.INTERNAL_ERROR
                ),

                message=(
                    "HomeSkill requires a "
                    "non-empty pose name."
                ),
            )

        # -------------------------------------------------------------
        # Delegate execution
        # -------------------------------------------------------------

        backend_result = (
            self.backend.home(
                name=name
            )
        )

        # -------------------------------------------------------------
        # Success
        # -------------------------------------------------------------

        if backend_result.ok:

            return SkillResult(
                status=SkillStatus.SUCCESS,

                message=(
                    f"Robot reached safe pose '{name}'."
                ),

                details={
                    **backend_result.details,
                    "pose_name": name,
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
                    or (
                        f"Timed out while moving "
                        f"to '{name}'."
                    )
                ),

                details=backend_result.details,
            )

        # -------------------------------------------------------------
        # Other backend failure
        # -------------------------------------------------------------

        return SkillResult(
            status=SkillStatus.FAILED,

            failure_code=(
                FailureCode.PATH_BLOCKED
            ),

            message=(
                backend_result.message
                or (
                    f"Could not reach "
                    f"named pose '{name}'."
                )
            ),

            details=backend_result.details,
        )