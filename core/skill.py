from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class SkillStatus(str, Enum):
    """Enum representing the status of a skill."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class FailureCode(str, Enum):
    OBJECT_NOT_FOUND = "object_not_found"
    OBJECT_POSE_UNKNOWN = "object_pose_unknown"

    UNREACHABLE = "unreachable"
    PATH_BLOCKED = "path_blocked"

    NO_VALID_GRASP = "no_valid_grasp"
    GRASP_FAILED = "grasp_failed"

    NOT_HOLDING_OBJECT = "not_holding_object"
    PLACE_FAILED = "place_failed"

    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"

@dataclass
class SkillResult:
    """Dataclass representing the result of a skill execution."""
    status: SkillStatus
    message: str = ""
    failure_code: FailureCode | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Check if the skill execution was successful."""
        return self.status == SkillStatus.SUCCESS


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str

    inputs: tuple[str, ...]
    preconditions: tuple[str, ...]
    effects: tuple[str, ...]
    failures: tuple[FailureCode, ...]


class BaseSkill(ABC):
    SPEC: SkillSpec

    @abstractmethod
    def execute(self, *args, **kwargs) -> SkillResult:
        ...