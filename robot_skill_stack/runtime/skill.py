from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SkillStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class FailureCode(str, Enum):
    BUSY = "busy"
    OBJECT_NOT_FOUND = "object_not_found"
    WORLD_STATE_UNAVAILABLE = "world_state_unavailable"
    OBJECT_POSE_UNKNOWN = "object_pose_unknown"
    OBJECT_MOVED = "object_moved"

    UNREACHABLE = "unreachable"
    PATH_BLOCKED = "path_blocked"

    NO_VALID_GRASP = "no_valid_grasp"
    GRASP_FAILED = "grasp_failed"
    GRASP_LOST = "grasp_lost"
    GRASP_UNCERTAIN = "grasp_uncertain"

    NOT_HOLDING_OBJECT = "not_holding_object"
    TARGET_OCCUPIED = "target_occupied"
    PLACE_FAILED = "place_failed"

    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


@dataclass
class SkillResult:
    status: SkillStatus
    message: str = ""
    failure_code: FailureCode | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == SkillStatus.SUCCESS


class BaseSkill(ABC):
    name: str

    @abstractmethod
    async def execute_async(self, *args, **kwargs) -> SkillResult:
        ...
