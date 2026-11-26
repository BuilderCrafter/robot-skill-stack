from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PlacementFailureReason(str, Enum):
    OBJECT_SIZE_UNKNOWN = "object_size_unknown"
    TARGET_OCCUPIED = "target_occupied"


@dataclass
class PlacementResult:
    failure_reason: PlacementFailureReason | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.failure_reason is None

    @classmethod
    def success(cls, **details) -> PlacementResult:
        return cls(details=details)

    @classmethod
    def failure(
        cls,
        reason: PlacementFailureReason,
        message: str,
        **details,
    ) -> PlacementResult:
        return cls(
            failure_reason=reason,
            message=message,
            details=details,
        )
