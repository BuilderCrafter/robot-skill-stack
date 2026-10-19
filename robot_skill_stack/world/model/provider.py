from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
import time

from robot_skill_stack.world.model.context import ObservationContext
from robot_skill_stack.world.model.observations import ObjectObservation


@dataclass(frozen=True)
class ObservationBatch:
    """A fresh sample (possibly empty), or no new sample when timestamp is None.

    Timestamps use this process's monotonic clock. Adapters must not return
    cached observations as fresh samples. Source failures raise exceptions.
    """
    observations: list[ObjectObservation] = field(default_factory=list)
    timestamp: float | None = None


@dataclass
class ObservationHealth:
    last_frame_at: float | None = None
    error: str | None = None
    stale_after: float = 1.0

    @property
    def status(self):
        if self.error:
            return 'failed'
        if self.last_frame_at is None:
            return 'waiting'
        return 'stale' if time.monotonic()-self.last_frame_at >= self.stale_after else 'fresh'


class WorldObservationProvider(Protocol):
    def observe(self, context: ObservationContext | None = None) -> ObservationBatch: ...
    def forget(self, object_id: str) -> None: ...
    def close(self) -> None: ...
