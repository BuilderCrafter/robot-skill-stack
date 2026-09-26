from __future__ import annotations

from typing import Protocol

from world_model.context import ObservationContext
from world_model.observations import ObjectObservation


class WorldObservationProvider(Protocol):
    def observe(
        self,
        context: ObservationContext | None = None,
    ) -> list[ObjectObservation]:
        ...
