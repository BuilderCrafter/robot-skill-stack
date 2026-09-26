from __future__ import annotations

from typing import Protocol

from robot_skill_stack.world.model.context import ObservationContext
from robot_skill_stack.world.model.observations import ObjectObservation


class WorldObservationProvider(Protocol):
    def observe(
        self,
        context: ObservationContext | None = None,
    ) -> list[ObjectObservation]:
        ...
