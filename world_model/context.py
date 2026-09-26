from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AssociationHint:
    object_id: str
    expected_position: np.ndarray
    max_distance: float
    expires_at: float
    reason: str = "unknown"

    def __post_init__(self):
        if not self.object_id:
            raise ValueError("AssociationHint.object_id cannot be empty")
        position = np.asarray(self.expected_position, dtype=float)
        if position.shape != (3,):
            raise ValueError("AssociationHint.expected_position must have shape (3,)")
        if not np.isfinite(position).all():
            raise ValueError("AssociationHint.expected_position must be finite")
        if self.max_distance <= 0:
            raise ValueError("AssociationHint.max_distance must be > 0")
        object.__setattr__(self, "expected_position", position.copy())
        object.__setattr__(self, "max_distance", float(self.max_distance))
        object.__setattr__(self, "expires_at", float(self.expires_at))


@dataclass(frozen=True)
class ObservationContext:
    held_object_id: str | None = None
    association_hints: tuple[AssociationHint, ...] = ()
