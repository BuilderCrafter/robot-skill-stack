from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.types import Pose


@dataclass
class ObjectObservation:
    object_id: str
    class_name: str | None = None
    pose: Pose | None = None
    size: np.ndarray | None = None
    graspable: bool | None = None
    visible: bool = True
    confidence: float | None = None
    source: str = "unknown"
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.object_id:
            raise ValueError("ObjectObservation.object_id cannot be empty")

        if self.size is not None:
            self.size = np.asarray(self.size, dtype=float)
            if self.size.shape != (3,):
                raise ValueError("ObjectObservation.size must have shape (3,)")

        if self.confidence is not None:
            self.confidence = float(self.confidence)
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("ObjectObservation.confidence must be in [0, 1]")