from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Detection:
    object_id: str
    mask: np.ndarray
    confidence: float = 1.0
    class_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.object_id:
            raise ValueError("Detection.object_id cannot be empty")

        self.mask = np.asarray(self.mask, dtype=bool)
        if self.mask.ndim != 2:
            raise ValueError("Detection.mask must be a 2D array")

        self.confidence = float(self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection.confidence must be in [0, 1]")


class Detector(Protocol):
    def detect(self, rgb: np.ndarray) -> list[Detection]:
        ...