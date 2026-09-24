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


class Detector(Protocol):
    def detect_all(self) -> list[Detection]:
        ...

    def detect(self, object_id: str) -> Detection | None:
        ...