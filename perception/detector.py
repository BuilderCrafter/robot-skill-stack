from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    object_id: str
    mask: np.ndarray
    confidence: float = 1.0


class Detector(Protocol):
    def detect(self, object_id: str) -> Detection | None:
        ...