from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from robot_skill_stack.common.types import Pose

@dataclass
class WorldObject:
    object_id: str
    class_name: str | None = None
    pose: Pose | None = None
    size: np.ndarray | None = None
    graspable: bool = True
    visible: bool = False
    confidence: float | None = None
    source: str | None = None
    last_seen: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        if not self.object_id: raise ValueError("WorldObject.object_id cannot be empty")
        if self.size is not None:
            self.size=np.asarray(self.size,dtype=float)
            if self.size.shape!=(3,): raise ValueError("WorldObject.size must have shape (3,)")
        if self.confidence is not None:
            self.confidence=float(self.confidence)
            if not 0 <= self.confidence <= 1: raise ValueError("WorldObject.confidence must be in [0, 1]")
