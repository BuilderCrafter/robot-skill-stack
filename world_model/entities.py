from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from core.types import Pose


@dataclass
class WorldObject:
    """
    Semantic representation of an object known to the robot.

    Nothing in this class is Isaac-specific.
    """

    object_id: str
    pose: Pose | None = None

    # Object dimensions [x, y, z] in meters
    size: np.ndarray | None = None

    graspable: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.size is not None:
            self.size = np.asarray(self.size, dtype=float)

            if self.size.shape != (3,):
                raise ValueError("WorldObject.size must have shape (3,)")