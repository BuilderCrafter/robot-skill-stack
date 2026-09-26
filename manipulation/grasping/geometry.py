from __future__ import annotations

from typing import Protocol

import numpy as np


class ObjectGeometryProvider(Protocol):
    """Lazy geometry source for future geometry/ML grasp planners."""

    def get_point_cloud(self, object_id: str) -> np.ndarray | None:
        ...
