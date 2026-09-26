from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerceptionFrame:
    frame_id: int
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    timestamp: float

    def __post_init__(self):
        rgb = np.asarray(self.rgb)
        depth = np.squeeze(np.asarray(self.depth))
        intrinsics = np.asarray(self.intrinsics, dtype=float)
        transform = np.asarray(self.world_from_camera, dtype=float)

        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError("rgb must have shape (H, W, C>=3)")
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError("depth must have shape (H, W) matching rgb")
        if intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must have shape (3, 3)")
        if transform.shape != (4, 4):
            raise ValueError("world_from_camera must have shape (4, 4)")

        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "world_from_camera", transform)
