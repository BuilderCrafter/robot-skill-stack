from __future__ import annotations

import numpy as np

from core.types import Pose


class PerceptionStateProvider:
    def __init__(self, camera, detector, localizer, object_sizes: dict):
        self.camera = camera
        self.detector = detector
        self.localizer = localizer
        self.object_sizes = {
            key: np.asarray(value, dtype=float)
            for key, value in object_sizes.items()
        }

    def get_object_pose(self, object_id: str) -> Pose | None:
        size = self.object_sizes.get(object_id)
        if size is None:
            return None

        detection = self.detector.detect(object_id)
        if detection is None:
            return None

        depth = self.camera.get_depth()
        if depth is None:
            return None

        depth = np.squeeze(np.asarray(depth))
        mask = detection.mask

        if mask.shape != depth.shape:
            return None

        valid = mask & np.isfinite(depth) & (depth > 0)
        ys, xs = np.nonzero(valid)

        if not xs.size:
            return None

        pixels = np.column_stack((xs, ys))
        depths = depth[ys, xs]

        points = self.localizer.pixels_to_world(
            pixels,
            depths,
            self.camera.get_world_from_camera_transform(),
        )
        points = points[np.isfinite(points).all(axis=1)]

        if not len(points):
            return None

        return Pose(
            position=self._estimate_center(points, size),
            orientation=None,
            frame="world",
        )

    @staticmethod
    def _estimate_center(points, size):
        lo = np.min(points, axis=0)
        hi = np.max(points, axis=0)

        return np.array([
            (lo[0] + hi[0]) / 2.0,
            (lo[1] + hi[1]) / 2.0,
            np.percentile(points[:, 2], 95) - size[2] / 2.0,
        ])