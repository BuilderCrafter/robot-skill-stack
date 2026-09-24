from __future__ import annotations

import numpy as np

from core.types import Pose
from world_model.observations import ObjectObservation


class PerceptionStateProvider:
    def __init__(
        self,
        camera,
        detector,
        localizer,
        *,
        object_sizes: dict | None = None,
        object_graspable: dict | None = None,
        source="isaac_semantic_perception",
    ):
        self.camera = camera
        self.detector = detector
        self.localizer = localizer
        self.source = source

        self.object_sizes = {
            key: np.asarray(value, dtype=float)
            for key, value in (object_sizes or {}).items()
            if value is not None
        }

        self.object_graspable = object_graspable or {}

    def observe(self) -> list[ObjectObservation]:
        rgb = self.camera.get_rgb()
        depth = self.camera.get_depth()

        if rgb is None or depth is None:
            return []

        rgb = np.asarray(rgb)
        depth = np.squeeze(np.asarray(depth))

        detections = self.detector.detect(rgb)
        transform = self.camera.get_world_from_camera_transform()

        observations = []

        for detection in detections:
            mask = detection.mask

            if mask.shape != depth.shape:
                continue

            valid = (
                mask
                & np.isfinite(depth)
                & (depth > 0)
            )

            ys, xs = np.nonzero(valid)

            if not xs.size:
                continue

            pixels = np.column_stack((xs, ys))
            depths = depth[ys, xs]

            points = self.localizer.pixels_to_world(
                pixels,
                depths,
                transform,
            )

            points = points[
                np.isfinite(points).all(axis=1)
            ]

            if not len(points):
                continue

            known_size = self.object_sizes.get(
                detection.object_id
            )

            position, size = self._estimate_geometry(
                points,
                known_size,
            )

            observations.append(
                ObjectObservation(
                    object_id=detection.object_id,
                    class_name=detection.class_name,
                    pose=Pose(
                        position=position,
                        orientation=None,
                        frame="world",
                    ),
                    size=size,
                    graspable=self.object_graspable.get(
                        detection.object_id
                    ),
                    visible=True,
                    confidence=detection.confidence,
                    source=self.source,
                    metadata={
                        **detection.metadata,
                        "mask_pixels": int(valid.sum()),
                    },
                )
            )

        return observations

    @staticmethod
    def _estimate_geometry(points, known_size):
        lo = np.min(points, axis=0)
        hi = np.max(points, axis=0)

        if known_size is None:
            return (
                (lo + hi) / 2.0,
                hi - lo,
            )

        size = known_size.copy()

        position = np.array([
            (lo[0] + hi[0]) / 2.0,
            (lo[1] + hi[1]) / 2.0,
            np.percentile(points[:, 2], 95)
            - size[2] / 2.0,
        ])

        return position, size