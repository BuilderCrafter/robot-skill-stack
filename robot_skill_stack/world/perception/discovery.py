from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_skill_stack.world.perception.frame import PerceptionFrame


@dataclass
class ObjectCandidate:
    position: np.ndarray
    size: np.ndarray
    mask: np.ndarray
    confidence: float = 1.0
    spawnable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.size = np.asarray(self.size, dtype=float)
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.position.shape != (3,):
            raise ValueError("candidate position must have shape (3,)")
        if self.size.shape != (3,):
            raise ValueError("candidate size must have shape (3,)")
        if self.mask.ndim != 2:
            raise ValueError("candidate mask must be 2D")


class DepthObjectDiscoverer:
    def __init__(
        self,
        localizer,
        *,
        support_plane_z=0.0,
        workspace_min=(0.20, -0.40, 0.0),
        workspace_max=(0.70, 0.40, 0.22),
        min_object_height=0.008,
        max_object_height=0.22,
        min_component_pixels=30,
        max_component_pixels=30000,
        candidate_max_extent=0.18,
        spawn_min_extent=0.02,
        spawn_max_extent=0.10,
        spawn_compactness_ratio=1.60,
        support_contact_tolerance=0.015,
    ):
        self.localizer = localizer
        self.support_plane_z = float(support_plane_z)
        self.workspace_min = np.asarray(workspace_min, dtype=float)
        self.workspace_max = np.asarray(workspace_max, dtype=float)
        self.min_object_height = float(min_object_height)
        self.max_object_height = float(max_object_height)
        self.min_component_pixels = int(min_component_pixels)
        self.max_component_pixels = int(max_component_pixels)
        self.candidate_max_extent = float(candidate_max_extent)
        self.spawn_min_extent = float(spawn_min_extent)
        self.spawn_max_extent = float(spawn_max_extent)
        self.spawn_compactness_ratio = float(spawn_compactness_ratio)
        self.support_contact_tolerance = float(support_contact_tolerance)

    @staticmethod
    def _components(mask):
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        visited = np.zeros_like(mask)
        components = []
        neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1))

        for y, x in np.argwhere(mask):
            if visited[y, x]:
                continue
            stack = [(int(y), int(x))]
            visited[y, x] = True
            ys, xs = [], []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            components.append((np.asarray(ys), np.asarray(xs)))
        return components

    def discover(self, frame: PerceptionFrame) -> list[ObjectCandidate]:
        depth = frame.depth
        valid = np.isfinite(depth) & (depth > 0)
        ys, xs = np.nonzero(valid)
        if not xs.size:
            return []

        pixels = np.column_stack((xs, ys))
        world_points = self.localizer.pixels_to_world(
            pixels,
            depth[ys, xs],
            frame.world_from_camera,
        )
        finite = np.isfinite(world_points).all(axis=1)
        ys, xs, world_points = ys[finite], xs[finite], world_points[finite]

        h, w = depth.shape
        world = np.full((h, w, 3), np.nan, dtype=np.float32)
        world[ys, xs] = world_points.astype(np.float32)

        p = world
        low_z = max(
            self.workspace_min[2],
            self.support_plane_z + self.min_object_height,
        )
        high_z = min(
            self.workspace_max[2],
            self.support_plane_z + self.max_object_height,
        )
        candidate_mask = (
            np.isfinite(p[..., 0])
            & (p[..., 0] >= self.workspace_min[0])
            & (p[..., 0] <= self.workspace_max[0])
            & (p[..., 1] >= self.workspace_min[1])
            & (p[..., 1] <= self.workspace_max[1])
            & (p[..., 2] >= low_z)
            & (p[..., 2] <= high_z)
        )

        candidates = []
        for cys, cxs in self._components(candidate_mask):
            n = len(cxs)
            if n < self.min_component_pixels or n > self.max_component_pixels:
                continue

            points = world[cys, cxs].astype(float)
            lo = np.percentile(points, 2, axis=0)
            hi = np.percentile(points, 98, axis=0)
            top_z = float(np.percentile(points[:, 2], 98))
            raw_z = max(float(hi[2] - lo[2]), 1e-6)
            supported = bool(
                lo[2] <= self.support_plane_z + self.support_contact_tolerance
            )

            if supported:
                size_z = max(top_z - self.support_plane_z, raw_z)
                center_z = self.support_plane_z + size_z / 2.0
            else:
                size_z = raw_z
                center_z = (lo[2] + hi[2]) / 2.0

            size = np.array([
                max(float(hi[0] - lo[0]), 1e-6),
                max(float(hi[1] - lo[1]), 1e-6),
                size_z,
            ])
            if np.max(size) > self.candidate_max_extent:
                continue

            position = np.array([
                (lo[0] + hi[0]) / 2.0,
                (lo[1] + hi[1]) / 2.0,
                center_z,
            ])
            min_extent = max(float(np.min(size)), 1e-6)
            compactness = float(np.max(size) / min_extent)
            spawnable = bool(
                np.all(size >= self.spawn_min_extent)
                and np.all(size <= self.spawn_max_extent)
                and compactness <= self.spawn_compactness_ratio
            )

            mask = np.zeros((h, w), dtype=bool)
            mask[cys, cxs] = True
            confidence = min(1.0, n / max(self.min_component_pixels * 4, 1))
            candidates.append(
                ObjectCandidate(
                    position=position,
                    size=size,
                    mask=mask,
                    confidence=confidence,
                    spawnable=spawnable,
                    metadata={
                        "component_pixels": n,
                        "compactness": compactness,
                        "supported": supported,
                    },
                )
            )

        return candidates
