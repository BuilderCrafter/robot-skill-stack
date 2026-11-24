"""Canonical camera and numerical settings for semantic perception."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PerceptionConfig:
    camera_prim_path: str | None = None
    resolution: tuple[int, int] = (640, 480)

    def __post_init__(self):
        if len(self.resolution) != 2 or any(type(v) is not int or v <= 0 for v in self.resolution):
            raise ValueError('resolution must contain two positive integers')
        object.__setattr__(self, 'resolution', tuple(self.resolution))


@dataclass(frozen=True)
class MultiViewConfig:
    camera_prim_paths: tuple[str, ...] = ()
    resolution: tuple[int, int] = (640, 480)
    max_skew: float = .025
    voxel_size: float = .002
    stride: int = 2

    def __post_init__(self):
        object.__setattr__(self, 'camera_prim_paths', tuple(self.camera_prim_paths))
        object.__setattr__(self, 'resolution', PerceptionConfig(resolution=self.resolution).resolution)
        for key in ('max_skew', 'voxel_size'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if type(self.stride) is not int or not 1 <= self.stride <= 8:
            raise ValueError('multi_view stride must be an integer between 1 and 8')
        if (any(not isinstance(p, str) or not p.startswith('/') or
                any(not part.isidentifier() for part in p[1:].split('/'))
                for p in self.camera_prim_paths) or
                len(set(self.camera_prim_paths)) != len(self.camera_prim_paths)):
            raise ValueError('multi_view requires distinct absolute camera prim paths')
