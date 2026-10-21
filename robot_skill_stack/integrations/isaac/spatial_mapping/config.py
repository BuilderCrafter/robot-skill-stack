"""Opt-in mapping settings, independent of the semantic observation provider."""
from dataclasses import dataclass
import math

from robot_skill_stack.world.spatial_mapping.occupancy import GridSpec


@dataclass(frozen=True)
class SpatialMappingConfig:
    enabled: bool = False
    source: str = 'ground_truth'
    bounds_min: tuple = (.10, -.45, -.05)
    bounds_max: tuple = (.85, .45, .65)
    voxel_size: float = .01
    max_voxels: int = 2_000_000
    camera_prim_paths: tuple[str, ...] = ()
    stride: int = 8
    max_rays: int = 1024
    max_voxel_visits: int = 500_000
    max_depth: float = 5.
    max_skew: float = .025
    max_frame_age: float = 1.
    stale_after: float = 1.
    evidence_ttl: float = 5.
    update_hz: float | None = None
    excluded_prim_paths: tuple[str, ...] = ('/Franka',)
    max_candidate_voxels: int = 8_000_000
    automatic_robot_filter: bool = False
    robot_depth_tolerance: float = .005
    robot_pose_tolerance: float = .01

    def __post_init__(self):
        if type(self.enabled) is not bool or self.source not in ('ground_truth', 'rgbd'):
            raise ValueError('spatial_mapping needs enabled=true/false and source=ground_truth or rgbd')
        if type(self.automatic_robot_filter) is not bool:
            raise ValueError('spatial_mapping automatic_robot_filter must be true/false')
        spec = self.grid_spec()
        object.__setattr__(self, 'bounds_min', tuple(spec.bounds_min))
        object.__setattr__(self, 'bounds_max', tuple(spec.bounds_max))
        for name in ('stride', 'max_rays', 'max_voxel_visits', 'max_candidate_voxels'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'spatial_mapping {name} must be a positive integer')
        for name in ('max_depth', 'max_skew', 'max_frame_age', 'stale_after', 'evidence_ttl',
                     'update_hz', 'robot_depth_tolerance', 'robot_pose_tolerance'):
            value = getattr(self, name)
            if value is None and name == 'update_hz':
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'spatial_mapping {name} must be finite and positive')
        for name in ('camera_prim_paths', 'excluded_prim_paths'):
            values = tuple(getattr(self, name))
            if (any(not isinstance(path, str) or not path.startswith('/') or
                    any(not part.isidentifier() for part in path[1:].split('/')) for path in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f'spatial_mapping {name} requires distinct absolute prim paths')
            object.__setattr__(self, name, values)

    def grid_spec(self):
        return GridSpec(self.bounds_min, self.bounds_max, self.voxel_size, self.max_voxels)
