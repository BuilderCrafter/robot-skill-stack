from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import tomllib
import numpy as np

from robot_skill_stack.integrations.isaac.spatial_mapping.config import SpatialMappingConfig


@dataclass(frozen=True)
class RobotConfig:
    backend: str
    prim_path: str


@dataclass(frozen=True)
class WorldConfig:
    provider: str
    update_hz: float | None = None
    stale_object_ttl: float = 15.0

    def __post_init__(self):
        for value in (self.update_hz, self.stale_object_ttl):
            if value is not None and (type(value) not in (int, float) or
                                      not math.isfinite(value) or value <= 0):
                raise ValueError('update_hz and stale_object_ttl must be finite and > 0')


@dataclass(frozen=True)
class SceneGeometry:
    support_plane_z: float
    workspace_min: np.ndarray
    workspace_max: np.ndarray

    def __post_init__(self):
        for name in ('workspace_min', 'workspace_max'):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f'{name} must contain three finite values')
            object.__setattr__(self, name, value)
        if not math.isfinite(self.support_plane_z) or np.any(self.workspace_min >= self.workspace_max):
            raise ValueError('Scene requires a finite support plane and increasing workspace bounds')


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
        if self.camera_prim_paths and (len(self.camera_prim_paths) < 2 or
                any(not isinstance(p, str) or not p.startswith('/') or
                    any(not part.isidentifier() for part in p[1:].split('/'))
                    for p in self.camera_prim_paths) or
                len(set(self.camera_prim_paths)) != len(self.camera_prim_paths)):
            raise ValueError('multi_view requires at least two distinct absolute camera prim paths')


@dataclass(frozen=True)
class ObjectConfig:
    prim_path: str
    size: np.ndarray | None = None
    graspable: bool = True

    def __post_init__(self):
        if self.size is not None:
            size = np.asarray(self.size, dtype=float)
            if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError('Object size must contain three finite positive values')
            object.__setattr__(self, 'size', size)


@dataclass(frozen=True)
class GroundTruthConfig:
    objects_root: str | None = None
    objects: dict[str, ObjectConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class MotionConfig:
    strategy: str = 'direct'
    max_obstacles: int = 64

    def __post_init__(self):
        if not isinstance(self.strategy, str):
            raise ValueError('motion.strategy must be a string')
        if self.strategy in ('global', 'hybrid'):
            raise ValueError(f'Unsupported motion strategy {self.strategy!r}: '
                             'not implemented; select \'direct\'')
        if self.strategy not in ('direct', 'reactive'):
            raise ValueError(f'Unknown motion strategy {self.strategy!r}; '
                             'supported strategies: direct, reactive')
        if type(self.max_obstacles) is not int or not 1 <= self.max_obstacles <= 128:
            raise ValueError('motion.max_obstacles must be an integer between 1 and 128')


@dataclass(frozen=True)
class SceneConfig:
    robot: RobotConfig
    world: WorldConfig
    scene: SceneGeometry
    perception: PerceptionConfig
    ground_truth: GroundTruthConfig
    multi_view: MultiViewConfig = field(default_factory=MultiViewConfig)
    spatial_mapping: SpatialMappingConfig = field(default_factory=SpatialMappingConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)

    def __post_init__(self):
        if self.motion.strategy != 'reactive':
            return
        if not self.spatial_mapping.enabled:
            raise ValueError('Reactive motion requires enabled Spatial Mapping')
        if self.spatial_mapping.source == 'rgbd' and not self.spatial_mapping.automatic_robot_filter:
            raise ValueError('Reactive RGB-D motion requires automatic robot self-filtering')
        if self.spatial_mapping.source == 'ground_truth' and not any(
                self.robot.prim_path == path or self.robot.prim_path.startswith(path+'/')
                for path in self.spatial_mapping.excluded_prim_paths):
            raise ValueError('Reactive ground-truth mapping must exclude the robot subtree')


def load_scene_config(path: str | Path) -> SceneConfig:
    with Path(path).open('rb') as stream:
        data = tomllib.load(stream)
    unknown = set(data)-{'robot', 'world', 'scene', 'perception', 'ground_truth',
                        'multi_view', 'spatial_mapping', 'motion'}
    if unknown:
        raise ValueError(f'Unsupported configuration sections: {sorted(unknown)}')
    try:
        ground_truth = dict(data.get('ground_truth', {}))
        objects = {name: ObjectConfig(**cfg) for name, cfg in ground_truth.pop('objects', {}).items()}
        config = SceneConfig(
            RobotConfig(**data['robot']), WorldConfig(**data['world']),
            SceneGeometry(**data['scene']), PerceptionConfig(**data.get('perception', {})),
            GroundTruthConfig(objects=objects, **ground_truth),
            MultiViewConfig(**data.get('multi_view', {})),
            SpatialMappingConfig(**data.get('spatial_mapping', {})),
            MotionConfig(**data.get('motion', {})),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Invalid scene configuration: {exc}') from exc
    if config.robot.backend != 'franka':
        raise ValueError(f'Unsupported robot backend: {config.robot.backend}')
    if config.world.provider not in ('perception', 'ground_truth', 'multi_view'):
        raise ValueError(f'Unsupported observation source: {config.world.provider}')
    required = [config.robot.prim_path]
    if config.world.provider == 'perception':
        required.append(config.perception.camera_prim_path)
    elif config.world.provider == 'multi_view':
        if not config.multi_view.camera_prim_paths:
            raise ValueError('multi_view requires at least two camera prim paths')
        required.extend(config.multi_view.camera_prim_paths)
    else:
        required += [config.ground_truth.objects_root, *(o.prim_path for o in objects.values())]
    if any(not isinstance(p, str) or not p.startswith('/') for p in required):
        raise ValueError('Selected robot/source requires absolute simulator prim paths')
    return config
