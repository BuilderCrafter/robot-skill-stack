from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import tomllib
import numpy as np


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
            if value is not None and (not isinstance(value, (int, float)) or
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
        if len(self.resolution) != 2 or any(not isinstance(v, int) or v <= 0 for v in self.resolution):
            raise ValueError('resolution must contain two positive integers')
        object.__setattr__(self, 'resolution', tuple(self.resolution))


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
class SceneConfig:
    robot: RobotConfig
    world: WorldConfig
    scene: SceneGeometry
    perception: PerceptionConfig
    ground_truth: GroundTruthConfig


def load_scene_config(path: str | Path) -> SceneConfig:
    with Path(path).open('rb') as stream:
        data = tomllib.load(stream)
    unknown = set(data)-{'robot', 'world', 'scene', 'perception', 'ground_truth'}
    if unknown:
        raise ValueError(f'Unsupported configuration sections: {sorted(unknown)}')
    try:
        ground_truth = dict(data.get('ground_truth', {}))
        objects = {name: ObjectConfig(**cfg) for name, cfg in ground_truth.pop('objects', {}).items()}
        config = SceneConfig(
            RobotConfig(**data['robot']), WorldConfig(**data['world']),
            SceneGeometry(**data['scene']), PerceptionConfig(**data.get('perception', {})),
            GroundTruthConfig(objects=objects, **ground_truth),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Invalid scene configuration: {exc}') from exc
    if config.robot.backend != 'franka':
        raise ValueError(f'Unsupported robot backend: {config.robot.backend}')
    if config.world.provider not in ('perception', 'ground_truth'):
        raise ValueError(f'Unsupported observation source: {config.world.provider}')
    required = [config.robot.prim_path]
    if config.world.provider == 'perception':
        required.append(config.perception.camera_prim_path)
    else:
        required += [config.ground_truth.objects_root, *(o.prim_path for o in objects.values())]
    if any(not isinstance(p, str) or not p.startswith('/') for p in required):
        raise ValueError('Selected robot/source requires absolute simulator prim paths')
    return config
