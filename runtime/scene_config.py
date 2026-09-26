from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

import numpy as np


@dataclass(frozen=True)
class RobotConfig:
    id: str
    type: str
    prim_path: str


@dataclass(frozen=True)
class WorldConfig:
    objects_root: str = "/World/Objects"
    provider: str = "ground_truth"
    update_hz: float | None = None


@dataclass(frozen=True)
class ObjectConfig:
    id: str
    prim_path: str
    size: np.ndarray | None
    graspable: bool


@dataclass(frozen=True)
class DiscoveryConfig:
    support_plane_z: float = 0.0
    workspace_min: np.ndarray = field(
        default_factory=lambda: np.array([0.20, -0.40, 0.0], dtype=float)
    )
    workspace_max: np.ndarray = field(
        default_factory=lambda: np.array([0.70, 0.40, 0.22], dtype=float)
    )
    min_object_height: float = 0.008
    max_object_height: float = 0.22
    min_component_pixels: int = 30
    max_component_pixels: int = 30000
    candidate_max_extent: float = 0.18
    spawn_min_extent: float = 0.02
    spawn_max_extent: float = 0.10
    spawn_compactness_ratio: float = 1.60
    support_contact_tolerance: float = 0.015


@dataclass(frozen=True)
class TrackingConfig:
    max_distance: float = 0.18
    max_size_ratio: float = 1.50
    max_misses: int = 30
    occlusion_distance: float = 0.12


@dataclass(frozen=True)
class SemanticConfig:
    window_size: int = 7
    min_samples: int = 3
    assignment_threshold: float = 0.70
    cube_ratio_max: float = 1.35
    cube_min_size: float = 0.025
    cube_max_size: float = 0.10


@dataclass(frozen=True)
class PerceptionConfig:
    camera_prim_path: str | None = None
    resolution: tuple[int, int] = (640, 480)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    semantics: SemanticConfig = field(default_factory=SemanticConfig)


@dataclass(frozen=True)
class SceneConfig:
    robot: RobotConfig
    world: WorldConfig
    objects: dict[str, ObjectConfig]
    perception: PerceptionConfig


def _vec3(data, key, default):
    value = np.asarray(data.get(key, default), dtype=float)
    if value.shape != (3,):
        raise ValueError(f"{key} must contain exactly three values")
    return value


def load_scene_config(path: str | Path) -> SceneConfig:
    with Path(path).open("rb") as f:
        data = tomllib.load(f)

    r = data["robot"]
    robot = RobotConfig(
        id=r["id"],
        type=r["type"],
        prim_path=r["prim_path"],
    )

    w = data.get("world", {})
    update_hz = w.get("update_hz")
    world = WorldConfig(
        objects_root=w.get("objects_root", "/World/Objects"),
        provider=w.get("provider", "ground_truth"),
        update_hz=None if update_hz is None else float(update_hz),
    )

    objects = {}
    for object_id, cfg in data.get("objects", {}).items():
        size = cfg.get("size")
        objects[object_id] = ObjectConfig(
            id=object_id,
            prim_path=cfg["prim_path"],
            size=None if size is None else np.asarray(size, dtype=float),
            graspable=cfg.get("graspable", True),
        )

    p = data.get("perception", {})
    resolution = p.get("resolution", [640, 480])
    d = p.get("discovery", {})
    t = p.get("tracking", {})
    s = p.get("semantics", {})

    perception = PerceptionConfig(
        camera_prim_path=p.get("camera_prim_path"),
        resolution=(int(resolution[0]), int(resolution[1])),
        discovery=DiscoveryConfig(
            support_plane_z=float(d.get("support_plane_z", 0.0)),
            workspace_min=_vec3(d, "workspace_min", [0.20, -0.40, 0.0]),
            workspace_max=_vec3(d, "workspace_max", [0.70, 0.40, 0.22]),
            min_object_height=float(d.get("min_object_height", 0.008)),
            max_object_height=float(d.get("max_object_height", 0.22)),
            min_component_pixels=int(d.get("min_component_pixels", 30)),
            max_component_pixels=int(d.get("max_component_pixels", 30000)),
            candidate_max_extent=float(d.get("candidate_max_extent", 0.18)),
            spawn_min_extent=float(d.get("spawn_min_extent", 0.02)),
            spawn_max_extent=float(d.get("spawn_max_extent", 0.10)),
            spawn_compactness_ratio=float(d.get("spawn_compactness_ratio", 1.60)),
            support_contact_tolerance=float(d.get("support_contact_tolerance", 0.015)),
        ),
        tracking=TrackingConfig(
            max_distance=float(t.get("max_distance", 0.18)),
            max_size_ratio=float(t.get("max_size_ratio", 1.50)),
            max_misses=int(t.get("max_misses", 30)),
            occlusion_distance=float(t.get("occlusion_distance", 0.12)),
        ),
        semantics=SemanticConfig(
            window_size=int(s.get("window_size", 7)),
            min_samples=int(s.get("min_samples", 3)),
            assignment_threshold=float(s.get("assignment_threshold", 0.70)),
            cube_ratio_max=float(s.get("cube_ratio_max", 1.35)),
            cube_min_size=float(s.get("cube_min_size", 0.025)),
            cube_max_size=float(s.get("cube_max_size", 0.10)),
        ),
    )

    return SceneConfig(
        robot=robot,
        world=world,
        objects=objects,
        perception=perception,
    )
