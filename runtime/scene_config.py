from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ObjectConfig:
    id: str
    prim_path: str
    size: np.ndarray | None
    graspable: bool


@dataclass(frozen=True)
class PerceptionConfig:
    camera_prim_path: str | None = None
    resolution: tuple[int, int] = (640, 480)


@dataclass(frozen=True)
class SceneConfig:
    robot: RobotConfig
    world: WorldConfig
    objects: dict[str, ObjectConfig]
    perception: PerceptionConfig


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
    world = WorldConfig(
        objects_root=w.get("objects_root", "/World/Objects"),
        provider=w.get("provider", "ground_truth"),
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

    perception = PerceptionConfig(
        camera_prim_path=p.get("camera_prim_path"),
        resolution=(
            int(resolution[0]),
            int(resolution[1]),
        ),
    )

    return SceneConfig(
        robot=robot,
        world=world,
        objects=objects,
        perception=perception,
    )