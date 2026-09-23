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
class ObjectConfig:
    id: str
    prim_path: str
    size: np.ndarray | None
    graspable: bool


@dataclass(frozen=True)
class PerceptionConfig:
    provider: str = "ground_truth"
    camera_prim_path: str | None = None
    resolution: tuple[int, int] = (640, 480)


@dataclass(frozen=True)
class SceneConfig:
    robot: RobotConfig
    objects: dict[str, ObjectConfig]
    perception: PerceptionConfig


def load_scene_config(path: str | Path) -> SceneConfig:
    with Path(path).open("rb") as f:
        data = tomllib.load(f)

    robot_data = data["robot"]
    robot = RobotConfig(
        id=robot_data["id"],
        type=robot_data["type"],
        prim_path=robot_data["prim_path"],
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
        provider=p.get("provider", "ground_truth"),
        camera_prim_path=p.get("camera_prim_path"),
        resolution=(int(resolution[0]), int(resolution[1])),
    )

    return SceneConfig(
        robot=robot,
        objects=objects,
        perception=perception,
    )