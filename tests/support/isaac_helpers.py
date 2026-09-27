from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from isaacsim.core.utils.stage import is_stage_loading, open_stage

from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
from tests.support.isaac_async import run_kit_coroutine

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "scenes" / "playground.usd"
PROFILE = ROOT / "config" / "scenes" / "playground.toml"


def provider_profile(output_dir, provider, name=None):
    if provider not in ("ground_truth", "perception"):
        raise ValueError(f"Unsupported provider: {provider}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = PROFILE.read_text(encoding="utf-8")
    text, count = re.subn(
        r'provider\s*=\s*"(?:ground_truth|perception)"',
        f'provider = "{provider}"',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not replace [world] provider")
    path = output_dir / (name or f"{provider}_test.toml")
    path.write_text(text, encoding="utf-8")
    return path


def perception_profile(output_dir, name="phase_a_perception.toml"):
    return provider_profile(output_dir, "perception", name)


def open_playground(simulation_app):
    if not open_stage(str(SCENE)):
        raise RuntimeError(f"Failed to open {SCENE}")
    while is_stage_loading():
        simulation_app.update()


def build_runtime_with_provider(simulation_app, output_dir, provider):
    profile = provider_profile(output_dir, provider)
    return run_kit_coroutine(build_runtime(profile), simulation_app)


def build_perception_runtime(simulation_app, output_dir):
    return build_runtime_with_provider(
        simulation_app,
        output_dir,
        "perception",
    )


def build_ground_truth_runtime(simulation_app, output_dir):
    return build_runtime_with_provider(
        simulation_app,
        output_dir,
        "ground_truth",
    )


def step(world, count):
    for _ in range(int(count)):
        world.step(render=True)


def settle(world, steps=72):
    step(world, steps)


def choose_object(model):
    visible = model.visible_objects()
    cubes = [obj for obj in visible if obj.class_name == "cube"]
    if len(cubes) == 1:
        return cubes[0]
    if len(visible) == 1:
        return visible[0]
    raise RuntimeError(
        "Expected one visible object, got "
        f"{[(o.object_id, o.class_name) for o in visible]}"
    )


def nearest_object(objects, position):
    position = np.asarray(position, dtype=float)
    visible = [obj for obj in objects if obj.visible and obj.pose is not None]
    if not visible:
        raise RuntimeError("No visible object available")
    return min(
        visible,
        key=lambda obj: np.linalg.norm(obj.pose.position - position),
    )


def set_cube_pose(cube, position, world, settle_steps=48):
    position = np.asarray(position, dtype=float)
    _, orientation = cube.get_world_pose()
    cube.set_world_pose(
        position=position,
        orientation=orientation,
    )
    step(world, settle_steps)
    return np.asarray(cube.get_world_pose()[0], dtype=float)


def object_snapshot(obj):
    return {
        "object_id": obj.object_id,
        "class_name": obj.class_name,
        "visible": bool(obj.visible),
        "position": (
            None
            if obj.pose is None
            else obj.pose.position.tolist()
        ),
        "size": (
            None if obj.size is None else obj.size.tolist()
        ),
        "metadata": dict(obj.metadata),
    }
