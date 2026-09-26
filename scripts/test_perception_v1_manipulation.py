from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import asyncio
import re
import sys
import traceback
from pathlib import Path

import numpy as np
from isaacsim.core.utils.stage import is_stage_loading, open_stage

from core.types import Pose
from runtime.runtime_builder import build_runtime
from scripts.perception_v1_common import write_result

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / "playground.usd"
PROFILE = ROOT / "config" / "scenes" / "playground.toml"
TARGET = np.array([0.4465, 0.25, 0.025])


def perception_profile(output_dir):
    text = PROFILE.read_text(encoding="utf-8")
    text, count = re.subn(
        r'provider\s*=\s*"(?:ground_truth|perception)"',
        'provider = "perception"',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not force [world] provider to perception")
    path = Path(output_dir) / "playground_perception_test.toml"
    path.write_text(text, encoding="utf-8")
    return path


def choose_object(world_model):
    visible = world_model.visible_objects()
    cubes = [obj for obj in visible if obj.class_name == "cube"]
    if len(cubes) == 1:
        return cubes[0]
    if len(visible) == 1:
        return visible[0]
    raise RuntimeError(
        f"Expected one discoverable manipulation object, got "
        f"{[(o.object_id, o.class_name) for o in visible]}"
    )


def main(result_path=None, artifacts_dir=None):
    metrics = {}
    output_dir = Path(artifacts_dir or (ROOT / "outputs" / "perception_v1_tmp"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Opening scene...")
    if not open_stage(str(SCENE)):
        raise RuntimeError(f"Failed to open {SCENE}")
    while is_stage_loading():
        simulation_app.update()

    print("[2] Building runtime with real geometry perception...")
    profile = perception_profile(output_dir)
    task = asyncio.ensure_future(build_runtime(profile))
    while not task.done():
        simulation_app.update()
    bundle = task.result()

    print("[3] Allowing tracker / semantic belief to settle...")
    for _ in range(180):
        bundle.world.step(render=True)

    obj = choose_object(bundle.world_model)
    actual, _ = bundle.objects["cube"].get_world_pose()
    initial_error = float(np.linalg.norm(obj.pose.position - actual))
    metrics.update(
        object_id=obj.object_id,
        semantic_class=obj.class_name,
        initial_position_error_mm=initial_error * 1000,
        initial_size=obj.size.tolist() if obj.size is not None else None,
    )
    print("Object ID:", obj.object_id)
    print("Class:", obj.class_name)
    print("Perceived:", np.round(obj.pose.position, 6))
    print("Ground truth:", np.round(actual, 6))
    print("Initial error:", f"{initial_error * 1000:.2f} mm")
    assert initial_error < 0.015

    print("\n[4] PICK using discovered persistent ID...")
    pick = bundle.runtime.execute("pick", object_id=obj.object_id)
    metrics["pick_status"] = pick.status.value
    metrics["pick_message"] = pick.message
    print(pick.status, "-", pick.message)
    if not pick.ok:
        raise RuntimeError(f"Pick failed: {pick.message}")

    print("\n[5] STABLE PLACE...")
    place = bundle.runtime.execute(
        "place",
        target=Pose(TARGET),
        mode="stable",
    )
    metrics["place_status"] = place.status.value
    metrics["place_message"] = place.message
    print(place.status, "-", place.message)
    if not place.ok:
        raise RuntimeError(f"Place failed: {place.message}")

    final_true, _ = bundle.objects["cube"].get_world_pose()
    target_error = float(np.linalg.norm(final_true - TARGET))
    metrics["true_target_error_mm"] = target_error * 1000
    metrics["held_after_place"] = bundle.world_model.held_object_id
    metrics["visible_objects_after_place"] = [
        {
            "object_id": o.object_id,
            "class_name": o.class_name,
            "visible": o.visible,
        }
        for o in bundle.world_model.objects()
    ]

    print("\n=== PERCEPTION V1 MANIPULATION ===")
    print("Persistent ID:", obj.object_id)
    print("Pick: SUCCESS")
    print("Place: SUCCESS")
    print("True target error:", f"{target_error * 1000:.2f} mm")
    print("Held:", bundle.world_model.held_object_id)
    print("PASS")
    print("===================================")
    write_result(result_path, status="PASS", metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result")
    parser.add_argument("--artifacts-dir")
    args = parser.parse_args()
    try:
        main(args.result, args.artifacts_dir)
    except Exception as exc:
        traceback.print_exc()
        write_result(
            args.result,
            status="FAIL",
            error=f"{type(exc).__name__}: {exc}",
        )
        sys.exit(1)
    finally:
        simulation_app.close()
