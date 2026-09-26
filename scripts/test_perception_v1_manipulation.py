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


def object_snapshot(obj):
    return {
        "object_id": obj.object_id,
        "class_name": obj.class_name,
        "visible": obj.visible,
        "position": None if obj.pose is None else obj.pose.position.tolist(),
        "size": None if obj.size is None else obj.size.tolist(),
        "metadata": dict(obj.metadata),
    }


def hints_snapshot(world_model):
    return [
        {
            "object_id": hint.object_id,
            "expected_position": hint.expected_position.tolist(),
            "max_distance": hint.max_distance,
            "reason": hint.reason,
        }
        for hint in world_model.association_hints()
    ]


def main(result_path=None, artifacts_dir=None):
    metrics = {}
    output_dir = Path(
        artifacts_dir
        or (ROOT / "outputs" / "perception_v1_tmp")
    )
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
    original_id = obj.object_id
    actual, _ = bundle.objects["cube"].get_world_pose()
    initial_true = np.asarray(actual, dtype=float).copy()
    initial_error = float(
        np.linalg.norm(obj.pose.position - actual)
    )
    metrics.update(
        object_id=original_id,
        semantic_class=obj.class_name,
        initial_position_error_mm=initial_error * 1000,
        initial_size=(
            obj.size.tolist()
            if obj.size is not None
            else None
        ),
        initial_object=object_snapshot(obj),
    )

    print("Object ID:", original_id)
    print("Class:", obj.class_name)
    print("Perceived:", np.round(obj.pose.position, 6))
    print("Ground truth:", np.round(actual, 6))
    print("Initial error:", f"{initial_error * 1000:.2f} mm")
    assert initial_error < 0.015

    print("\n[4] PICK using discovered persistent ID...")
    pick = bundle.runtime.execute(
        "pick",
        object_id=original_id,
    )
    true_after_pick, _ = bundle.objects["cube"].get_world_pose()
    true_after_pick = np.asarray(true_after_pick, dtype=float)
    metrics.update(
        pick_status=pick.status.value,
        pick_message=pick.message,
        pick_details=pick.details,
        true_lift_after_pick_mm=float(
            (true_after_pick[2] - initial_true[2]) * 1000
        ),
        true_cube_after_pick=true_after_pick.tolist(),
        object_after_pick=object_snapshot(obj),
        held_after_pick=bundle.world_model.held_object_id,
        hints_after_pick=hints_snapshot(bundle.world_model),
    )

    print(pick.status, "-", pick.message)
    print(
        "True lift:",
        f"{metrics['true_lift_after_pick_mm']:.2f} mm",
    )
    print("Visible after pick:", obj.visible)
    print("Held:", bundle.world_model.held_object_id)

    if not pick.ok:
        write_result(
            result_path,
            status="FAIL",
            metrics=metrics,
            error=f"Pick failed: {pick.message}",
        )
        raise RuntimeError(f"Pick failed: {pick.message}")

    print("\n[5] STABLE PLACE with action-aware reacquisition...")
    place = bundle.runtime.execute(
        "place",
        target=Pose(TARGET),
        mode="stable",
    )
    final_true, _ = bundle.objects["cube"].get_world_pose()
    final_true = np.asarray(final_true, dtype=float)
    target_error = float(
        np.linalg.norm(final_true - TARGET)
    )
    obj_after = bundle.world_model.get(original_id)
    visible_ids = [
        o.object_id
        for o in bundle.world_model.visible_objects()
    ]
    metrics.update(
        place_status=place.status.value,
        place_message=place.message,
        place_details=place.details,
        true_target_error_mm=target_error * 1000,
        true_cube_after_place=final_true.tolist(),
        held_after_place=bundle.world_model.held_object_id,
        original_object_after_place=(
            None
            if obj_after is None
            else object_snapshot(obj_after)
        ),
        visible_ids_after_place=visible_ids,
        visible_objects_after_place=[
            object_snapshot(o)
            for o in bundle.world_model.objects()
        ],
        active_hints_after_place=hints_snapshot(
            bundle.world_model
        ),
    )

    print(place.status, "-", place.message)
    print("True target error:", f"{target_error * 1000:.2f} mm")
    print("Visible IDs:", visible_ids)
    print(
        "Original ID visible:",
        None if obj_after is None else obj_after.visible,
    )
    print(
        "Hint matched:",
        None
        if obj_after is None
        else obj_after.metadata.get("association_hint_match"),
    )

    if not place.ok:
        write_result(
            result_path,
            status="FAIL",
            metrics=metrics,
            error=f"Place failed: {place.message}",
        )
        raise RuntimeError(f"Place failed: {place.message}")

    if obj_after is None or not obj_after.visible:
        write_result(
            result_path,
            status="FAIL",
            metrics=metrics,
            error="Original object ID was not reacquired after place",
        )
        raise RuntimeError(
            "Original object ID was not reacquired after place"
        )

    if original_id not in visible_ids:
        raise RuntimeError(
            f"Persistent ID {original_id} is not visible after place"
        )

    duplicates = [
        o.object_id
        for o in bundle.world_model.visible_objects()
        if o.object_id != original_id
        and o.class_name == obj_after.class_name
    ]
    metrics["duplicate_visible_same_class_ids"] = duplicates
    if duplicates:
        raise RuntimeError(
            f"Unexpected duplicate visible tracks after place: {duplicates}"
        )

    if bundle.world_model.association_hints():
        raise RuntimeError(
            "Placement association hint was not consumed"
        )

    print("\n=== PERCEPTION V1 MANIPULATION ===")
    print("Persistent ID:", original_id)
    print("Pick: SUCCESS")
    print("Place: SUCCESS")
    print("Reacquired same ID: YES")
    print("Duplicate track: NO")
    print(
        "True target error:",
        f"{target_error * 1000:.2f} mm",
    )
    print("Held:", bundle.world_model.held_object_id)
    print("PASS")
    print("===================================")
    write_result(
        result_path,
        status="PASS",
        metrics=metrics,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result")
    parser.add_argument("--artifacts-dir")
    args = parser.parse_args()
    try:
        main(args.result, args.artifacts_dir)
    except Exception as exc:
        traceback.print_exc()
        if not args.result or not Path(args.result).exists():
            write_result(
                args.result,
                status="FAIL",
                error=f"{type(exc).__name__}: {exc}",
            )
        sys.exit(1)
    finally:
        simulation_app.close()
