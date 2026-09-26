from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np
from isaacsim.core.utils.stage import is_stage_loading, open_stage

from runtime.runtime_builder import build_runtime
from scripts.isaac_async import run_kit_coroutine
from scripts.perception_v1_common import save_pgm, save_ppm, write_result

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / "playground.usd"
PROFILE = ROOT / "config" / "scenes" / "playground.toml"
LIFT_THRESHOLD = 0.03


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
    path = Path(output_dir) / "playground_perception_diagnostic.toml"
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
        "Expected one visible manipulation object, got "
        f"{[(o.object_id, o.class_name) for o in visible]}"
    )


def arr(value):
    if value is None:
        return None
    return np.asarray(value, dtype=float).tolist()


def track_snapshot(provider, object_id):
    tracker = getattr(provider, "tracker", None)
    track = None if tracker is None else tracker.get(object_id)
    if track is None:
        return None
    return {
        "object_id": track.object_id,
        "position": arr(track.position),
        "size": arr(track.size),
        "visible": bool(track.visible),
        "hits": int(track.hits),
        "misses": int(track.misses),
        "frame_id": track.frame_id,
        "mask_pixels": 0 if track.mask is None else int(track.mask.sum()),
        "class_name": track.class_name,
        "class_confidence": float(track.class_confidence),
        "semantic_samples": int(track.belief.samples),
    }


def world_snapshot(bundle, object_id):
    obj = bundle.world_model.get(object_id)
    if obj is None:
        return None
    return {
        "object_id": obj.object_id,
        "position": None if obj.pose is None else arr(obj.pose.position),
        "orientation": None if obj.pose is None else arr(obj.pose.orientation),
        "size": arr(obj.size),
        "visible": bool(obj.visible),
        "class_name": obj.class_name,
        "confidence": obj.confidence,
        "source": obj.source,
        "last_seen": obj.last_seen,
        "metadata": dict(obj.metadata),
    }


def robot_snapshot(bundle):
    ee = bundle.backend.get_end_effector_pose()
    joints = bundle.backend.robot.gripper.get_joint_positions()
    return {
        "ee_position": arr(ee.position),
        "ee_orientation": arr(ee.orientation),
        "gripper_joint_positions": arr(joints),
    }


def true_cube_snapshot(bundle):
    position, orientation = bundle.objects["cube"].get_world_pose()
    return {
        "position": arr(position),
        "orientation": arr(orientation),
    }


def save_frame_artifacts(provider, object_id, out_dir, prefix):
    geometry = getattr(provider, "geometry", None)
    frame = None if geometry is None else geometry.latest_frame
    if frame is None:
        return {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    rgb_path = out_dir / f"{prefix}_rgb.ppm"
    save_ppm(rgb_path, frame.rgb)
    saved["rgb"] = rgb_path.name

    depth = np.asarray(frame.depth)
    valid = np.isfinite(depth) & (depth > 0)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        scaled = np.zeros_like(depth, dtype=np.uint8)
        normalized = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0, 1)
        scaled[valid] = ((1.0 - normalized[valid]) * 255).astype(np.uint8)
        depth_path = out_dir / f"{prefix}_depth.pgm"
        save_pgm(depth_path, scaled)
        saved["depth"] = depth_path.name

    mask = provider.get_mask(object_id) if hasattr(provider, "get_mask") else None
    if mask is not None:
        mask_path = out_dir / f"{prefix}_mask.pgm"
        save_pgm(mask_path, np.asarray(mask, dtype=np.uint8) * 255)
        saved["mask"] = mask_path.name

    return saved


def capture(bundle, object_id, artifacts_dir, label):
    return {
        "label": label,
        "world_model": world_snapshot(bundle, object_id),
        "track": track_snapshot(bundle.state_provider, object_id),
        "robot": robot_snapshot(bundle),
        "true_cube": true_cube_snapshot(bundle),
        "artifacts": save_frame_artifacts(
            bundle.state_provider,
            object_id,
            artifacts_dir,
            label,
        ),
    }


def diagnose(initial, immediate, pick_ok):
    initial_true = np.asarray(initial["true_cube"]["position"], dtype=float)
    final_true = np.asarray(immediate["true_cube"]["position"], dtype=float)
    true_lift = float(final_true[2] - initial_true[2])

    initial_cached = initial["world_model"]["position"]
    final_cached = immediate["world_model"]["position"]
    cached_lift = None
    if initial_cached is not None and final_cached is not None:
        cached_lift = float(final_cached[2] - initial_cached[2])

    if pick_ok:
        verdict = "PICK_REPORTED_SUCCESS"
    elif true_lift >= LIFT_THRESHOLD and (cached_lift is None or cached_lift < LIFT_THRESHOLD):
        verdict = "PERCEPTION_FALSE_NEGATIVE"
    elif true_lift >= LIFT_THRESHOLD:
        verdict = "PICK_VERIFICATION_MISMATCH"
    else:
        verdict = "PHYSICAL_GRASP_OR_MOTION_FAILURE"

    return {
        "verdict": verdict,
        "true_lift_m": true_lift,
        "cached_lift_m": cached_lift,
        "lift_threshold_m": LIFT_THRESHOLD,
    }


def main(result_path=None, artifacts_dir=None):
    metrics = {"snapshots": {}}
    output_dir = Path(artifacts_dir or (ROOT / "outputs" / "perception_pick_diagnostic_tmp"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Opening scene...")
    if not open_stage(str(SCENE)):
        raise RuntimeError(f"Failed to open {SCENE}")
    while is_stage_loading():
        simulation_app.update()

    print("[2] Building perception runtime...")
    profile = perception_profile(output_dir)
    bundle = run_kit_coroutine(build_runtime(profile), simulation_app)

    print("[3] Calibrating gripper feedback with no object in the gripper...")
    opened = bundle.backend.open_gripper()
    open_joints = bundle.backend.robot.gripper.get_joint_positions()
    closed_empty = bundle.backend.close_gripper()
    empty_closed_joints = bundle.backend.robot.gripper.get_joint_positions()
    reopened = bundle.backend.open_gripper()
    reopened_joints = bundle.backend.robot.gripper.get_joint_positions()
    metrics["gripper_calibration"] = {
        "open_ok": bool(opened.ok),
        "open_joint_positions": arr(open_joints),
        "empty_close_ok": bool(closed_empty.ok),
        "empty_closed_joint_positions": arr(empty_closed_joints),
        "reopen_ok": bool(reopened.ok),
        "reopened_joint_positions": arr(reopened_joints),
    }
    print("Open joints:", metrics["gripper_calibration"]["open_joint_positions"])
    print("Empty-closed joints:", metrics["gripper_calibration"]["empty_closed_joint_positions"])
    print("Reopened joints:", metrics["gripper_calibration"]["reopened_joint_positions"])

    print("[4] Letting perception settle...")
    for _ in range(180):
        bundle.world.step(render=True)

    obj = choose_object(bundle.world_model)
    metrics["object_id"] = obj.object_id
    metrics["semantic_class"] = obj.class_name
    metrics["initial"] = capture(bundle, obj.object_id, output_dir, "pre_pick")
    metrics["snapshots"]["pre_pick"] = metrics["initial"]

    actual = np.asarray(metrics["initial"]["true_cube"]["position"], dtype=float)
    perceived = np.asarray(metrics["initial"]["world_model"]["position"], dtype=float)
    metrics["initial_position_error_mm"] = float(np.linalg.norm(perceived - actual) * 1000)

    print("Object ID:", obj.object_id)
    print("Class:", obj.class_name)
    print("Perceived:", np.round(perceived, 6))
    print("Ground truth:", np.round(actual, 6))
    print("Initial error:", f"{metrics['initial_position_error_mm']:.2f} mm")
    print("Track before pick:", metrics["initial"]["track"])
    print("Gripper before pick:", metrics["initial"]["robot"]["gripper_joint_positions"])

    print("\n[5] Running PICK...")
    pick = bundle.runtime.execute("pick", object_id=obj.object_id)
    metrics["pick"] = {
        "ok": bool(pick.ok),
        "status": pick.status.value,
        "message": pick.message,
        "failure_code": None if pick.failure_code is None else pick.failure_code.value,
        "details": pick.details,
    }
    print(pick.status, "-", pick.message)
    print("Pick details:", pick.details)

    immediate = capture(bundle, obj.object_id, output_dir, "post_pick_immediate")
    metrics["snapshots"]["post_pick_immediate"] = immediate
    metrics["diagnosis"] = diagnose(metrics["initial"], immediate, pick.ok)

    print("\n[6] Immediate post-pick evidence...")
    print("True cube:", np.round(immediate["true_cube"]["position"], 6))
    print("Cached cube:", immediate["world_model"]["position"])
    print("Track:", immediate["track"])
    print("Gripper:", immediate["robot"]["gripper_joint_positions"])
    print("Diagnosis:", metrics["diagnosis"])

    for label, steps in (("post_pick_0p2s", 12), ("post_pick_0p5s", 18), ("post_pick_1p0s", 30)):
        for _ in range(steps):
            bundle.world.step(render=True)
        snap = capture(bundle, obj.object_id, output_dir, label)
        metrics["snapshots"][label] = snap
        print(f"\n[{label}]")
        print("True cube:", np.round(snap["true_cube"]["position"], 6))
        print("Cached cube:", None if snap["world_model"] is None else snap["world_model"]["position"])
        print("Track:", snap["track"])
        print("Gripper:", snap["robot"]["gripper_joint_positions"])

    print("\n=== PICK DIAGNOSTIC ===")
    print("Pick reported:", "SUCCESS" if pick.ok else "FAIL")
    print("Diagnosis:", metrics["diagnosis"]["verdict"])
    print("True lift:", f"{metrics['diagnosis']['true_lift_m'] * 1000:.2f} mm")
    cached = metrics["diagnosis"]["cached_lift_m"]
    print("Cached lift:", "unknown" if cached is None else f"{cached * 1000:.2f} mm")
    print("Result archive should be used for the production fix.")
    print("=======================")

    write_result(result_path, status="PASS", metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result")
    parser.add_argument("--artifacts-dir")
    args = parser.parse_args()
    metrics_on_failure = {}
    try:
        main(args.result, args.artifacts_dir)
    except Exception as exc:
        traceback.print_exc()
        write_result(
            args.result,
            status="FAIL",
            metrics=metrics_on_failure,
            error=f"{type(exc).__name__}: {exc}",
        )
        sys.exit(1)
    finally:
        simulation_app.close()
