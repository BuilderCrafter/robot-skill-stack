from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

from core.types import Pose
from scripts.perception_v1_common import write_result
from scripts.phase_a_common import (
    build_perception_runtime,
    choose_object,
    object_snapshot,
    open_playground,
    set_cube_pose,
    settle,
)

PERCEPTION_POSITIONS = [
    [0.40, -0.12, 0.025],
    [0.44, -0.12, 0.025],
    [0.48, -0.12, 0.025],
    [0.48, -0.04, 0.025],
    [0.44, -0.04, 0.025],
    [0.40, -0.04, 0.025],
    [0.40,  0.04, 0.025],
    [0.44,  0.04, 0.025],
    [0.48,  0.04, 0.025],
    [0.48,  0.12, 0.025],
    [0.44,  0.12, 0.025],
    [0.40,  0.12, 0.025],
]

MANIPULATION_TRIALS = [
    ([0.40,  0.12, 0.025], [0.44,  0.24, 0.025]),
    ([0.48,  0.10, 0.025], [0.48, -0.08, 0.025]),
    ([0.42, -0.10, 0.025], [0.42,  0.16, 0.025]),
    ([0.46,  0.02, 0.025], [0.46, -0.16, 0.025]),
]


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def main(result_path=None, artifacts_dir=None):
    metrics = {
        "perception_trials": [],
        "manipulation_trials": [],
    }
    out = Path(
        artifacts_dir
        or Path("outputs") / "phase_a_benchmark_tmp"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("[1] Opening playground...")
    open_playground(simulation_app)

    print("[2] Building perception runtime...")
    bundle = build_perception_runtime(simulation_app, out)
    settle(bundle.world, 180)

    cube = bundle.objects["cube"]
    obj = choose_object(bundle.world_model)
    object_id = obj.object_id

    print("[3] Deterministic workspace perception benchmark...")
    position_errors = []
    size_max_errors = []

    for index, target in enumerate(PERCEPTION_POSITIONS, 1):
        truth = set_cube_pose(
            cube,
            target,
            bundle.world,
            settle_steps=48,
        )
        tracked = bundle.world_model.get(object_id)
        if tracked is None or not tracked.visible or tracked.pose is None:
            raise RuntimeError(
                f"Lost persistent object {object_id} at benchmark pose {index}"
            )

        pos_error = float(
            np.linalg.norm(tracked.pose.position - truth)
        )
        size_error = (
            None
            if tracked.size is None
            else np.abs(
                tracked.size - np.array([0.05, 0.05, 0.05])
            )
        )

        position_errors.append(pos_error * 1000)
        size_max_errors.append(
            float(np.max(size_error) * 1000)
            if size_error is not None
            else float("inf")
        )

        row = {
            "index": index,
            "target": list(target),
            "truth": truth.tolist(),
            "perceived": tracked.pose.position.tolist(),
            "position_error_mm": pos_error * 1000,
            "size_error_mm": (
                None
                if size_error is None
                else (size_error * 1000).tolist()
            ),
            "object_id": tracked.object_id,
        }
        metrics["perception_trials"].append(row)
        print(
            f"  pose {index:02d}: "
            f"pos={row['position_error_mm']:.2f} mm "
            f"size_max={size_max_errors[-1]:.2f} mm "
            f"id={tracked.object_id}"
        )

    metrics["perception_position_error_mm"] = stats(
        position_errors
    )
    metrics["perception_size_max_error_mm"] = stats(
        size_max_errors
    )
    metrics["persistent_id"] = object_id

    if metrics["perception_position_error_mm"]["max"] >= 15.0:
        raise AssertionError(
            "Perception benchmark exceeded 15 mm max position error"
        )
    if metrics["perception_size_max_error_mm"]["max"] >= 12.0:
        raise AssertionError(
            "Perception benchmark exceeded 12 mm max size error"
        )

    print("[4] Four full perception-driven Pick + Place trials...")
    successes = 0
    target_errors = []

    for index, (start, target) in enumerate(
        MANIPULATION_TRIALS,
        1,
    ):
        home = bundle.runtime.execute("home")
        if not home.ok:
            raise RuntimeError(
                f"Home failed before trial {index}: {home.message}"
            )

        truth_start = set_cube_pose(
            cube,
            start,
            bundle.world,
            settle_steps=60,
        )
        tracked = bundle.world_model.get(object_id)
        if tracked is None or not tracked.visible:
            raise RuntimeError(
                f"Object not visible before manipulation trial {index}"
            )

        pick = bundle.runtime.execute(
            "pick",
            object_id=object_id,
        )
        place = None
        if pick.ok:
            place = bundle.runtime.execute(
                "place",
                target=Pose(target),
                mode="stable",
            )

        final_true = np.asarray(
            cube.get_world_pose()[0],
            dtype=float,
        )
        target_error = float(
            np.linalg.norm(final_true - np.asarray(target))
        )
        target_errors.append(target_error * 1000)

        ok = bool(pick.ok and place is not None and place.ok)
        successes += int(ok)
        row = {
            "index": index,
            "start": list(start),
            "true_start": truth_start.tolist(),
            "target": list(target),
            "pick_ok": bool(pick.ok),
            "pick_message": pick.message,
            "pick_failure_code": (
                None
                if pick.failure_code is None
                else pick.failure_code.value
            ),
            "pick_details": pick.details,
            "place_ok": bool(place is not None and place.ok),
            "place_message": None if place is None else place.message,
            "place_failure_code": (
                None
                if place is None or place.failure_code is None
                else place.failure_code.value
            ),
            "place_details": None if place is None else place.details,
            "true_final": final_true.tolist(),
            "target_error_mm": target_error * 1000,
            "object_after": object_snapshot(
                bundle.world_model.require(object_id)
            ),
        }
        metrics["manipulation_trials"].append(row)
        print(
            f"  trial {index}: "
            f"pick={'PASS' if pick.ok else 'FAIL'} "
            f"place={'PASS' if place is not None and place.ok else 'FAIL'} "
            f"target_error={target_error * 1000:.2f} mm"
        )

        if not ok:
            if bundle.world_model.held_object_id is not None:
                bundle.backend.open_gripper()
                bundle.world_model.set_held(None)

    metrics["manipulation_successes"] = successes
    metrics["manipulation_total"] = len(MANIPULATION_TRIALS)
    metrics["manipulation_success_rate"] = (
        successes / len(MANIPULATION_TRIALS)
    )
    metrics["target_error_mm"] = stats(target_errors)

    if successes != len(MANIPULATION_TRIALS):
        raise AssertionError(
            f"Only {successes}/{len(MANIPULATION_TRIALS)} "
            "benchmark manipulations succeeded"
        )
    if metrics["target_error_mm"]["max"] >= 25.0:
        raise AssertionError(
            "Benchmark placement exceeded 25 mm true target error"
        )

    print("\n=== PHASE A BENCHMARK ===")
    print(
        "Perception position error [mm]:",
        metrics["perception_position_error_mm"],
    )
    print(
        "Perception size max error [mm]:",
        metrics["perception_size_max_error_mm"],
    )
    print(
        "Manipulation success:",
        f"{successes}/{len(MANIPULATION_TRIALS)}",
    )
    print(
        "Target error [mm]:",
        metrics["target_error_mm"],
    )
    print("PASS")
    print("=========================")
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
