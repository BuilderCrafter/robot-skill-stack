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
    settle,
)

MOVE_DELTA = np.array([0.0, 0.05, 0.0])
PLACE_TARGET = np.array([0.4465, 0.22, 0.025])


def main(result_path=None, artifacts_dir=None):
    metrics = {}
    out = Path(
        artifacts_dir
        or Path("outputs") / "phase_a_reactive_tmp"
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
    initial_true = np.asarray(
        cube.get_world_pose()[0],
        dtype=float,
    )
    _, cube_orientation = cube.get_world_pose()

    state = {
        "steps": 0,
        "moved": False,
        "true_target": None,
    }

    def disturb(step_size):
        state["steps"] += 1
        if not state["moved"] and state["steps"] >= 80:
            position = np.asarray(
                cube.get_world_pose()[0],
                dtype=float,
            )
            target = position + MOVE_DELTA
            cube.set_world_pose(
                position=target,
                orientation=cube_orientation,
            )
            state["moved"] = True
            state["true_target"] = target.tolist()
            print(
                "[disturbance] moved cube by",
                MOVE_DELTA.tolist(),
            )

    callback = "phase_a_reactive_pick_disturbance"
    bundle.world.add_physics_callback(
        callback,
        callback_fn=disturb,
    )

    print("[3] PICK while cube is moved during approach...")
    pick = bundle.runtime.execute(
        "pick",
        object_id=object_id,
    )

    bundle.world.remove_physics_callback(callback)

    true_after_pick = np.asarray(
        cube.get_world_pose()[0],
        dtype=float,
    )
    true_lift = float(
        true_after_pick[2] - initial_true[2]
    )

    metrics.update(
        object_id=object_id,
        disturbance_triggered=state["moved"],
        disturbance_step=state["steps"],
        disturbance_target=state["true_target"],
        pick_ok=bool(pick.ok),
        pick_message=pick.message,
        pick_failure_code=(
            None
            if pick.failure_code is None
            else pick.failure_code.value
        ),
        pick_details=pick.details,
        local_replans=pick.details.get("local_replans"),
        true_lift_mm=true_lift * 1000,
        object_after_pick=object_snapshot(obj),
    )

    print("Pick:", pick.status, "-", pick.message)
    print("Local replans:", metrics["local_replans"])
    print("True lift:", f"{true_lift * 1000:.2f} mm")

    if not state["moved"]:
        raise RuntimeError("Disturbance callback did not trigger")
    if not pick.ok:
        raise RuntimeError(
            f"Reactive Pick failed: {pick.message}"
        )
    if int(pick.details.get("local_replans", 0)) < 1:
        raise AssertionError(
            "Pick succeeded without recording a local replan"
        )
    if true_lift < 0.03:
        raise AssertionError("Cube was not physically lifted")

    print("[4] Place after reactive Pick...")
    place = bundle.runtime.execute(
        "place",
        target=Pose(PLACE_TARGET),
        mode="stable",
    )
    final_true = np.asarray(
        cube.get_world_pose()[0],
        dtype=float,
    )
    target_error = float(
        np.linalg.norm(final_true - PLACE_TARGET)
    )
    metrics.update(
        place_ok=bool(place.ok),
        place_message=place.message,
        place_details=place.details,
        true_target_error_mm=target_error * 1000,
        final_object=object_snapshot(
            bundle.world_model.require(object_id)
        ),
    )

    if not place.ok:
        raise RuntimeError(
            f"Place after reactive Pick failed: {place.message}"
        )
    if target_error >= 0.025:
        raise AssertionError(
            f"True placement error too large: {target_error:.3f} m"
        )

    print("\n=== PHASE A REACTIVE PICK ===")
    print("Disturbance: TRIGGERED")
    print("Local replans:", metrics["local_replans"])
    print("Pick: SUCCESS")
    print("Place: SUCCESS")
    print(
        "True target error:",
        f"{target_error * 1000:.2f} mm",
    )
    print("PASS")
    print("=============================")
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
