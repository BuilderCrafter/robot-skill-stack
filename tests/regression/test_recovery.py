from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import py_trees

from robot_skill_stack.orchestration.behavior_trees.executor import BTExecutor
from robot_skill_stack.orchestration.behavior_trees.tasks.pick_and_place_recovery import (
    create_pick_and_place_recovery_tree,
)
from robot_skill_stack.runtime.skill import FailureCode
from robot_skill_stack.common.types import Pose
from tests.support.results import write_result
from tests.support.isaac_helpers import (
    build_perception_runtime,
    choose_object,
    object_snapshot,
    open_playground,
    settle,
    step,
)

HIDDEN_POSITION = np.array([0.80, 0.0, 0.025])
RESTORE_POSITION = np.array([0.4465, 0.0, 0.025])
PLACE_TARGET = np.array([0.4465, 0.22, 0.025])


def result_dict(result):
    if result is None:
        return None
    return {
        "ok": bool(result.ok),
        "status": result.status.value,
        "message": result.message,
        "failure_code": (
            None
            if result.failure_code is None
            else result.failure_code.value
        ),
        "details": result.details,
    }


def main(result_path=None, artifacts_dir=None):
    metrics = {}
    out = Path(
        artifacts_dir
        or Path("outputs") / "phase_a_recovery_tmp"
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
    _, orientation = cube.get_world_pose()

    print("[3] Move robot away from Home to make recovery observable...")
    ee = bundle.backend.get_end_effector_pose()
    away = ee.translated([0.0, 0.0, 0.15])
    moved = bundle.runtime.execute(
        "move_to_pose",
        target=away,
        speed=0.3,
    )
    if not moved.ok:
        raise RuntimeError(
            f"Could not prepare recovery pose: {moved.message}"
        )

    print("[4] Hide cube from perception...")
    cube.set_world_pose(
        position=HIDDEN_POSITION,
        orientation=orientation,
    )
    step(bundle.world, 48)

    if obj.visible:
        raise AssertionError(
            "Object did not become invisible before recovery test"
        )

    restore = {"done": False, "steps": 0}

    def restore_during_home(step_size):
        restore["steps"] += 1
        if not restore["done"]:
            cube.set_world_pose(
                position=RESTORE_POSITION,
                orientation=orientation,
            )
            restore["done"] = True
            print("[recovery disturbance] cube restored")

    callback = "phase_a_recovery_restore"
    bundle.world.add_physics_callback(
        callback,
        callback_fn=restore_during_home,
    )

    print("[5] Run recovery BT...")
    root = create_pick_and_place_recovery_tree(
        bundle.runtime,
        object_id=object_id,
        target=Pose(PLACE_TARGET),
        return_home=True,
    )
    status = BTExecutor(root).run(verbose=True)

    bundle.world.remove_physics_callback(callback)

    nodes = {
        node.name: node
        for node in root.iterate()
    }
    first_pick = nodes.get(f"Pick {object_id}")
    retry_pick = nodes.get(f"Retry Pick {object_id}")
    recovery_home = nodes.get("Recovery Home")

    first_result = None if first_pick is None else first_pick.result
    retry_result = None if retry_pick is None else retry_pick.result
    home_result = None if recovery_home is None else recovery_home.result

    final_true = np.asarray(
        cube.get_world_pose()[0],
        dtype=float,
    )
    target_error = float(
        np.linalg.norm(final_true - PLACE_TARGET)
    )

    metrics.update(
        object_id=object_id,
        restore_triggered=restore["done"],
        restore_callback_steps=restore["steps"],
        first_pick=result_dict(first_result),
        recovery_home=result_dict(home_result),
        retry_pick=result_dict(retry_result),
        bt_status=str(status),
        true_target_error_mm=target_error * 1000,
        final_object=object_snapshot(
            bundle.world_model.require(object_id)
        ),
    )

    if not restore["done"]:
        raise RuntimeError("Recovery restore callback did not run")
    if first_result is None or first_result.ok:
        raise AssertionError("First Pick was expected to fail")
    if first_result.failure_code != FailureCode.OBJECT_POSE_UNKNOWN:
        raise AssertionError(
            "First Pick did not fail with OBJECT_POSE_UNKNOWN"
        )
    if home_result is None or not home_result.ok:
        raise AssertionError("Recovery Home did not succeed")
    if retry_result is None or not retry_result.ok:
        raise AssertionError("Retry Pick did not succeed")
    if status != py_trees.common.Status.SUCCESS:
        raise AssertionError(
            f"Recovery BT ended with {status}"
        )
    if target_error >= 0.025:
        raise AssertionError(
            f"Recovered placement error too large: "
            f"{target_error:.3f} m"
        )

    print("\n=== PHASE A RECOVERY ===")
    print(
        "First Pick:",
        first_result.failure_code.value,
    )
    print("Recovery Home: SUCCESS")
    print("Retry Pick: SUCCESS")
    print("BT: SUCCESS")
    print(
        "True target error:",
        f"{target_error * 1000:.2f} mm",
    )
    print("PASS")
    print("========================")
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
