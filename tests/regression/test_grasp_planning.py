from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

from robot_skill_stack.runtime.skill import FailureCode
from robot_skill_stack.common.types import Pose
from tests.support.results import write_result
from tests.support.isaac_helpers import (
    build_perception_runtime,
    choose_object,
    object_snapshot,
    open_playground,
    settle,
)
from robot_skill_stack.world.model.entities import WorldObject

TARGET = np.array([0.4465, 0.22, 0.025])


def main(result_path=None, artifacts_dir=None):
    metrics = {}
    out = Path(
        artifacts_dir
        or Path("outputs") / "phase_b_isaac_tmp"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("[1] Opening playground...")
    open_playground(simulation_app)

    print("[2] Building runtime...")
    bundle = build_perception_runtime(simulation_app, out)
    settle(bundle.world, 180)

    obj = choose_object(bundle.world_model)
    object_id = obj.object_id

    print("[3] Verify oversized object is rejected before motion...")
    ee_before = bundle.backend.get_end_effector_pose().position.copy()
    fake_id = "oversized_test_object"
    bundle.world_model.register(
        WorldObject(
            object_id=fake_id,
            class_name="cube",
            pose=Pose([0.45, -0.10, 0.045]),
            size=np.array([0.09, 0.09, 0.09]),
            graspable=True,
            visible=True,
        )
    )

    oversized = bundle.runtime.execute(
        "pick",
        object_id=fake_id,
    )
    ee_after = bundle.backend.get_end_effector_pose().position.copy()
    ee_motion = float(np.linalg.norm(ee_after - ee_before))

    metrics.update(
        oversized_status=oversized.status.value,
        oversized_message=oversized.message,
        oversized_failure_code=(
            None
            if oversized.failure_code is None
            else oversized.failure_code.value
        ),
        oversized_details=oversized.details,
        ee_motion_during_rejection_mm=ee_motion * 1000,
    )

    print(
        "Oversized result:",
        oversized.status,
        oversized.failure_code,
        oversized.message,
    )
    print(
        "EE motion during rejection:",
        f"{ee_motion * 1000:.4f} mm",
    )

    if oversized.ok:
        raise AssertionError("Oversized object was unexpectedly accepted")
    if oversized.failure_code != FailureCode.NO_VALID_GRASP:
        raise AssertionError(
            f"Expected NO_VALID_GRASP, got {oversized.failure_code}"
        )
    if (
        oversized.details.get("grasp_failure_reason")
        != "object_too_large"
    ):
        raise AssertionError(
            "Planner did not report object_too_large"
        )
    if ee_motion > 1e-4:
        raise AssertionError(
            "Robot moved even though grasp feasibility failed"
        )

    bundle.world_model._objects.pop(fake_id, None)

    print("[4] Real perception-driven cube Pick + Place...")
    pick = bundle.runtime.execute(
        "pick",
        object_id=object_id,
    )
    if not pick.ok:
        raise RuntimeError(f"Pick failed: {pick.message}")

    place = bundle.runtime.execute(
        "place",
        target=Pose(TARGET),
        mode="stable",
    )
    if not place.ok:
        raise RuntimeError(f"Place failed: {place.message}")

    true_final = np.asarray(
        bundle.objects["cube"].get_world_pose()[0],
        dtype=float,
    )
    target_error = float(
        np.linalg.norm(true_final - TARGET)
    )

    metrics.update(
        object_id=object_id,
        pick_status=pick.status.value,
        pick_message=pick.message,
        pick_details=pick.details,
        place_status=place.status.value,
        place_message=place.message,
        place_details=place.details,
        true_target_error_mm=target_error * 1000,
        final_object=object_snapshot(
            bundle.world_model.require(object_id)
        ),
    )

    planner_details = pick.details.get(
        "grasp_planner_details",
        {},
    )
    if not planner_details.get("feasibility_checked"):
        raise AssertionError(
            "Pick did not record planner feasibility evidence"
        )
    if abs(
        planner_details.get("gripper_max_width_m", 0.0) - 0.075
    ) > 1e-6:
        raise AssertionError(
            "Unexpected configured gripper max width"
        )
    if target_error >= 0.025:
        raise AssertionError(
            f"True target error too large: {target_error:.3f} m"
        )

    print("\n=== PHASE B ISAAC ===")
    print("Oversized object: REJECTED BEFORE MOTION")
    print("Small cube feasibility: PASS")
    print("Pick: SUCCESS")
    print("Place: SUCCESS")
    print(
        "True target error:",
        f"{target_error * 1000:.2f} mm",
    )
    print("PASS")
    print("=====================")
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
