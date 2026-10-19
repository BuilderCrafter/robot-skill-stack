# Match the GUI launch contract before Isaac/Kit loads native extensions.
import numpy as np

if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
from contextlib import ExitStack
from unittest.mock import patch
import sys
import traceback
from pathlib import Path

import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics

from robot_skill_stack.common.types import Pose
from robot_skill_stack.runtime.skill import FailureCode
from tests.support.isaac_helpers import (
    build_ground_truth_runtime,
    nearest_object,
    open_playground,
    settle,
)
from tests.support.isaac_async import run_kit_coroutine
from tests.support.results import write_result

OBSTACLE_PATH = "/World/Objects/PlacementObstacle"
OBSTACLE_POS = np.array([0.40, -0.14, 0.025])
SAFE_TARGET = np.array([0.48, 0.20, 0.025])


def spawn_obstacle():
    stage = omni.usd.get_context().get_stage()
    cube = UsdGeom.Cube.Define(stage, OBSTACLE_PATH)
    cube.CreateSizeAttr(0.05)
    prim = cube.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*OBSTACLE_POS.tolist()))
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.05)
    return prim


def _run_placement(metrics, result_path=None, artifacts_dir=None):
    out = Path(artifacts_dir or Path("outputs") / "placement_feasibility_tmp")
    out.mkdir(parents=True, exist_ok=True)

    print("[1] Opening playground and spawning obstacle...")
    open_playground(simulation_app)
    spawn_obstacle()

    print("[2] Building ground-truth runtime...")
    bundle = build_ground_truth_runtime(simulation_app, out)
    settle(bundle.world, 90)

    held_id = "cube"
    obstacle = nearest_object(
        [obj for obj in bundle.world_model.visible_objects() if obj.object_id != held_id],
        OBSTACLE_POS,
    )
    print("Obstacle WorldModel ID:", obstacle.object_id)

    print("[3] Pick configured cube...")
    pick = run_kit_coroutine(bundle.runtime.execute_async("pick", object_id=held_id), simulation_app)
    if not pick.ok:
        raise RuntimeError(f"Pick failed: {pick.message}")

    ee_before = bundle.backend.get_end_effector_pose().position.copy()

    print("[4] Attempt occupied placement; no backend commands may be issued...")
    # Kit updates can continue an earlier physical action even when Place rejects.
    # Spy on command invocations (including unawaited calls), allowing read-only queries.
    with ExitStack() as stack:
        commands = {name: stack.enter_context(patch.object(
            bundle.backend, name, wraps=getattr(bundle.backend, name)))
            for name in ("move_to_pose_async", "open_gripper_async", "close_gripper_async", "home_async")}
        occupied = run_kit_coroutine(bundle.runtime.execute_async(
            "place",
            target=Pose(obstacle.pose.position.copy()),
            mode="stable",
        ), simulation_app)
        command_counts = {name: spy.call_count for name, spy in commands.items()}
    ee_after = bundle.backend.get_end_effector_pose().position.copy()
    ee_motion = float(np.linalg.norm(ee_after - ee_before))

    metrics.update(
        occupied_status=occupied.status.value,
        occupied_message=occupied.message,
        occupied_failure_code=(
            None if occupied.failure_code is None else occupied.failure_code.value
        ),
        occupied_details=occupied.details,
        ee_motion_during_rejection_mm=ee_motion * 1000,
        rejection_backend_command_counts=command_counts,
        held_after_rejection=bundle.world_model.held_object_id,
    )

    if occupied.ok:
        raise AssertionError("Occupied placement unexpectedly succeeded")
    if occupied.failure_code != FailureCode.TARGET_OCCUPIED:
        raise AssertionError(
            f"Expected TARGET_OCCUPIED, got {occupied.failure_code}"
        )
    print("EE drift during Kit updates [mm]:", ee_motion * 1000, "commands:", command_counts)
    if any(command_counts.values()):
        raise AssertionError(f"Backend commands issued despite placement rejection: {command_counts}")
    if bundle.world_model.held_object_id != held_id:
        raise AssertionError("Held-object state changed after rejected placement")

    print("[5] Place at a free target...")
    place = run_kit_coroutine(bundle.runtime.execute_async(
        "place",
        target=Pose(SAFE_TARGET),
        mode="stable",
    ), simulation_app)
    if not place.ok:
        raise RuntimeError(f"Safe placement failed: {place.message}")

    final_true = np.asarray(bundle.objects["cube"].get_world_pose()[0], dtype=float)
    error = float(np.linalg.norm(final_true - SAFE_TARGET))
    metrics.update(
        safe_place_status=place.status.value,
        safe_place_message=place.message,
        safe_place_details=place.details,
        true_target_error_mm=error * 1000,
    )
    if error >= 0.025:
        raise AssertionError(f"Safe placement error too large: {error:.3f} m")

    print("\n=== PLACEMENT FEASIBILITY ===")
    print("Occupied target: REJECTED")
    print("EE motion during rejection:", f"{ee_motion * 1000:.4f} mm")
    print("Safe place: SUCCESS")
    print("True target error:", f"{error * 1000:.2f} mm")
    print("PASS")
    print("=============================")
    write_result(result_path, status="PASS", metrics=metrics)


def main(result_path=None, artifacts_dir=None):
    metrics = {}
    try:
        _run_placement(metrics, result_path, artifacts_dir)
    except Exception as exc:
        write_result(result_path, status="FAIL", metrics=metrics,
                     error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result")
    parser.add_argument("--artifacts-dir")
    args = parser.parse_args()
    try:
        main(args.result, args.artifacts_dir)
    except Exception as exc:
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
