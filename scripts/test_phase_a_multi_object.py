from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from core.types import Pose
from scripts.perception_v1_common import write_result
from scripts.phase_a_common import (
    build_perception_runtime,
    nearest_object,
    object_snapshot,
    open_playground,
    settle,
)

SECOND_PATH = "/World/Objects/PhaseASecondCube"
SECOND_START = np.array([0.40, -0.14, 0.025])
FIRST_TARGET = np.array([0.48, 0.22, 0.025])
SECOND_TARGET = np.array([0.38, 0.10, 0.025])


def spawn_rigid_cube():
    stage = omni.usd.get_context().get_stage()
    cube = UsdGeom.Cube.Define(stage, SECOND_PATH)
    cube.CreateSizeAttr(0.05)
    prim = cube.GetPrim()

    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(
        Gf.Vec3d(*SECOND_START.tolist())
    )

    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(0.05)
    return prim


def prim_position(prim):
    transform = UsdGeom.Xformable(
        prim
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.asarray(
        transform.ExtractTranslation(),
        dtype=float,
    )


def main(result_path=None, artifacts_dir=None):
    metrics = {}
    out = Path(
        artifacts_dir
        or Path("outputs") / "phase_a_multi_tmp"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("[1] Opening playground...")
    open_playground(simulation_app)

    print("[2] Spawning second physical cube...")
    second_prim = spawn_rigid_cube()

    print("[3] Building runtime and discovering both cubes...")
    bundle = build_perception_runtime(simulation_app, out)
    settle(bundle.world, 240)

    first_true = np.asarray(
        bundle.objects["cube"].get_world_pose()[0],
        dtype=float,
    )
    second_true = prim_position(second_prim)

    visible = bundle.world_model.visible_objects()
    if len(visible) != 2:
        raise RuntimeError(
            "Expected exactly two visible objects, got "
            f"{[(o.object_id, o.pose.position.tolist()) for o in visible]}"
        )

    first_obj = nearest_object(visible, first_true)
    second_obj = nearest_object(visible, second_true)

    if first_obj.object_id == second_obj.object_id:
        raise AssertionError("Both physical cubes mapped to one track")

    first_id = first_obj.object_id
    second_id = second_obj.object_id
    metrics.update(
        first_id=first_id,
        second_id=second_id,
        first_initial=object_snapshot(first_obj),
        second_initial=object_snapshot(second_obj),
    )

    print("First cube ID:", first_id)
    print("Second cube ID:", second_id)

    print("[4] Manipulate first cube...")
    pick1 = bundle.runtime.execute(
        "pick",
        object_id=first_id,
    )
    if not pick1.ok:
        raise RuntimeError(
            f"First Pick failed: {pick1.message}"
        )
    place1 = bundle.runtime.execute(
        "place",
        target=Pose(FIRST_TARGET),
        mode="stable",
    )
    if not place1.ok:
        raise RuntimeError(
            f"First Place failed: {place1.message}"
        )
    bundle.runtime.execute("home")
    settle(bundle.world, 60)

    first_true_after = np.asarray(
        bundle.objects["cube"].get_world_pose()[0],
        dtype=float,
    )
    first_error = float(
        np.linalg.norm(first_true_after - FIRST_TARGET)
    )

    if not bundle.world_model.require(second_id).visible:
        raise AssertionError(
            "Second cube was lost while manipulating first cube"
        )

    print("[5] Manipulate second cube...")
    pick2 = bundle.runtime.execute(
        "pick",
        object_id=second_id,
    )
    if not pick2.ok:
        raise RuntimeError(
            f"Second Pick failed: {pick2.message}"
        )
    place2 = bundle.runtime.execute(
        "place",
        target=Pose(SECOND_TARGET),
        mode="stable",
    )
    if not place2.ok:
        raise RuntimeError(
            f"Second Place failed: {place2.message}"
        )
    bundle.runtime.execute("home")
    settle(bundle.world, 60)

    second_true_after = prim_position(second_prim)
    second_error = float(
        np.linalg.norm(second_true_after - SECOND_TARGET)
    )

    first_final = bundle.world_model.require(first_id)
    second_final = bundle.world_model.require(second_id)
    visible_final = bundle.world_model.visible_objects()

    metrics.update(
        pick1={
            "ok": bool(pick1.ok),
            "message": pick1.message,
            "details": pick1.details,
        },
        place1={
            "ok": bool(place1.ok),
            "message": place1.message,
            "details": place1.details,
        },
        pick2={
            "ok": bool(pick2.ok),
            "message": pick2.message,
            "details": pick2.details,
        },
        place2={
            "ok": bool(place2.ok),
            "message": place2.message,
            "details": place2.details,
        },
        first_true_target_error_mm=first_error * 1000,
        second_true_target_error_mm=second_error * 1000,
        first_final=object_snapshot(first_final),
        second_final=object_snapshot(second_final),
        visible_final_ids=[
            obj.object_id for obj in visible_final
        ],
    )

    if not first_final.visible or not second_final.visible:
        raise AssertionError(
            "Both persistent IDs must be visible at the end"
        )
    if len(visible_final) != 2:
        raise AssertionError(
            f"Expected two final visible tracks, got {len(visible_final)}"
        )
    if {
        obj.object_id for obj in visible_final
    } != {first_id, second_id}:
        raise AssertionError(
            "Final visible IDs changed or duplicates appeared"
        )
    if first_error >= 0.025 or second_error >= 0.025:
        raise AssertionError(
            "One of the physical placement errors exceeded 25 mm"
        )

    print("\n=== PHASE A MULTI-OBJECT ===")
    print("First ID:", first_id, "preserved")
    print("Second ID:", second_id, "preserved")
    print(
        "First target error:",
        f"{first_error * 1000:.2f} mm",
    )
    print(
        "Second target error:",
        f"{second_error * 1000:.2f} mm",
    )
    print("Visible final IDs:", metrics["visible_final_ids"])
    print("PASS")
    print("============================")
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
