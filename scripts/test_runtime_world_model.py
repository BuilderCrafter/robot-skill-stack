from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import asyncio
import traceback
from pathlib import Path

import numpy as np
from isaacsim.core.utils.stage import is_stage_loading, open_stage

from core.types import Pose
from runtime.runtime_builder import build_runtime

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / "playground.usd"
PROFILE = ROOT / "config" / "scenes" / "playground.toml"
TARGET = np.array([0.4465, 0.25, 0.025])


def choose_object(model):
    if model.exists("cube") and model.require("cube").visible:
        return model.require("cube")
    visible = model.visible_objects()
    cubes = [o for o in visible if o.class_name == "cube"]
    if len(cubes) == 1:
        return cubes[0]
    if len(visible) == 1:
        return visible[0]
    raise RuntimeError(f"Expected one manipulation object, got {len(visible)}")


try:
    print("[1] Opening scene...")
    if not open_stage(str(SCENE)):
        raise RuntimeError(f"Failed to open {SCENE}")
    while is_stage_loading():
        simulation_app.update()

    print("[2] Building real runtime...")
    task = asyncio.ensure_future(build_runtime(PROFILE))
    while not task.done():
        simulation_app.update()
    bundle = task.result()

    for _ in range(180):
        bundle.world.step(render=True)

    print("[3] Querying WorldModel...")
    for obj in bundle.world_model.objects():
        print(
            obj.object_id,
            f"class={obj.class_name}",
            f"pose={None if obj.pose is None else np.round(obj.pose.position, 6)}",
            f"size={obj.size}",
            f"visible={obj.visible}",
            f"source={obj.source}",
        )

    obj = choose_object(bundle.world_model)

    print("\n[4] PICK through RobotRuntime...")
    pick = bundle.runtime.execute("pick", object_id=obj.object_id)
    print(pick.status, "-", pick.message)
    if not pick.ok:
        raise RuntimeError(pick.message)

    print("\n[5] PLACE through RobotRuntime...")
    place = bundle.runtime.execute(
        "place",
        target=Pose(TARGET),
        mode="stable",
    )
    print(place.status, "-", place.message)
    if not place.ok:
        raise RuntimeError(place.message)

    actual, _ = bundle.objects["cube"].get_world_pose()
    cache_error = np.linalg.norm(obj.pose.position - actual)

    print("\n=== RUNTIME WORLD MODEL ===")
    print("Selected object:", obj.object_id)
    print("Source:", obj.source)
    print("Pick: SUCCESS")
    print("Place: SUCCESS")
    print("Cached vs actual:", f"{cache_error * 1000:.4f} mm")
    print("PASS")
    print("===========================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()
