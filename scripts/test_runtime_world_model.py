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

    print("[3] Querying WorldModel...")
    for obj in bundle.world_model.objects():
        print(
            obj.object_id,
            f"class={obj.class_name}",
            f"pose={np.round(obj.pose.position, 6)}",
            f"size={obj.size}",
            f"visible={obj.visible}",
            f"source={obj.source}",
        )

    cube = bundle.world_model.require("cube")

    assert cube.pose is not None
    assert cube.visible
    assert cube.source == "ground_truth"

    print("\n[4] PICK through RobotRuntime...")
    pick = bundle.runtime.execute("pick", object_id="cube")
    print(pick.status, "-", pick.message)

    if not pick.ok:
        raise RuntimeError(pick.message)

    print("Cached lifted pose:", cube.pose.position)

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
    cache_error = np.linalg.norm(cube.pose.position - actual)

    print("\n=== RUNTIME WORLD MODEL ===")
    print("Objects:", [obj.object_id for obj in bundle.world_model.objects()])
    print("WorldModel source:", cube.source)
    print("Pick:", "SUCCESS")
    print("Place:", "SUCCESS")
    print("Cached vs actual:", f"{cache_error * 1000:.4f} mm")
    print("PASS")
    print("===========================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()