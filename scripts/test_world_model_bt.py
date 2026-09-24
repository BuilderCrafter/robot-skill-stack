from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import asyncio
import traceback
from pathlib import Path

import numpy as np
import py_trees
from isaacsim.core.utils.stage import is_stage_loading, open_stage

from behavior_trees.executor import BTExecutor
from behavior_trees.tasks.pick_and_place import create_pick_and_place_tree
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

    print("[2] Building runtime...")
    task = asyncio.ensure_future(build_runtime(PROFILE))
    while not task.done():
        simulation_app.update()

    bundle = task.result()
    cube = bundle.world_model.require("cube")

    print("[3] WorldModel:")
    print("Pose:", cube.pose.position)
    print("Visible:", cube.visible)
    print("Source:", cube.source)

    print("\n[4] Running Pick & Place BT...")
    root = create_pick_and_place_tree(
        bundle.runtime,
        object_id="cube",
        target=Pose(TARGET),
        return_home=True,
    )

    status = BTExecutor(root).run(verbose=True)

    actual, _ = bundle.objects["cube"].get_world_pose()
    cached = cube.pose.position.copy()

    print("\n=== WORLD MODEL BT ===")
    print("BT status:", status)
    print("Held:", bundle.world_model.held_object_id)
    print("Cached final:", np.round(cached, 6))
    print("Actual final:", np.round(actual, 6))
    print(
        "Cached vs actual:",
        f"{np.linalg.norm(cached - actual) * 1000:.4f} mm",
    )
    print(
        "Target error:",
        f"{np.linalg.norm(actual - TARGET) * 1000:.2f} mm",
    )
    print(
        "Result:",
        "PASS" if status == py_trees.common.Status.SUCCESS else "FAIL",
    )
    print("======================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()