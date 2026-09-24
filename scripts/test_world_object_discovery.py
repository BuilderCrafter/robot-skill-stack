from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import traceback
from pathlib import Path

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from pxr import Gf, UsdGeom

from backends.isaac.ground_truth_provider import IsaacGroundTruthProvider
from runtime.scene_config import load_scene_config
from world_model.updater import WorldModelUpdater
from world_model.world_model import WorldModel

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / "playground.usd"
PROFILE = ROOT / "config" / "scenes" / "playground.toml"

try:
    print("[1] Opening scene...")
    if not open_stage(str(SCENE)):
        raise RuntimeError(f"Failed to open {SCENE}")

    while is_stage_loading():
        simulation_app.update()

    config = load_scene_config(PROFILE)
    world = World(stage_units_in_meters=1.0)
    world.reset()
    world.play()

    provider = IsaacGroundTruthProvider(
        config.world.objects_root,
        config.objects,
    )
    model = WorldModel()
    updater = WorldModelUpdater(model, provider)
    updater.start(world)

    for _ in range(20):
        world.step(render=True)

    print("[2] Initial discovery...")
    print("Objects:", [o.object_id for o in model.objects()])

    cube = model.require("cube")
    assert cube.visible
    assert cube.class_name == "cube"
    assert cube.pose is not None
    assert cube.pose.orientation is not None
    assert cube.size is not None

    print(
        "cube:",
        f"pose={np.round(cube.pose.position, 6)}",
        f"size={np.round(cube.size, 6)}",
        f"source={cube.source}",
    )

    print("\n[3] Adding an UNCONFIGURED object...")
    stage = omni.usd.get_context().get_stage()

    box = UsdGeom.Cube.Define(
        stage,
        f"{config.world.objects_root}/test_box",
    )
    box.CreateSizeAttr(0.04)

    xform = UsdGeom.Xformable(box.GetPrim())
    xform.AddTranslateOp().Set(
        Gf.Vec3d(0.30, -0.20, 0.02)
    )

    box.GetPrim().SetCustomDataByKey(
        "class_name",
        "box",
    )
    box.GetPrim().SetCustomDataByKey(
        "graspable",
        False,
    )

    for _ in range(10):
        world.step(render=True)

    discovered = model.require("test_box")

    print("Objects:", [o.object_id for o in model.objects()])
    print(
        "test_box:",
        f"class={discovered.class_name}",
        f"pose={np.round(discovered.pose.position, 6)}",
        f"size={np.round(discovered.size, 6)}",
        f"graspable={discovered.graspable}",
    )

    assert discovered.visible
    assert discovered.class_name == "box"
    assert not discovered.graspable
    assert np.allclose(discovered.size, [0.04, 0.04, 0.04])

    print("\n[4] Removing object from scene...")
    stage.RemovePrim(
        f"{config.world.objects_root}/test_box"
    )

    for _ in range(10):
        world.step(render=True)

    assert model.exists("test_box")
    assert not model.require("test_box").visible

    print(
        "test_box retained in WorldModel:",
        model.exists("test_box"),
    )
    print(
        "test_box visible:",
        model.require("test_box").visible,
    )

    print("\n=== WORLD OBJECT DISCOVERY ===")
    print("Configured cube discovered: YES")
    print("Unconfigured object discovered: YES")
    print("Pose discovered: YES")
    print("Orientation discovered: YES")
    print("Size discovered: YES")
    print("Removed object preserved as invisible: YES")
    print("PASS")
    print("==============================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()