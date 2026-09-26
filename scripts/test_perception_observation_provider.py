from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import traceback
from pathlib import Path

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from pxr import Usd, UsdGeom

from backends.isaac.rgbd_camera import IsaacRgbdCamera
from perception.factory import build_perception_provider
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

    print("[2] Creating geometry perception...")
    camera = IsaacRgbdCamera(
        config.perception.camera_prim_path,
        resolution=config.perception.resolution,
    )
    camera.initialize(semantic_segmentation=False)
    for _ in range(60):
        world.step(render=True)

    provider = build_perception_provider(camera, config)
    model = WorldModel()
    updater = WorldModelUpdater(model, provider)

    print("[3] Updating blank WorldModel...")
    for _ in range(5):
        for _ in range(6):
            world.step(render=True)
        observations = updater.update()

    visible = model.visible_objects()
    if len(visible) != 1:
        raise RuntimeError(f"Expected one visible object, got {len(visible)}")
    obj = visible[0]

    print("Object ID:", obj.object_id)
    print("Class:", obj.class_name)
    print("Pose:", obj.pose.position)
    print("Orientation:", obj.pose.orientation)
    print("Size:", obj.size)
    print("Visible:", obj.visible)
    print("Source:", obj.source)

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(config.objects["cube"].prim_path)
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    truth = np.asarray(transform.ExtractTranslation(), dtype=float)
    error = float(np.linalg.norm(obj.pose.position - truth))

    print("Ground truth:", truth)
    print("Position error:", f"{error * 1000:.2f} mm")
    assert error < 0.010
    assert obj.pose.orientation is None
    assert obj.source == "rgbd_geometry_perception"

    print("\n=== PERCEPTION OBSERVATION PROVIDER ===")
    print("Blank WorldModel populated: YES")
    print("Simulator semantic labels used: NO")
    print("Persistent object ID:", obj.object_id)
    print("PASS")
    print("=======================================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()
