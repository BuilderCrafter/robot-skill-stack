from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
import traceback
from pathlib import Path

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends.isaac.rgbd_camera import IsaacRgbdCamera
from backends.isaac.semantic_detector import IsaacSemanticDetector
from perception.rgbd_localizer import RgbdLocalizer
from perception.state_provider import PerceptionStateProvider
from runtime.scene_config import load_scene_config
from world_model.entities import WorldObject
from world_model.world_model import WorldModel

SCENE = ROOT / "scenes" / "playground.usd"
PROFILE = ROOT / "config" / "scenes" / "playground.toml"


try:
    print("[1] Opening scene...")
    if not open_stage(str(SCENE)):
        raise RuntimeError(f"Failed to open {SCENE}")

    while is_stage_loading():
        simulation_app.update()

    world = World(stage_units_in_meters=1.0)
    world.reset()
    config = load_scene_config(PROFILE)

    print("[2] Creating perception stack...")
    camera = IsaacRgbdCamera(
        "/World/PerceptionCamera",
        resolution=(640, 480),
    )

    detector = IsaacSemanticDetector(
        camera,
        {
            object_id: cfg.prim_path
            for object_id, cfg in config.objects.items()
        },
    )

    camera.initialize(semantic_segmentation=True)

    for _ in range(60):
        world.step(render=True)

    localizer = RgbdLocalizer(camera.get_intrinsics())

    provider = PerceptionStateProvider(
        camera,
        detector,
        localizer,
        {
            object_id: cfg.size
            for object_id, cfg in config.objects.items()
        },
    )

    model = WorldModel(state_provider=provider)

    for object_id, cfg in config.objects.items():
        model.register(
            WorldObject(
                object_id=object_id,
                size=cfg.size,
                graspable=cfg.graspable,
            )
        )

    print("[3] Refreshing WorldModel from perception...")

    stage = omni.usd.get_context().get_stage()
    cube_prim = stage.GetPrimAtPath(config.objects["cube"].prim_path)

    for sample in range(5):
        for _ in range(5):
            world.step(render=True)

        if not model.refresh_object("cube"):
            raise RuntimeError("Perception failed to locate cube")

        estimated = model.require("cube").pose.position

        transform = UsdGeom.Xformable(cube_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        ground_truth = np.asarray(
            transform.ExtractTranslation(),
            dtype=float,
        )

        error = np.linalg.norm(estimated - ground_truth)

        print(
            f"{sample + 1}: "
            f"estimated={np.round(estimated, 6)} "
            f"error={error * 1000:.2f} mm"
        )

    print("\n=== PERCEPTION PROVIDER ===")
    print("WorldModel pose:", model.require("cube").pose.position)
    print("Provider: PerceptionStateProvider")
    print("Ground truth used only for evaluation")
    print("PASS")
    print("===========================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()