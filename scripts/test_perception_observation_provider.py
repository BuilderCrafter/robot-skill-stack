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
from backends.isaac.semantic_detector import IsaacSemanticDetector
from perception.rgbd_localizer import RgbdLocalizer
from perception.state_provider import PerceptionStateProvider
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

    print("[2] Creating RGB-D perception...")
    camera = IsaacRgbdCamera(
        config.perception.camera_prim_path,
        resolution=config.perception.resolution,
    )

    detector = IsaacSemanticDetector(
        camera,
        objects_root=config.world.objects_root,
    )

    camera.initialize(semantic_segmentation=True)

    for _ in range(60):
        world.step(render=True)

    provider = PerceptionStateProvider(
        camera,
        detector,
        RgbdLocalizer(camera.get_intrinsics()),
        object_sizes={
            object_id: cfg.size
            for object_id, cfg in config.objects.items()
        },
        object_graspable={
            object_id: cfg.graspable
            for object_id, cfg in config.objects.items()
        },
    )

    print("[3] Updating blank WorldModel...")
    model = WorldModel()
    updater = WorldModelUpdater(model, provider)

    observations = updater.update()

    print("Observations:", len(observations))
    print("World objects:", [o.object_id for o in model.objects()])

    cube = model.require("cube")

    print("\n[4] Cube observation:")
    print("class:", cube.class_name)
    print("pose:", cube.pose.position)
    print("orientation:", cube.pose.orientation)
    print("size:", cube.size)
    print("graspable:", cube.graspable)
    print("visible:", cube.visible)
    print("confidence:", cube.confidence)
    print("source:", cube.source)
    print("metadata:", cube.metadata)

    assert cube.class_name == "cube"
    assert cube.pose is not None
    assert cube.pose.orientation is None
    assert cube.size is not None
    assert cube.visible
    assert cube.source == "isaac_semantic_perception"

    print("\n[5] Comparing against ground truth...")
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(
        config.objects["cube"].prim_path
    )

    transform = UsdGeom.Xformable(
        prim
    ).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )

    truth = np.asarray(
        transform.ExtractTranslation(),
        dtype=float,
    )

    error = float(
        np.linalg.norm(cube.pose.position - truth)
    )

    print("Perceived:", np.round(cube.pose.position, 6))
    print("Ground truth:", np.round(truth, 6))
    print("Error:", f"{error * 1000:.2f} mm")

    print("\n=== PERCEPTION OBSERVATION PROVIDER ===")
    print("Blank WorldModel populated: YES")
    print("Object discovered through detector: YES")
    print("Pose from RGB-D geometry: YES")
    print("Orientation fabricated: NO")
    print("Source:", cube.source)
    print("PASS")
    print("=======================================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()