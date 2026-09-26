from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import traceback
from pathlib import Path

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import is_stage_loading, open_stage

from backends.isaac.ground_truth_provider import IsaacGroundTruthProvider
from backends.isaac.rgbd_camera import IsaacRgbdCamera
from perception.factory import build_perception_provider
from runtime.scene_config import load_scene_config

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

    ground_truth = IsaacGroundTruthProvider(
        config.world.objects_root,
        config.objects,
    )

    camera = IsaacRgbdCamera(
        config.perception.camera_prim_path,
        resolution=config.perception.resolution,
    )
    camera.initialize(semantic_segmentation=False)
    for _ in range(60):
        world.step(render=True)
    perception = build_perception_provider(camera, config)

    for _ in range(5):
        for _ in range(6):
            world.step(render=True)
        perceived = [o for o in perception.observe() if o.visible]

    gt = ground_truth.observe()
    print("Ground truth objects:", [o.object_id for o in gt])
    print("Perceived objects:", [o.object_id for o in perceived])
    if len(gt) != len(perceived):
        raise RuntimeError(
            f"Object count mismatch: ground_truth={len(gt)} perception={len(perceived)}"
        )

    unmatched = perceived.copy()
    print("\n=== PROVIDER COMPARISON ===")
    for expected in gt:
        actual = min(
            unmatched,
            key=lambda o: np.linalg.norm(o.pose.position - expected.pose.position),
        )
        unmatched.remove(actual)
        position_error = float(
            np.linalg.norm(expected.pose.position - actual.pose.position)
        )
        size_error = np.abs(expected.size - actual.size)
        print(f"\nGT {expected.object_id} -> perceived {actual.object_id}")
        print("  semantic class:", actual.class_name)
        print("  position error:", f"{position_error * 1000:.2f} mm")
        print("  size error [mm]:", np.round(size_error * 1000, 2))
        print("  perception orientation:", actual.pose.orientation)
        assert position_error < 0.010
        assert float(np.max(size_error)) < 0.012
        assert actual.pose.orientation is None

    print("\nObject counts match: YES")
    print("Positions within 10 mm: YES")
    print("Sizes within 12 mm/dimension: YES")
    print("Ground truth IDs need not equal perception track IDs: EXPECTED")
    print("PASS")
    print("===========================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()
