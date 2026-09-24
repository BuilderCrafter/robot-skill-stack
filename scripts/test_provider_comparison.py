from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import traceback
from pathlib import Path

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import is_stage_loading, open_stage

from backends.isaac.ground_truth_provider import IsaacGroundTruthProvider
from backends.isaac.rgbd_camera import IsaacRgbdCamera
from backends.isaac.semantic_detector import IsaacSemanticDetector
from perception.rgbd_localizer import RgbdLocalizer
from perception.state_provider import PerceptionStateProvider
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

    print("[2] Creating ground-truth provider...")
    ground_truth = IsaacGroundTruthProvider(
        config.world.objects_root,
        config.objects,
    )

    print("[3] Creating perception provider...")
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

    perception = PerceptionStateProvider(
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

    print("[4] Observing world from both sources...")
    gt = {o.object_id: o for o in ground_truth.observe()}
    perceived = {o.object_id: o for o in perception.observe()}

    print("\nGround truth objects:", sorted(gt))
    print("Perceived objects:   ", sorted(perceived))

    assert set(gt) == set(perceived)

    print("\n=== PROVIDER COMPARISON ===")

    for object_id in sorted(gt):
        a = gt[object_id]
        b = perceived[object_id]

        print(f"\n{object_id}")
        print("  class:")
        print("    ground truth:", a.class_name)
        print("    perception:  ", b.class_name)

        print("  position:")
        print("    ground truth:", np.round(a.pose.position, 6))
        print("    perception:  ", np.round(b.pose.position, 6))

        position_error = float(
            np.linalg.norm(a.pose.position - b.pose.position)
        )

        print(
            "    error:       ",
            f"{position_error * 1000:.2f} mm",
        )

        print("  orientation:")
        print("    ground truth:", a.pose.orientation)
        print("    perception:  ", b.pose.orientation)

        print("  size:")
        print("    ground truth:", a.size)
        print("    perception:  ", b.size)

        print("  graspable:")
        print("    ground truth:", a.graspable)
        print("    perception:  ", b.graspable)

        print("  sources:")
        print("    ground truth:", a.source)
        print("    perception:  ", b.source)

        assert a.class_name == b.class_name
        assert a.visible and b.visible
        assert np.allclose(a.size, b.size)
        assert a.graspable == b.graspable
        assert position_error < 0.01

    print("\nObject sets match: YES")
    print("Semantic classes match: YES")
    print("Sizes match: YES")
    print("Graspability matches: YES")
    print("Positions within 10 mm: YES")
    print("Orientation difference expected: perception=None")
    print("PASS")
    print("===========================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()