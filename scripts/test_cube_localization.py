from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
import tomllib
import traceback
from pathlib import Path

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends.isaac.rgbd_camera import IsaacRgbdCamera
from perception.rgbd_localizer import RgbdLocalizer

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

    stage = omni.usd.get_context().get_stage()
    cube = stage.GetPrimAtPath("/World/Cube")
    if not cube.IsValid():
        raise RuntimeError("/World/Cube not found")

    print("[2] Labelling cube...")
    add_labels(cube, labels=["cube"], instance_name="class")

    print("[3] Initializing camera...")
    camera = IsaacRgbdCamera(
        "/World/PerceptionCamera",
        resolution=(640, 480),
    )
    camera.initialize(semantic_segmentation=True)

    print("[4] Rendering...")
    for _ in range(60):
        world.step(render=True)

    print("[5] Reading perception...")
    depth = np.squeeze(camera.get_depth())
    seg, info = camera.get_semantic_segmentation()

    if depth is None or seg is None:
        raise RuntimeError("Camera data unavailable")

    cube_ids = [
        int(object_id)
        for object_id, labels in info.get("idToLabels", {}).items()
        if labels.get("class") == "cube"
    ]
    if not cube_ids:
        raise RuntimeError("Cube semantic ID not found")

    mask = np.isin(seg, cube_ids)
    valid = mask & np.isfinite(depth) & (depth > 0)

    ys, xs = np.nonzero(valid)
    if not xs.size:
        raise RuntimeError("No valid cube depth pixels")

    pixels = np.column_stack((xs, ys))
    depths = depth[ys, xs]

    print("[6] Back-projecting cube pixels...")

    isaac_points = camera.image_points_to_world(pixels, depths)

    localizer = RgbdLocalizer(camera.get_intrinsics())
    world_from_camera = camera.get_world_from_camera_transform()

    points = localizer.pixels_to_world(
        pixels,
        depths,
        world_from_camera,
    )

    difference = np.linalg.norm(points - isaac_points, axis=1)

    print(
        "Localizer vs Isaac:",
        f"mean={difference.mean():.9f} m",
        f"max={difference.max():.9f} m",
    )

    points = points[np.isfinite(points).all(axis=1)]

    if not len(points):
        raise RuntimeError("No valid world points")

    with open(PROFILE, "rb") as f:
        config = tomllib.load(f)

    cube_height = float(config["objects"]["cube"]["size"][2])

    top_z = np.percentile(points[:, 2], 95)
    top_band = points[points[:, 2] >= np.percentile(points[:, 2], 70)]

    estimated = np.array([
        np.median(top_band[:, 0]),
        np.median(top_band[:, 1]),
        top_z - cube_height / 2.0,
    ])

    transform = UsdGeom.Xformable(cube).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    ground_truth = np.asarray(transform.ExtractTranslation(), dtype=float)

    error = estimated - ground_truth

    print("\n=== CUBE 3D LOCALIZATION ===")
    print("Mask pixels:", len(pixels))
    print("Valid 3D points:", len(points))
    print()
    print("Point-cloud min:", np.min(points, axis=0))
    print("Point-cloud max:", np.max(points, axis=0))
    print()
    print("Estimated center:", estimated)
    print("Ground truth:    ", ground_truth)
    print()
    print("Error XYZ [m]:   ", error)
    print("XY error [m]:    ", float(np.linalg.norm(error[:2])))
    print("Z error [m]:     ", float(abs(error[2])))
    print("3D error [m]:    ", float(np.linalg.norm(error)))
    print("============================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()