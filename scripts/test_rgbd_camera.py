from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
import traceback
from pathlib import Path

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import is_stage_loading, open_stage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends.isaac.rgbd_camera import IsaacRgbdCamera

SCENE = ROOT / "scenes" / "playground.usd"
OUT = ROOT / "outputs" / "perception"


def save_ppm(path, image):
    image = np.asarray(image)[..., :3].astype(np.uint8)
    h, w = image.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(image.tobytes())


def save_pgm(path, image):
    image = np.asarray(image).astype(np.uint8)
    h, w = image.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(image.tobytes())


try:
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1] Opening scene...", flush=True)
    if not open_stage(str(SCENE)):
        raise RuntimeError(f"Failed to open {SCENE}")

    while is_stage_loading():
        simulation_app.update()

    print("[2] Creating world...", flush=True)
    world = World(stage_units_in_meters=1.0)
    world.reset()

    print("[3] Initializing camera...", flush=True)
    camera = IsaacRgbdCamera(
        "/World/PerceptionCamera",
        resolution=(640, 480),
    )
    camera.initialize()

    print("[4] Rendering warm-up frames...", flush=True)
    for _ in range(60):
        world.step(render=True)

    print("[5] Reading frames...", flush=True)
    rgb = camera.get_rgb()
    depth = camera.get_depth()

    print("RGB available:", rgb is not None)
    print("Depth available:", depth is not None)

    if rgb is None:
        raise RuntimeError("RGB frame unavailable")
    if depth is None:
        raise RuntimeError("Depth frame unavailable")

    rgb = np.asarray(rgb)
    depth = np.squeeze(np.asarray(depth))

    K = camera.get_intrinsics()
    position, orientation = camera.get_world_pose()

    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise RuntimeError("No valid depth pixels")

    lo, hi = np.percentile(depth[valid], [2, 98])
    depth_vis = np.zeros_like(depth, dtype=np.uint8)
    scaled = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    depth_vis[valid] = ((1.0 - scaled[valid]) * 255).astype(np.uint8)

    save_ppm(OUT / "rgb.ppm", rgb)
    save_pgm(OUT / "depth.pgm", depth_vis)
    np.save(OUT / "depth.npy", depth)

    print("\n=== RGB-D CAMERA ===")
    print("RGB:", rgb.shape, rgb.dtype)
    print("Depth:", depth.shape, depth.dtype)
    print(
        "Depth min/max:",
        float(depth[valid].min()),
        float(depth[valid].max()),
    )
    print("Intrinsics:\n", K)
    print("Camera position:", position)
    print("Camera quaternion:", orientation)
    print("Saved:", OUT / "rgb.ppm")
    print("Saved:", OUT / "depth.pgm")
    print("Saved:", OUT / "depth.npy")
    print("=====================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()