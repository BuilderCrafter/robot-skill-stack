from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
import traceback
from pathlib import Path

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import is_stage_loading, open_stage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends.isaac.rgbd_camera import IsaacRgbdCamera

SCENE = ROOT / "scenes" / "playground.usd"
OUT = ROOT / "outputs" / "perception"


def save_pgm(path, image):
    image = np.asarray(image).astype(np.uint8)
    h, w = image.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(image.tobytes())


try:
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1] Opening scene...")
    if not open_stage(str(SCENE)):
        raise RuntimeError(f"Failed to open {SCENE}")

    while is_stage_loading():
        simulation_app.update()

    world = World(stage_units_in_meters=1.0)
    world.reset()

    print("[2] Labelling cube...")
    stage = omni.usd.get_context().get_stage()
    cube = stage.GetPrimAtPath("/World/Cube")
    if not cube.IsValid():
        raise RuntimeError("/World/Cube not found")

    add_labels(cube, labels=["cube"], instance_name="class")

    print("[3] Initializing camera + segmentation...")
    camera = IsaacRgbdCamera(
        "/World/PerceptionCamera",
        resolution=(640, 480),
    )
    camera.initialize(semantic_segmentation=True)

    print("[4] Rendering...")
    for _ in range(60):
        world.step(render=True)

    print("[5] Reading segmentation...")
    seg, info = camera.get_semantic_segmentation()

    if seg is None:
        raise RuntimeError("Semantic segmentation unavailable")

    id_to_labels = info.get("idToLabels", {})

    print("\n=== SEMANTIC SEGMENTATION ===")
    print("Shape:", seg.shape)
    print("dtype:", seg.dtype)
    print("Labels:", id_to_labels)

    cube_ids = [
        int(object_id)
        for object_id, labels in id_to_labels.items()
        if labels.get("class") == "cube"
    ]

    if not cube_ids:
        raise RuntimeError("No semantic ID for 'cube'")

    mask = np.isin(seg, cube_ids)

    if not mask.any():
        raise RuntimeError("Cube label exists but no cube pixels were rendered")

    save_pgm(
        OUT / "cube_mask.pgm",
        mask.astype(np.uint8) * 255,
    )

    ys, xs = np.nonzero(mask)

    print("Cube IDs:", cube_ids)
    print("Cube pixels:", int(mask.sum()))
    print(
        "Pixel center:",
        (float(xs.mean()), float(ys.mean())),
    )
    print(
        "Bounding box:",
        (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ),
    )
    print("Saved:", OUT / "cube_mask.pgm")
    print("==============================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()