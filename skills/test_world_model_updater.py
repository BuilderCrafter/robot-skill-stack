from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import traceback
from pathlib import Path

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import is_stage_loading, open_stage

from backends.isaac.ground_truth_provider import IsaacGroundTruthProvider
from runtime.scene_config import load_scene_config
from world_model.world_model import WorldModel
from world_model.updater import WorldModelUpdater

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / "playground.usd"
PROFILE = ROOT / "config" / "scenes" / "playground.toml"

POSITIONS = [
    np.array([0.4465, 0.00, 0.025]),
    np.array([0.4200, 0.10, 0.025]),
    np.array([0.4800, -0.10, 0.025]),
]


try:
    print("[1] Opening scene...")
    if not open_stage(str(SCENE)):
        raise RuntimeError(f"Failed to open {SCENE}")

    while is_stage_loading():
        simulation_app.update()

    config = load_scene_config(PROFILE)
    world = World(stage_units_in_meters=1.0)

    objects = {}
    for object_id, cfg in config.objects.items():
        objects[object_id] = world.scene.add(
            SingleXFormPrim(
                prim_path=cfg.prim_path,
                name=object_id,
                reset_xform_properties=False,
            )
        )

    world.reset()
    world.play()

    print("[2] Creating provider + WorldModel...")
    provider = IsaacGroundTruthProvider(objects, config.objects)
    model = WorldModel()
    updater = WorldModelUpdater(model, provider)

    print("[3] Initial update...")
    observations = updater.update()

    assert len(observations) == len(config.objects)
    assert model.exists("cube")

    cube = model.require("cube")

    print("Objects:", [obj.object_id for obj in model.objects()])
    print("Cube pose:", cube.pose.position)
    print("Cube orientation:", cube.pose.orientation)
    print("Cube size:", cube.size)
    print("Cube visible:", cube.visible)
    print("Cube confidence:", cube.confidence)
    print("Cube source:", cube.source)

    assert cube.pose is not None
    assert cube.pose.orientation is not None
    assert cube.size is not None
    assert cube.visible
    assert cube.confidence == 1.0
    assert cube.source == "ground_truth"

    print("\n[4] Moving cube and updating cached state...")

    _, orientation = objects["cube"].get_world_pose()

    for i, target in enumerate(POSITIONS, 1):
        objects["cube"].set_world_pose(
            position=target,
            orientation=orientation,
        )

        for _ in range(10):
            world.step(render=True)

        updater.update()

        cached = model.require("cube").pose.position
        actual, _ = objects["cube"].get_world_pose()
        error = float(np.linalg.norm(cached - actual))

        print(
            f"{i}: actual={np.round(actual, 6)} "
            f"cached={np.round(cached, 6)} "
            f"error={error * 1000:.6f} mm"
        )

        assert error < 1e-6

    print("\n=== WORLD MODEL UPDATER ===")
    print("Provider: IsaacGroundTruthProvider")
    print("Objects:", len(model.objects()))
    print("Cached state follows scene state: YES")
    print("PASS")
    print("===========================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()