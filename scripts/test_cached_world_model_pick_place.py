from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import traceback
from pathlib import Path

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from isaacsim.robot.manipulators.examples.franka import Franka

from backends.isaac.franka_backend import IsaacFrankaBackend
from backends.isaac.ground_truth_provider import IsaacGroundTruthProvider
from core.types import Pose
from manipulation.grasp_planner import TopDownGraspPlanner
from runtime.robot_runtime import RobotRuntime
from runtime.scene_config import load_scene_config
from runtime.skill_registry import SkillRegistry
from skills.home import HomeSkill
from skills.move_to_pose import MoveToPoseSkill
from skills.pick import PickSkill
from skills.place import PlaceSkill
from world_model.entities import WorldObject
from world_model.updater import WorldModelUpdater
from world_model.world_model import WorldModel

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / "playground.usd"
PROFILE = ROOT / "config" / "scenes" / "playground.toml"
TARGET = np.array([0.4465, 0.25, 0.025])


try:
    print("[1] Opening scene...")
    if not open_stage(str(SCENE)):
        raise RuntimeError(f"Failed to open {SCENE}")

    while is_stage_loading():
        simulation_app.update()

    config = load_scene_config(PROFILE)
    world = World(stage_units_in_meters=1.0)

    print("[2] Creating robot + objects...")
    robot = world.scene.add(
        Franka(
            prim_path=config.robot.prim_path,
            name=config.robot.id,
        )
    )

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

    print("[3] Creating continuously updated WorldModel...")
    provider = IsaacGroundTruthProvider(objects, config.objects)
    model = WorldModel()

    for object_id, cfg in config.objects.items():
        model.register(
            WorldObject(
                object_id=object_id,
                class_name=object_id,
                size=cfg.size,
                graspable=cfg.graspable,
            )
        )

    updater = WorldModelUpdater(model, provider)
    updater.start(world)

    for _ in range(60):
        world.step(render=True)

    cube = model.require("cube")

    print("Cached pose:", cube.pose.position)
    print("Source:", cube.source)
    print("Visible:", cube.visible)

    print("[4] Creating manipulation runtime...")
    backend = IsaacFrankaBackend(
        world=world,
        robot=robot,
        position_tolerance=0.01,
        orientation_tolerance=0.05,
        max_motion_steps=1000,
    )

    planner = TopDownGraspPlanner(
        approach_height=0.10,
        default_lift_height=0.12,
        grasp_z_offset=0.0,
    )

    registry = SkillRegistry()
    registry.register(MoveToPoseSkill(backend))
    registry.register(HomeSkill(backend))
    registry.register(PickSkill(backend, model, planner))
    registry.register(PlaceSkill(backend, model))
    runtime = RobotRuntime(registry)

    initial = cube.pose.position.copy()

    print("\n[5] PICK using cached WorldModel...")
    pick = runtime.execute("pick", object_id="cube")

    print("Pick:", pick.status, "-", pick.message)
    print("Held:", model.held_object_id)
    print("Cached after pick:", cube.pose.position)

    if not pick.ok:
        raise RuntimeError(f"Pick failed: {pick.message}")

    cached_lift = float(cube.pose.position[2] - initial[2])
    print("Cached lift:", f"{cached_lift:.4f} m")

    print("\n[6] PLACE using cached WorldModel...")
    place = runtime.execute(
        "place",
        target=Pose(TARGET),
        mode="stable",
    )

    print("Place:", place.status, "-", place.message)
    print("Held:", model.held_object_id)
    print("Cached final:", cube.pose.position)

    if not place.ok:
        raise RuntimeError(f"Place failed: {place.message}")

    actual, _ = objects["cube"].get_world_pose()
    cached_error = float(np.linalg.norm(cube.pose.position - actual))
    target_error = float(np.linalg.norm(actual - TARGET))

    print("\n=== CACHED WORLD MODEL PICK & PLACE ===")
    print("WorldModel provider attached: NO")
    print("Updater running: YES")
    print("Pick: SUCCESS")
    print("Place: SUCCESS")
    print("Cached vs actual:", f"{cached_error * 1000:.4f} mm")
    print("True target error:", f"{target_error * 1000:.2f} mm")
    print("PASS")
    print("=======================================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()