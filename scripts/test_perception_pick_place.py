from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
import traceback
from pathlib import Path

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from isaacsim.robot.manipulators.examples.franka import Franka

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends.isaac.franka_backend import IsaacFrankaBackend
from backends.isaac.rgbd_camera import IsaacRgbdCamera
from backends.isaac.semantic_detector import IsaacSemanticDetector
from core.types import Pose
from manipulation.grasp_planner import TopDownGraspPlanner
from perception.rgbd_localizer import RgbdLocalizer
from perception.state_provider import PerceptionStateProvider
from runtime.robot_runtime import RobotRuntime
from runtime.scene_config import load_scene_config
from runtime.skill_registry import SkillRegistry
from skills.home import HomeSkill
from skills.move_to_pose import MoveToPoseSkill
from skills.pick import PickSkill
from skills.place import PlaceSkill
from world_model.entities import WorldObject
from world_model.world_model import WorldModel

SCENE = ROOT / "scenes" / "playground.usd"
PROFILE = ROOT / "config" / "scenes" / "playground.toml"
PLACE_TARGET = np.array([0.4465, 0.25, 0.025])


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

    print("[3] Creating RGB-D perception...")
    camera = IsaacRgbdCamera(
        config.perception.camera_prim_path,
        resolution=config.perception.resolution,
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

    provider = PerceptionStateProvider(
        camera,
        detector,
        RgbdLocalizer(camera.get_intrinsics()),
        {
            object_id: cfg.size
            for object_id, cfg in config.objects.items()
            if cfg.size is not None
        },
    )

    print("[4] Creating runtime...")
    backend = IsaacFrankaBackend(
        world=world,
        robot=robot,
        position_tolerance=0.01,
        orientation_tolerance=0.05,
        max_motion_steps=1000,
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

    print("[5] Perceiving cube...")
    if not model.refresh_object("cube"):
        raise RuntimeError("Could not perceive cube")

    perceived = model.require("cube").pose.position.copy()
    actual, _ = objects["cube"].get_world_pose()

    print("Perceived:", np.round(perceived, 6))
    print("Ground truth:", np.round(actual, 6))
    print(
        "Localization error:",
        f"{np.linalg.norm(perceived - actual) * 1000:.2f} mm",
    )

    print("\n[6] PICK...\n")
    pick = runtime.execute("pick", object_id="cube")

    print("Pick status:", pick.status)
    print("Pick message:", pick.message)
    print("Pick failure:", pick.failure_code)
    print("Held:", model.held_object_id)

    if not pick.ok:
        raise RuntimeError(f"Pick failed: {pick.message}")

    print("\n[7] STABLE PLACE...\n")
    place = runtime.execute(
        "place",
        target=Pose(PLACE_TARGET),
        mode="stable",
    )

    print("Place status:", place.status)
    print("Place message:", place.message)
    print("Place failure:", place.failure_code)
    print("Place details:", place.details)
    print("Held:", model.held_object_id)

    final_true, _ = objects["cube"].get_world_pose()
    final_perceived = model.require("cube").pose.position.copy()

    print("\n=== PERCEPTION PICK & PLACE ===")
    print("Requested target: ", PLACE_TARGET)
    print("Perceived final:  ", np.round(final_perceived, 6))
    print("Ground truth final:", np.round(final_true, 6))
    print(
        "Perceived target error:",
        f"{np.linalg.norm(final_perceived - PLACE_TARGET) * 1000:.2f} mm",
    )
    print(
        "True target error:     ",
        f"{np.linalg.norm(final_true - PLACE_TARGET) * 1000:.2f} mm",
    )
    print("Result:", "PASS" if pick.ok and place.ok else "FAIL")
    print("===============================")

except Exception:
    print("\n!!! TEST FAILED !!!")
    traceback.print_exc()

finally:
    simulation_app.close()