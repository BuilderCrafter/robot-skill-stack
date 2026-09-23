from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka

from backends.isaac.franka_backend import IsaacFrankaBackend
from backends.isaac.ground_truth_provider import IsaacGroundTruthProvider
from backends.isaac.rgbd_camera import IsaacRgbdCamera
from backends.isaac.semantic_detector import IsaacSemanticDetector
from manipulation.grasp_planner import TopDownGraspPlanner
from perception.rgbd_localizer import RgbdLocalizer
from perception.state_provider import PerceptionStateProvider
from runtime.robot_runtime import RobotRuntime
from runtime.scene_config import SceneConfig, load_scene_config
from runtime.skill_registry import SkillRegistry
from skills.home import HomeSkill
from skills.move_to_pose import MoveToPoseSkill
from skills.pick import PickSkill
from skills.place import PlaceSkill
from world_model.entities import WorldObject
from world_model.world_model import WorldModel


@dataclass
class IsaacRuntimeBundle:
    world: World
    backend: IsaacFrankaBackend
    world_model: WorldModel
    runtime: RobotRuntime
    objects: dict
    config: SceneConfig


def _validate_stage(config: SceneConfig):
    stage = omni.usd.get_context().get_stage()
    paths = [config.robot.prim_path]
    paths += [obj.prim_path for obj in config.objects.values()]

    if config.perception.provider == "rgbd":
        if not config.perception.camera_prim_path:
            raise RuntimeError("RGB-D perception requires camera_prim_path")
        paths.append(config.perception.camera_prim_path)

    missing = [p for p in paths if not stage.GetPrimAtPath(p).IsValid()]
    if missing:
        raise RuntimeError(f"Missing prims in current stage: {missing}")


def _create_robot(world: World, config: SceneConfig):
    cfg = config.robot
    if cfg.type != "franka":
        raise RuntimeError(f"Unsupported robot type: {cfg.type}")

    robot = world.scene.get_object(cfg.id)
    if robot is None:
        robot = world.scene.add(Franka(prim_path=cfg.prim_path, name=cfg.id))
    return robot


def _create_objects(world: World, config: SceneConfig):
    objects = {}
    for object_id, cfg in config.objects.items():
        obj = world.scene.get_object(object_id)
        if obj is None:
            obj = world.scene.add(
                SingleXFormPrim(
                    prim_path=cfg.prim_path,
                    name=object_id,
                    reset_xform_properties=False,
                )
            )
        objects[object_id] = obj
    return objects


def _create_state_provider(world, config, objects):
    provider = config.perception.provider

    if provider == "ground_truth":
        return IsaacGroundTruthProvider(objects)

    if provider != "rgbd":
        raise RuntimeError(f"Unsupported perception provider: {provider}")

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

    localizer = RgbdLocalizer(camera.get_intrinsics())

    sizes = {
        object_id: cfg.size
        for object_id, cfg in config.objects.items()
        if cfg.size is not None
    }

    return PerceptionStateProvider(
        camera,
        detector,
        localizer,
        sizes,
    )


async def build_runtime(profile_path: str | Path) -> IsaacRuntimeBundle:
    config = load_scene_config(profile_path)
    _validate_stage(config)

    world = World.instance()
    if world is None:
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()

    robot = _create_robot(world, config)
    objects = _create_objects(world, config)

    await world.reset_async()
    await world.play_async()

    backend = IsaacFrankaBackend(
        world=world,
        robot=robot,
        position_tolerance=0.01,
        orientation_tolerance=0.05,
        joint_tolerance=0.02,
        max_motion_steps=1000,
        max_home_steps=1000,
    )

    provider = _create_state_provider(world, config, objects)
    model = WorldModel(state_provider=provider)

    for object_id, cfg in config.objects.items():
        model.register(
            WorldObject(
                object_id=object_id,
                size=cfg.size,
                graspable=cfg.graspable,
            )
        )

    model.refresh_all()

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

    return IsaacRuntimeBundle(
        world=world,
        backend=backend,
        world_model=model,
        runtime=RobotRuntime(registry),
        objects=objects,
        config=config,
    )