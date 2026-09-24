from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import omni.kit.app
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka

from backends.isaac.franka_backend import IsaacFrankaBackend
from backends.isaac.ground_truth_provider import IsaacGroundTruthProvider
from manipulation.grasp_planner import TopDownGraspPlanner
from runtime.robot_runtime import RobotRuntime
from runtime.scene_config import SceneConfig, load_scene_config
from runtime.skill_registry import SkillRegistry
from skills.home import HomeSkill
from skills.move_to_pose import MoveToPoseSkill
from skills.pick import PickSkill
from skills.place import PlaceSkill
from world_model.updater import WorldModelUpdater
from world_model.world_model import WorldModel


@dataclass
class IsaacRuntimeBundle:
    world: World
    backend: IsaacFrankaBackend
    world_model: WorldModel
    world_updater: WorldModelUpdater
    runtime: RobotRuntime
    objects: dict
    config: SceneConfig


def _validate_stage(config: SceneConfig):
    stage = omni.usd.get_context().get_stage()
    paths = [config.robot.prim_path, config.world.objects_root]
    paths += [obj.prim_path for obj in config.objects.values()]

    missing = [p for p in paths if not stage.GetPrimAtPath(p).IsValid()]
    if missing:
        raise RuntimeError(f"Missing prims in current stage: {missing}")


def _create_robot(world: World, config: SceneConfig):
    cfg = config.robot

    if cfg.type != "franka":
        raise RuntimeError(f"Unsupported robot type: {cfg.type}")

    robot = world.scene.get_object(cfg.id)

    if robot is None:
        robot = world.scene.add(
            Franka(
                prim_path=cfg.prim_path,
                name=cfg.id,
            )
        )

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

    for _ in range(30):
        await omni.kit.app.get_app().next_update_async()

    provider = IsaacGroundTruthProvider(
        config.world.objects_root,
        config.objects,
    )
    model = WorldModel()
    updater = WorldModelUpdater(model, provider)

    updater.update()
    updater.start(world)

    backend = IsaacFrankaBackend(
        world=world,
        robot=robot,
        position_tolerance=0.01,
        orientation_tolerance=0.05,
        joint_tolerance=0.02,
        max_motion_steps=1000,
        max_home_steps=1000,
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

    return IsaacRuntimeBundle(
        world=world,
        backend=backend,
        world_model=model,
        world_updater=updater,
        runtime=RobotRuntime(registry),
        objects=objects,
        config=config,
    )