from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka

from robot_skill_stack.integrations.isaac.manipulation.franka_backend import IsaacFrankaBackend
from robot_skill_stack.integrations.isaac.world.ground_truth_provider import IsaacGroundTruthProvider
from robot_skill_stack.integrations.isaac.sensors.rgbd_camera import IsaacRgbdCamera
from robot_skill_stack.manipulation.grasping import (
    SimplePrimitiveGraspPlanner,
)
from robot_skill_stack.manipulation.placement import SimplePlacementPlanner
from robot_skill_stack.world.perception.factory import build_perception_provider
from robot_skill_stack.runtime.runtime import RobotRuntime
from robot_skill_stack.integrations.isaac.config import SceneConfig, load_scene_config
from robot_skill_stack.runtime.registry import SkillRegistry
from robot_skill_stack.manipulation.skills.home import HomeSkill
from robot_skill_stack.manipulation.skills.move_to_pose import MoveToPoseSkill
from robot_skill_stack.manipulation.skills.pick import PickSkill
from robot_skill_stack.manipulation.skills.place import PlaceSkill
from robot_skill_stack.world.model.provider import WorldObservationProvider
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.integrations.isaac.runtime_lifecycle import PhysicsCallbackScope, construct_world_deferred


@dataclass
class IsaacRuntimeBundle:
    world: World
    backend: IsaacFrankaBackend
    world_model: WorldModel
    world_updater: WorldModelUpdater
    runtime: RobotRuntime
    objects: dict
    config: SceneConfig
    state_provider: WorldObservationProvider
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.world_updater.stop(self.world)
        finally:
            close = getattr(getattr(self.state_provider, 'camera', None), 'close', None)
            try:
                if close is not None:
                    close()
            finally:
                if getattr(self.world, '_robot_skill_stack_bundle', None) is self:
                    self.world._robot_skill_stack_bundle = None


_BUILDING = False


def _validate_environment(config: SceneConfig):
    if config.world.provider == "perception":
        major = int(np.__version__.split(".", 1)[0])
        if major >= 2:
            raise RuntimeError(
                "Perception requires repo-local numpy==1.26.4, but "
                f"NumPy {np.__version__} was loaded from {np.__file__}. "
                "Install .deps and launch Isaac through ./run_isaac_sim.sh."
            )


def _validate_stage(config: SceneConfig):
    stage = omni.usd.get_context().get_stage()
    paths = [
        config.robot.prim_path,
        config.world.objects_root,
        *[obj.prim_path for obj in config.objects.values()],
    ]

    if config.world.provider == "perception":
        if not config.perception.camera_prim_path:
            raise RuntimeError("Perception provider requires camera_prim_path")
        paths.append(config.perception.camera_prim_path)

    missing = [
        path
        for path in paths
        if not stage.GetPrimAtPath(path).IsValid()
    ]
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


async def _create_state_provider(
    world: World,
    config: SceneConfig,
) -> WorldObservationProvider:
    if config.world.provider == "ground_truth":
        return IsaacGroundTruthProvider(
            config.world.objects_root,
            config.objects,
        )

    if config.world.provider != "perception":
        raise RuntimeError(
            f"Unsupported world provider: {config.world.provider}"
        )
    if not config.perception.camera_prim_path:
        raise RuntimeError("Perception provider requires camera_prim_path")

    camera = IsaacRgbdCamera(
        config.perception.camera_prim_path,
        resolution=config.perception.resolution,
    )
    robot_source = None
    try:
        camera.initialize(semantic_segmentation=False)
        if config.perception.type == "yolo_v2" and config.perception.v2.robot_self_filter:
            from robot_skill_stack.integrations.isaac.sensors.robot_self_filter import IsaacRobotSnapshotSource
            robot_source = IsaacRobotSnapshotSource(world, camera, config.robot.prim_path,
                                                   [config.world.objects_root, *(o.prim_path for o in config.objects.values())],
                                                   config.perception.v2.robot_time_tolerance_s)
        app = omni.kit.app.get_app()
        for _ in range(60):
            await app.next_update_async()
        provider = build_perception_provider(camera, config, robot_source=robot_source)
        return provider
    except BaseException:
        if robot_source is not None:
            robot_source.close()
        camera.close()
        raise


def _update_hz(config: SceneConfig):
    if config.world.update_hz is not None:
        return config.world.update_hz
    return 5.0 if config.world.provider == "perception" else None


async def _build_runtime(
    profile_path: str | Path,
) -> IsaacRuntimeBundle:
    config = load_scene_config(profile_path)
    _validate_environment(config)
    _validate_stage(config)

    world = World.instance()
    if world is None:
        world = construct_world_deferred(World, stage_units_in_meters=1.0)
    if world.get_physics_context() is None:
        await world.initialize_simulation_context_async()

    previous = getattr(world, '_robot_skill_stack_bundle', None)
    if previous is not None:
        previous.close()
    print(f'[robot_skill_stack.runtime] profile={Path(profile_path).resolve()} '
          f'provider={config.world.provider} perception={config.perception.type} '
          f'configured_objects={list(config.objects)}', flush=True)
    robot = _create_robot(world, config)
    objects = _create_objects(world, config)

    await world.reset_async()
    if not world.is_playing():
        await world.play_async()

    app = omni.kit.app.get_app()
    for _ in range(30):
        await app.next_update_async()

    provider = await _create_state_provider(world, config)
    model = WorldModel(stale_object_ttl=config.world.stale_object_ttl)
    updater = WorldModelUpdater(
        model,
        provider,
        update_hz=_update_hz(config),
    )
    try:
        backend = IsaacFrankaBackend(
            world=world,
            robot=robot,
            position_tolerance=0.01,
            orientation_tolerance=0.05,
            joint_tolerance=0.02,
            max_motion_steps=1000,
            max_home_steps=1000,
        )

        grasp = config.grasping
        planner = SimplePrimitiveGraspPlanner(
            approach_height=grasp.approach_height,
            default_lift_height=grasp.default_lift_height,
            grasp_z_offset=grasp.grasp_z_offset,
            gripper=replace(
                grasp.gripper,
                max_width=min(grasp.gripper.max_width, backend.grasp_max_width),
                min_width=max(grasp.gripper.min_width, backend.grasp_min_width),
            ),
            safety=grasp.safety,
            support_plane_z=config.perception.discovery.support_plane_z,
            world_model=model,
            current_pose=backend.get_end_effector_pose,
            pose_reachable=backend.check_reachability,
        )

        registry = SkillRegistry()
        registry.register(MoveToPoseSkill(backend))
        registry.register(HomeSkill(backend))
        placement_planner = SimplePlacementPlanner(clearance=0.005)

        registry.register(PickSkill(backend, model, planner))
        registry.register(PlaceSkill(backend, model, placement_planner))

        bundle = IsaacRuntimeBundle(
            world=world,
            backend=backend,
            world_model=model,
            world_updater=updater,
            runtime=RobotRuntime(registry),
            objects=objects,
            config=config,
            state_provider=provider,
        )
        # Register only after every construction step succeeds.
        updater.update()
        updater.start(PhysicsCallbackScope(world))
        world._robot_skill_stack_bundle = bundle
        print(f'[robot_skill_stack.runtime] ready: objects={len(model.objects())}', flush=True)
        return bundle
    except BaseException:
        updater.stop(world)
        close = getattr(getattr(provider, 'camera', None), 'close', None)
        if close is not None:
            close()
        raise


async def build_runtime(profile_path: str | Path) -> IsaacRuntimeBundle:
    global _BUILDING
    if _BUILDING:
        raise RuntimeError('Runtime initialization is already in progress')
    _BUILDING = True
    try:
        # Leave the button/draw callback before initializing simulator resources.
        await omni.kit.app.get_app().next_update_async()
        return await _build_runtime(profile_path)
    finally:
        _BUILDING = False
