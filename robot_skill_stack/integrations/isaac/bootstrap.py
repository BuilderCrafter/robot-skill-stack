from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from functools import partial

import numpy as np
import omni.kit.app
import omni.usd
from isaacsim.core.api import World

from robot_skill_stack.integrations.isaac.manipulation.factory import create_robot, create_backend
from robot_skill_stack.manipulation.backend import ManipulationBackend
from robot_skill_stack.manipulation.grasping import SimplePrimitiveGraspPlanner
from robot_skill_stack.manipulation.placement import SimplePlacementPlanner
from robot_skill_stack.integrations.isaac.world.factory import create_source, required_source_prims
from robot_skill_stack.runtime.runtime import RobotRuntime
from robot_skill_stack.integrations.isaac.config import SceneConfig, load_scene_config
from robot_skill_stack.runtime.registry import SkillRegistry
from robot_skill_stack.manipulation.skills.home import HomeSkill
from robot_skill_stack.manipulation.skills.move_to_pose import MoveToPoseSkill
from robot_skill_stack.manipulation.skills.pick import PickSkill
from robot_skill_stack.manipulation.skills.place import PlaceSkill
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.perception.diagnostics import save_capture
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.integrations.isaac.runtime_lifecycle import PhysicsCallbackScope, construct_world_deferred


@dataclass
class IsaacRuntimeBundle:
    world: World
    backend: ManipulationBackend
    world_model: WorldModel
    world_updater: WorldModelUpdater
    runtime: RobotRuntime
    config: SceneConfig
    capture_frame: Callable[[Path], Path] | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.world_updater.stop(self.world)
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
    paths = [config.robot.prim_path, *required_source_prims(config)]

    missing = [
        path
        for path in paths
        if not stage.GetPrimAtPath(path).IsValid()
    ]
    if missing:
        raise RuntimeError(f"Missing prims in current stage: {missing}")


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
          f'provider={config.world.provider} perception=geometry_v1 '
          f'configured_objects={list(config.ground_truth.objects)}', flush=True)
    robot = create_robot(world, config.robot)

    await world.reset_async()
    if not world.is_playing():
        await world.play_async()

    app = omni.kit.app.get_app()
    for _ in range(30):
        await app.next_update_async()

    provider = await create_source(config)
    model = WorldModel()
    try:
        updater = WorldModelUpdater(
            model,
            provider,
            update_hz=_update_hz(config),
            stale_object_ttl=config.world.stale_object_ttl,
        )
    except BaseException:
        provider.close()
        raise
    try:
        backend, gripper = create_backend(world, robot, config.robot)
        planner = SimplePrimitiveGraspPlanner(
            gripper=gripper,
            support_plane_z=config.scene.support_plane_z,
            world_model=model,
            current_pose=backend.get_end_effector_pose,
            pose_reachable=backend.check_reachability,
        )

        registry = SkillRegistry()
        registry.register(MoveToPoseSkill(backend, model))
        registry.register(HomeSkill(backend, model))
        placement_planner = SimplePlacementPlanner(clearance=0.005)

        registry.register(PickSkill(backend, model, planner))
        registry.register(PlaceSkill(backend, model, placement_planner))

        bundle = IsaacRuntimeBundle(
            world=world,
            backend=backend,
            world_model=model,
            world_updater=updater,
            runtime=RobotRuntime(registry, model),
            config=config,
            capture_frame=partial(save_capture, provider) if config.world.provider == "perception" else None,
        )
        # Register only after every construction step succeeds.
        updater.update()
        updater.start(PhysicsCallbackScope(world))
        world._robot_skill_stack_bundle = bundle
        print(f'[robot_skill_stack.runtime] ready: objects={len(model.objects())}', flush=True)
        return bundle
    except BaseException:
        updater.stop(world)
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
