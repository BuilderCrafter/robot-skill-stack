"""Explicit Isaac robot selection; SDK construction order is owned by bootstrap."""
from isaacsim.robot.manipulators.examples.franka import Franka

from robot_skill_stack.integrations.isaac.config import MotionConfig
from robot_skill_stack.integrations.isaac.manipulation.franka_backend import IsaacFrankaBackend
from robot_skill_stack.manipulation.backend import ManipulationBackend
from robot_skill_stack.manipulation.grasping.types import ParallelJawGripperSpec
from robot_skill_stack.manipulation.motion import MappingSourceProvider


def create_robot(world, config):
    if config.backend != 'franka':
        raise ValueError(f'Unsupported robot backend: {config.backend}')
    robot = world.scene.get_object('franka')
    if robot is None:
        robot = world.scene.add(Franka(prim_path=config.prim_path, name='franka'))
    elif robot.prim_path != config.prim_path:
        raise RuntimeError('Robot prim changed; restart Isaac before changing robot setups')
    return robot


def create_backend(world, robot, config, *, motion: MotionConfig | None = None,
                   mapping_source_provider: MappingSourceProvider | None = None
                   ) -> tuple[ManipulationBackend, ParallelJawGripperSpec]:
    if config.backend != 'franka':
        raise ValueError(f'Unsupported robot backend: {config.backend}')
    # Validate before constructing SDK resources, including programmatic callers.
    selected = MotionConfig(strategy='direct' if motion is None else motion.strategy,
        max_obstacles=64 if motion is None else getattr(motion, 'max_obstacles', 64),
        rrt_seed=123456 if motion is None else getattr(motion, 'rrt_seed', 123456),
        rrt_max_iterations=10000 if motion is None else getattr(motion, 'rrt_max_iterations', 10000),
        global_collision_model='stock' if motion is None else getattr(motion, 'global_collision_model', 'stock'),
        global_obstacle_source='static' if motion is None else getattr(motion, 'global_obstacle_source', 'static'))
    if selected.strategy == 'reactive' and not callable(mapping_source_provider):
        raise ValueError('Reactive motion requires enabled Spatial Mapping access')
    if (selected.strategy == 'global' and selected.global_obstacle_source == 'frozen_rgbd'
            and not callable(mapping_source_provider)):
        raise ValueError('Frozen RGB-D global motion requires enabled Spatial Mapping access')

    def strategy_factory(backend):
        if selected.strategy == 'global':
            from .motion.global_rrt import GlobalRrtStrategy
            model = ({} if selected.global_collision_model == 'stock' else
                dict(collision_model=selected.global_collision_model))
            if selected.global_obstacle_source == 'frozen_rgbd':
                model.update(obstacle_source=selected.global_obstacle_source,
                    mapping_source_provider=mapping_source_provider, max_obstacles=selected.max_obstacles)
            return GlobalRrtStrategy(backend, seed=selected.rrt_seed,
                max_iterations=selected.rrt_max_iterations, **model)
        if selected.strategy == 'reactive':
            from .motion.reactive import ReactiveRmpflowStrategy
            return ReactiveRmpflowStrategy(backend,
                mapping_source_provider=mapping_source_provider,
                max_obstacles=selected.max_obstacles)
        from .motion.direct import DirectRmpflowStrategy
        return DirectRmpflowStrategy(backend,
            mapping_source_provider=mapping_source_provider)

    backend = IsaacFrankaBackend(
        world=world, robot=robot, position_tolerance=.01,
        orientation_tolerance=.05, max_motion_steps=1000,
        strategy_factory=strategy_factory,
    )
    # Franka geometry belongs to its adapter construction, not scene tuning.
    gripper = ParallelJawGripperSpec(max_width=backend.grasp_max_width,
                                    min_width=backend.grasp_min_width, open_width=.080)
    return backend, gripper
