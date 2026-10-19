"""Explicit Isaac robot selection; SDK construction order is owned by bootstrap."""
from isaacsim.robot.manipulators.examples.franka import Franka

from robot_skill_stack.integrations.isaac.manipulation.franka_backend import IsaacFrankaBackend
from robot_skill_stack.manipulation.backend import ManipulationBackend
from robot_skill_stack.manipulation.grasping.types import ParallelJawGripperSpec


def create_robot(world, config):
    if config.backend != 'franka':
        raise ValueError(f'Unsupported robot backend: {config.backend}')
    robot = world.scene.get_object('franka')
    if robot is None:
        robot = world.scene.add(Franka(prim_path=config.prim_path, name='franka'))
    elif robot.prim_path != config.prim_path:
        raise RuntimeError('Robot prim changed; restart Isaac before changing robot setups')
    return robot


def create_backend(world, robot, config) -> tuple[ManipulationBackend, ParallelJawGripperSpec]:
    if config.backend != 'franka':
        raise ValueError(f'Unsupported robot backend: {config.backend}')
    backend = IsaacFrankaBackend(
        world=world, robot=robot, position_tolerance=.01,
        orientation_tolerance=.05, max_motion_steps=1000,
    )
    # Franka geometry belongs to its adapter construction, not scene tuning.
    gripper = ParallelJawGripperSpec(max_width=backend.grasp_max_width,
                                    min_width=backend.grasp_min_width, open_width=.080)
    return backend, gripper
