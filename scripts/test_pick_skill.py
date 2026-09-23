from pathlib import Path

from isaacsim import SimulationApp


# ---------------------------------------------------------------------
# Start Isaac first
# ---------------------------------------------------------------------

simulation_app = SimulationApp(
    {
        "headless": False,
    }
)


# ---------------------------------------------------------------------
# Imports after SimulationApp
# ---------------------------------------------------------------------

import numpy as np

from isaacsim.core.api import World

from isaacsim.core.prims import (
    SingleXFormPrim,
)

from isaacsim.core.utils.stage import (
    open_stage,
    is_stage_loading,
)

from isaacsim.robot.manipulators.examples.franka import (
    Franka,
)


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from backends.isaac.franka_backend import (
    IsaacFrankaBackend,
)

from backends.isaac.ground_truth_provider import (
    IsaacGroundTruthProvider,
)

from manipulation.grasp_planner import (
    TopDownGraspPlanner,
)

from skills.pick import (
    PickSkill,
)

from world_model.entities import (
    WorldObject,
)

from world_model.world_model import (
    WorldModel,
)


# ---------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

SCENE_PATH = (
    PROJECT_ROOT
    / "scenes"
    / "playground.usd"
)


print(
    "\n==================================="
)

print(
    "PICK SKILL TEST"
)

print(
    "==================================="
)


print(
    f"\nLoading scene:\n{SCENE_PATH}"
)


if not open_stage(
    str(SCENE_PATH)
):

    simulation_app.close()

    raise RuntimeError(
        f"Could not open scene: "
        f"{SCENE_PATH}"
    )


while is_stage_loading():

    simulation_app.update()


# ---------------------------------------------------------------------
# World
# ---------------------------------------------------------------------

world = World(
    stage_units_in_meters=1.0
)


# ---------------------------------------------------------------------
# Existing Franka
# ---------------------------------------------------------------------

franka = Franka(
    prim_path="/Franka",
    name="franka",
)

world.scene.add(
    franka
)


# ---------------------------------------------------------------------
# Existing cube
# ---------------------------------------------------------------------

cube = SingleXFormPrim(
    prim_path="/World/Cube",
    name="cube",
    reset_xform_properties=False,
)

world.scene.add(
    cube
)


# ---------------------------------------------------------------------
# Initialize physics
# ---------------------------------------------------------------------

world.reset()


# Allow cube/robot to settle.
for _ in range(120):

    world.step(
        render=True
    )


# ---------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------

backend = IsaacFrankaBackend(
    world=world,
    robot=franka,

    position_tolerance=0.01,

    orientation_tolerance=0.05,

    joint_tolerance=0.02,

    max_motion_steps=1000,

    max_home_steps=1000,
)


# ---------------------------------------------------------------------
# Ground-truth world-state provider
#
# This is temporary until perception exists.
# ---------------------------------------------------------------------

ground_truth = (
    IsaacGroundTruthProvider(
        objects={
            "cube": cube,
        }
    )
)


# ---------------------------------------------------------------------
# Semantic world model
# ---------------------------------------------------------------------

world_model = WorldModel(
    state_provider=ground_truth
)


world_model.register(
    WorldObject(
        object_id="cube",

        # Provider will fill this immediately.
        pose=None,

        # Our current test cube is approximately 5 cm.
        size=np.array(
            [
                0.05,
                0.05,
                0.05,
            ]
        ),

        graspable=True,
    )
)


if not world_model.refresh_object(
    "cube"
):

    simulation_app.close()

    raise RuntimeError(
        "Could not obtain initial cube pose."
    )


initial_cube_pose = (
    world_model
    .require("cube")
    .pose
)


print(
    "\nInitial cube position:"
)

print(
    initial_cube_pose.position
)


# ---------------------------------------------------------------------
# Grasp planner
# ---------------------------------------------------------------------

grasp_planner = (
    TopDownGraspPlanner(
        approach_height=0.10,

        default_lift_height=0.12,

        # Start with no extra gripper-frame offset.
        # This is one of the values we may calibrate
        # if the fingers descend too high/low.
        grasp_z_offset=0.0,
    )
)


# ---------------------------------------------------------------------
# Pick skill
# ---------------------------------------------------------------------

pick_skill = PickSkill(
    backend=backend,

    world_model=world_model,

    grasp_planner=grasp_planner,

    grasp_lift_threshold=0.03,

    approach_speed=0.4,

    grasp_speed=0.15,

    lift_speed=0.2,
)


# ---------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------

print(
    "\n==================================="
)

print(
    "Executing PickSkill..."
)

print(
    "==================================="
)


result = (
    pick_skill.execute(
        object_id="cube",

        grasp_hint="top",

        lift_height=0.12,
    )
)


# ---------------------------------------------------------------------
# Semantic result
# ---------------------------------------------------------------------

print(
    "\n==================================="
)

print(
    "PICK SKILL RESULT"
)

print(
    "==================================="
)


print(
    f"Status: "
    f"{result.status.value}"
)

print(
    f"Message: "
    f"{result.message}"
)

print(
    f"Failure code: "
    f"{result.failure_code}"
)


if result.details:

    print(
        "\nDetails:"
    )

    for key, value in (
        result.details.items()
    ):

        print(
            f"  {key}: {value}"
        )


# ---------------------------------------------------------------------
# Independent ground-truth verification
# ---------------------------------------------------------------------

final_cube_position, _ = (
    cube.get_world_pose()
)


final_cube_position = (
    np.asarray(
        final_cube_position,
        dtype=float,
    )
)


initial_position = (
    initial_cube_pose
    .position
)


actual_lift = float(
    final_cube_position[2]
    - initial_position[2]
)


print(
    "\n==================================="
)

print(
    "INDEPENDENT VERIFICATION"
)

print(
    "==================================="
)


print(
    f"Initial cube position: "
    f"{initial_position}"
)

print(
    f"Final cube position:   "
    f"{final_cube_position}"
)

print(
    f"Actual lift: "
    f"{actual_lift:.4f} m"
)

print(
    f"World model held object: "
    f"{world_model.held_object_id}"
)


# ---------------------------------------------------------------------
# Overall test result
# ---------------------------------------------------------------------

print(
    "\n==================================="
)


if (
    result.ok
    and actual_lift >= 0.03
    and world_model.held_object_id == "cube"
):

    print(
        "PICK SKILL TEST: SUCCESS"
    )

else:

    print(
        "PICK SKILL TEST: FAILED"
    )


print(
    "==================================="
)


# ---------------------------------------------------------------------
# Leave robot holding cube briefly.
# ---------------------------------------------------------------------

for _ in range(180):

    world.step(
        render=True
    )


simulation_app.close()