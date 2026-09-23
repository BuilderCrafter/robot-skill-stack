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
# Isaac imports
# ---------------------------------------------------------------------

import numpy as np

from isaacsim.core.api import World

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

from core.types import Pose

from skills.move_to_pose import (
    MoveToPoseSkill,
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
    "MOVE TO POSE SKILL TEST"
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
# Initialize simulation
# ---------------------------------------------------------------------

world.reset()


for _ in range(30):
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

    max_motion_steps=1000,
)


# ---------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------

move_skill = MoveToPoseSkill(
    backend=backend
)


# ---------------------------------------------------------------------
# Current pose
# ---------------------------------------------------------------------

initial_pose = (
    backend.get_end_effector_pose()
)


print(
    "\nInitial pose:"
)

print(
    f"  position:    "
    f"{initial_pose.position}"
)

print(
    f"  orientation: "
    f"{initial_pose.orientation}"
)


# ---------------------------------------------------------------------
# Target
#
# Keep current orientation and move by a modest offset.
# ---------------------------------------------------------------------

target_pose = Pose(

    position=(
        initial_pose.position
        + np.array(
            [
                0.20,
                0.10,
                -0.10,
            ]
        )
    ),

    orientation=(
        initial_pose.orientation.copy()
    ),

    frame="world",
)


print(
    "\nTarget pose:"
)

print(
    f"  position:    "
    f"{target_pose.position}"
)

print(
    f"  orientation: "
    f"{target_pose.orientation}"
)


# ---------------------------------------------------------------------
# Execute semantic skill
# ---------------------------------------------------------------------

print(
    "\nExecuting MoveToPoseSkill..."
)


result = (
    move_skill.execute(
        target=target_pose,
        speed=0.5,
    )
)


# ---------------------------------------------------------------------
# Report semantic result
# ---------------------------------------------------------------------

print(
    "\n==================================="
)
print(
    "SKILL RESULT"
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
# Independent verification
# ---------------------------------------------------------------------

final_pose = (
    backend.get_end_effector_pose()
)


position_error = float(
    np.linalg.norm(
        final_pose.position
        - target_pose.position
    )
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
    f"Final position: "
    f"{final_pose.position}"
)

print(
    f"Target position: "
    f"{target_pose.position}"
)

print(
    f"Position error: "
    f"{position_error:.4f} m"
)


# ---------------------------------------------------------------------
# Overall test
# ---------------------------------------------------------------------

print(
    "\n==================================="
)

if result.ok:

    print(
        "MOVE TO POSE SKILL: SUCCESS"
    )

else:

    print(
        "MOVE TO POSE SKILL: FAILED"
    )

print(
    "==================================="
)


# ---------------------------------------------------------------------
# Keep simulation visible briefly
# ---------------------------------------------------------------------

for _ in range(180):
    world.step(
        render=True
    )


simulation_app.close()