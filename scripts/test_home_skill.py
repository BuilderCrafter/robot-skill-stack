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

from skills.home import (
    HomeSkill,
)

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
    "HOME SKILL TEST"
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
# Initialize physics
# ---------------------------------------------------------------------

world.reset()


for _ in range(30):
    world.step(
        render=True
    )


# ---------------------------------------------------------------------
# IMPORTANT:
#
# Create the backend AFTER world.reset().
#
# The backend records the current joint configuration as "home".
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
# Skills
# ---------------------------------------------------------------------

move_skill = MoveToPoseSkill(
    backend=backend
)

home_skill = HomeSkill(
    backend=backend
)


# ---------------------------------------------------------------------
# Record the original home configuration independently.
# ---------------------------------------------------------------------

initial_joint_positions = np.asarray(
    franka.get_joint_positions(),
    dtype=float,
)

arm_joint_indices = np.array(
    [
        franka.get_dof_index(
            f"panda_joint{i}"
        )
        for i in range(
            1,
            8,
        )
    ],
    dtype=np.int64,
)

initial_arm_positions = (
    initial_joint_positions[
        arm_joint_indices
    ].copy()
)


initial_ee_pose = (
    backend.get_end_effector_pose()
)


print(
    "\nInitial/home arm joints:"
)

print(
    initial_arm_positions
)


print(
    "\nInitial end-effector position:"
)

print(
    initial_ee_pose.position
)


# ---------------------------------------------------------------------
# First move away from home.
# ---------------------------------------------------------------------

away_pose = Pose(

    position=(
        initial_ee_pose.position
        + np.array(
            [
                0.20,
                0.10,
                -0.10,
            ]
        )
    ),

    orientation=(
        initial_ee_pose.orientation.copy()
    ),

    frame="world",
)


print(
    "\nMoving robot away from home..."
)


move_result = (
    move_skill.execute(
        target=away_pose,
        speed=0.5,
    )
)


print(
    "\nMove result:"
)

print(
    move_result
)


if not move_result.ok:

    print(
        "\nTEST ABORTED:"
        "\nCould not move robot away from home."
    )

    simulation_app.close()

    raise SystemExit(1)


# ---------------------------------------------------------------------
# Confirm we really moved away.
# ---------------------------------------------------------------------

away_joint_positions = np.asarray(
    franka.get_joint_positions(),
    dtype=float,
)

away_arm_positions = (
    away_joint_positions[
        arm_joint_indices
    ]
)


distance_from_home_before = float(
    np.max(
        np.abs(
            away_arm_positions
            - initial_arm_positions
        )
    )
)


print(
    "\nMaximum joint difference "
    "from home after MoveToPose:"
)

print(
    f"{distance_from_home_before:.4f} rad"
)


# ---------------------------------------------------------------------
# Execute HomeSkill.
# ---------------------------------------------------------------------

print(
    "\n==================================="
)

print(
    "Executing HomeSkill..."
)

print(
    "==================================="
)


home_result = (
    home_skill.execute(
        name="home"
    )
)


# ---------------------------------------------------------------------
# Semantic result
# ---------------------------------------------------------------------

print(
    "\n==================================="
)

print(
    "HOME SKILL RESULT"
)

print(
    "==================================="
)


print(
    f"Status: "
    f"{home_result.status.value}"
)

print(
    f"Message: "
    f"{home_result.message}"
)

print(
    f"Failure code: "
    f"{home_result.failure_code}"
)


if home_result.details:

    print(
        "\nDetails:"
    )

    for key, value in (
        home_result.details.items()
    ):

        print(
            f"  {key}: {value}"
        )


# ---------------------------------------------------------------------
# Independent joint-space verification.
# ---------------------------------------------------------------------

final_joint_positions = np.asarray(
    franka.get_joint_positions(),
    dtype=float,
)

final_arm_positions = (
    final_joint_positions[
        arm_joint_indices
    ]
)


joint_errors = (
    final_arm_positions
    - initial_arm_positions
)


max_joint_error = float(
    np.max(
        np.abs(
            joint_errors
        )
    )
)


# ---------------------------------------------------------------------
# Also verify end-effector returned approximately home.
# ---------------------------------------------------------------------

final_ee_pose = (
    backend.get_end_effector_pose()
)


ee_position_error = float(
    np.linalg.norm(
        final_ee_pose.position
        - initial_ee_pose.position
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
    "\nInitial home joints:"
)

print(
    initial_arm_positions
)


print(
    "\nFinal arm joints:"
)

print(
    final_arm_positions
)


print(
    f"\nMax joint error: "
    f"{max_joint_error:.6f} rad"
)


print(
    f"End-effector position error: "
    f"{ee_position_error:.6f} m"
)


# ---------------------------------------------------------------------
# Overall test result.
# ---------------------------------------------------------------------

joint_test_ok = (
    max_joint_error
    <= 0.03
)

ee_test_ok = (
    ee_position_error
    <= 0.02
)


print(
    "\n==================================="
)


if (
    home_result.ok
    and joint_test_ok
    and ee_test_ok
):

    print(
        "HOME SKILL TEST: SUCCESS"
    )

else:

    print(
        "HOME SKILL TEST: FAILED"
    )


print(
    "==================================="
)


# ---------------------------------------------------------------------
# Keep simulation visible briefly.
# ---------------------------------------------------------------------

for _ in range(180):

    world.step(
        render=True
    )


simulation_app.close()