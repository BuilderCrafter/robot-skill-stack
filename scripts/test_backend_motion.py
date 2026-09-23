from pathlib import Path

from isaacsim import SimulationApp


# ---------------------------------------------------------------------
# Start Isaac before importing Isaac modules or our Isaac backend.
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
#
# IsaacFrankaBackend itself imports Isaac modules, so it must also
# come after SimulationApp startup.
# ---------------------------------------------------------------------

from backends.isaac.franka_backend import (
    IsaacFrankaBackend,
)

from core.types import Pose


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
    "BACKEND MOTION TEST"
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
# Existing Franka from playground.usd
# ---------------------------------------------------------------------

franka = Franka(
    prim_path="/Franka",
    name="franka",
)

world.scene.add(
    franka
)


# ---------------------------------------------------------------------
# Initialize physics/articulation
# ---------------------------------------------------------------------

world.reset()


# Give simulation a few frames to settle.
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

    # Approximately 2.9 degrees.
    orientation_tolerance=0.05,

    max_motion_steps=1000,
)


# ---------------------------------------------------------------------
# Read initial end-effector pose
# ---------------------------------------------------------------------

initial_pose = (
    backend.get_end_effector_pose()
)


print(
    "\nInitial right_gripper pose:"
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
# Create a safe test target.
#
# Rather than inventing a completely unrelated orientation,
# keep the current orientation and move the gripper:
#
#   +25 cm X
#   0 cm Y
#   -15 cm Z
#
# This should produce a visible but relatively safe first motion.
# ---------------------------------------------------------------------

target_pose = Pose(

    position=(
        initial_pose.position
        + np.array(
            [
                0.25,
                0.00,
                -0.15,
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
# Reachability
# ---------------------------------------------------------------------

reachable = (
    backend.check_reachability(
        target_pose
    )
)


print(
    f"\nReachable: {reachable}"
)


if not reachable:

    print(
        "\nTEST FAILED:"
        "\nTarget was not reachable."
    )

    simulation_app.close()

    raise SystemExit(1)


# ---------------------------------------------------------------------
# Gripper primitive test
# ---------------------------------------------------------------------

print(
    "\nOpening gripper..."
)

open_result = (
    backend.open_gripper()
)

print(
    open_result
)


print(
    "\nClosing gripper..."
)

close_result = (
    backend.close_gripper()
)

print(
    close_result
)


print(
    "\nOpening gripper again..."
)

open_result = (
    backend.open_gripper()
)

print(
    open_result
)


# ---------------------------------------------------------------------
# MoveToPose backend primitive
# ---------------------------------------------------------------------

print(
    "\nMoving to target..."
)

motion_result = (
    backend.move_to_pose(
        target=target_pose,
        speed=0.5,
    )
)


print(
    "\nMotion result:"
)

print(
    motion_result
)


# ---------------------------------------------------------------------
# Final pose
# ---------------------------------------------------------------------

final_pose = (
    backend.get_end_effector_pose()
)


print(
    "\nFinal right_gripper pose:"
)

print(
    f"  position:    "
    f"{final_pose.position}"
)

print(
    f"  orientation: "
    f"{final_pose.orientation}"
)


position_error = float(
    np.linalg.norm(
        final_pose.position
        - target_pose.position
    )
)


print(
    f"\nIndependent position error: "
    f"{position_error:.4f} m"
)


# ---------------------------------------------------------------------
# Overall result
# ---------------------------------------------------------------------

print(
    "\n==================================="
)

if motion_result.ok:

    print(
        "BACKEND TEST: SUCCESS"
    )

else:

    print(
        "BACKEND TEST: FAILED"
    )

print(
    "==================================="
)


# ---------------------------------------------------------------------
# Keep the result visible briefly.
# ---------------------------------------------------------------------

for _ in range(180):
    world.step(
        render=True
    )


simulation_app.close()