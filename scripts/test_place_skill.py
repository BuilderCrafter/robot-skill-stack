from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import open_stage, is_stage_loading
from isaacsim.robot.manipulators.examples.franka import Franka

from backends.isaac.franka_backend import IsaacFrankaBackend
from backends.isaac.ground_truth_provider import IsaacGroundTruthProvider
from core.types import Pose
from manipulation.grasp_planner import TopDownGraspPlanner
from skills.pick import PickSkill
from skills.place import PlaceSkill
from world_model.entities import WorldObject
from world_model.world_model import WorldModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = PROJECT_ROOT / "scenes" / "playground.usd"

print("\n=== PLACE SKILL TEST ===")
print(f"Loading scene:\n{SCENE_PATH}")

if not open_stage(str(SCENE_PATH)):
    simulation_app.close()
    raise RuntimeError(f"Could not open scene: {SCENE_PATH}")

while is_stage_loading():
    simulation_app.update()

world = World(stage_units_in_meters=1.0)

franka = Franka(prim_path="/Franka", name="franka")
cube = SingleXFormPrim(
    prim_path="/World/Cube",
    name="cube",
    reset_xform_properties=False,
)

world.scene.add(franka)
world.scene.add(cube)
world.reset()

for _ in range(120):
    world.step(render=True)

backend = IsaacFrankaBackend(
    world=world,
    robot=franka,
    position_tolerance=0.01,
    orientation_tolerance=0.05,
    joint_tolerance=0.02,
    max_motion_steps=1000,
    max_home_steps=1000,
)

ground_truth = IsaacGroundTruthProvider({"cube": cube})
world_model = WorldModel(state_provider=ground_truth)

world_model.register(
    WorldObject(
        object_id="cube",
        size=np.array([0.05, 0.05, 0.05]),
        graspable=True,
    )
)

if not world_model.refresh_object("cube"):
    simulation_app.close()
    raise RuntimeError("Could not obtain cube pose.")

initial_position = world_model.require("cube").pose.position.copy()

# Put the cube 25 cm in +Y, at the same table height.
target = Pose(
    position=initial_position + np.array([0.0, 0.25, 0.0]),
    orientation=None,
    frame="world",
)

grasp_planner = TopDownGraspPlanner(
    approach_height=0.10,
    default_lift_height=0.12,
    grasp_z_offset=0.0,
)

pick_skill = PickSkill(
    backend=backend,
    world_model=world_model,
    grasp_planner=grasp_planner,
)

place_skill = PlaceSkill(
    backend=backend,
    world_model=world_model,
    approach_height=0.10,
    retreat_height=0.12,
    placement_tolerance=0.05,
)

print(f"\nInitial cube position: {initial_position}")
print(f"Place target:          {target.position}")

print("\n--- PICK ---")

pick_result = pick_skill.execute(
    object_id="cube",
    grasp_hint="top",
    lift_height=0.12,
)

print(f"Status:  {pick_result.status.value}")
print(f"Message: {pick_result.message}")
print(f"Failure: {pick_result.failure_code}")

if not pick_result.ok:
    print("\nPLACE SKILL TEST: ABORTED — PICK FAILED")

    for _ in range(180):
        world.step(render=True)

    simulation_app.close()
    raise SystemExit(1)

print(f"Held object: {world_model.held_object_id}")

print("\n--- PLACE ---")

place_result = place_skill.execute(target)

print(f"Status:  {place_result.status.value}")
print(f"Message: {place_result.message}")
print(f"Failure: {place_result.failure_code}")

if place_result.details:
    print("Details:")
    for key, value in place_result.details.items():
        print(f"  {key}: {value}")

# Independent verification.
final_position, _ = cube.get_world_pose()
final_position = np.asarray(final_position, dtype=float)

error = float(np.linalg.norm(final_position - target.position))

print("\n--- VERIFICATION ---")
print(f"Initial:      {initial_position}")
print(f"Target:       {target.position}")
print(f"Final:        {final_position}")
print(f"Error:        {error:.4f} m")
print(f"Held object:  {world_model.held_object_id}")

if place_result.ok and error <= 0.05 and world_model.held_object_id is None:
    print("\nPLACE SKILL TEST: SUCCESS")
else:
    print("\nPLACE SKILL TEST: FAILED")

for _ in range(180):
    world.step(render=True)

simulation_app.close()