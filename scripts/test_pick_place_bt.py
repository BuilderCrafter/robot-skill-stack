from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".deps"))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import py_trees

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from isaacsim.robot.manipulators.examples.franka import Franka

from backends.isaac.franka_backend import IsaacFrankaBackend
from backends.isaac.ground_truth_provider import IsaacGroundTruthProvider
from behavior_trees.executor import BTExecutor
from behavior_trees.tasks.pick_and_place import create_pick_and_place_tree
from core.types import Pose
from manipulation.grasp_planner import TopDownGraspPlanner
from runtime.robot_runtime import RobotRuntime
from runtime.skill_registry import SkillRegistry
from skills.home import HomeSkill
from skills.move_to_pose import MoveToPoseSkill
from skills.pick import PickSkill
from skills.place import PlaceSkill
from world_model.entities import WorldObject
from world_model.world_model import WorldModel


SCENE = ROOT / "scenes" / "playground.usd"

if not open_stage(str(SCENE)):
    raise RuntimeError(f"Could not open {SCENE}")

while is_stage_loading():
    simulation_app.update()

world = World(stage_units_in_meters=1.0)

franka = Franka("/Franka", name="franka")
cube = SingleXFormPrim("/World/Cube", name="cube", reset_xform_properties=False)

world.scene.add(franka)
world.scene.add(cube)
world.reset()

for _ in range(120):
    world.step(render=True)

backend = IsaacFrankaBackend(world, franka)

provider = IsaacGroundTruthProvider({"cube": cube})
model = WorldModel(provider)

model.register(
    WorldObject(
        "cube",
        size=np.array([0.05, 0.05, 0.05]),
        graspable=True,
    )
)
model.refresh_all()

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

runtime = RobotRuntime(registry)

start = model.require("cube").pose.position.copy()
target = Pose(start + np.array([0.0, 0.25, 0.0]))

root = create_pick_and_place_tree(
    runtime,
    object_id="cube",
    target=target,
    return_home=True,
)

print("\nInitial tree:")
print(py_trees.display.unicode_tree(root))

status = BTExecutor(root).run()

print("\n============================")
print(f"BT RESULT: {status.name}")
print("============================")

model.refresh_object("cube")
final = model.require("cube").pose.position

print(f"Start:  {start}")
print(f"Target: {target.position}")
print(f"Final:  {final}")
print(f"Error:  {np.linalg.norm(final - target.position):.4f} m")

for _ in range(180):
    world.step(render=True)

simulation_app.close()