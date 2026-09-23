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
from isaacsim.core.utils.stage import open_stage, is_stage_loading
from isaacsim.robot.manipulators.examples.franka import Franka

from backends.isaac.franka_backend import IsaacFrankaBackend
from backends.isaac.ground_truth_provider import IsaacGroundTruthProvider
from behavior_trees.executor import BTExecutor
from behavior_trees.nodes.skill_node import SkillNode
from behavior_trees.nodes.test_nodes import FailOnce
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
cube = SingleXFormPrim(
    "/World/Cube",
    name="cube",
    reset_xform_properties=False,
)

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

# ------------------------------------------------------------------
# Recovery subtree:
#
# forced failure -> Home -> actual Pick
# ------------------------------------------------------------------

recovery_pick = py_trees.composites.Selector(
    name="Pick With Recovery",
    memory=False,
    children=[
        FailOnce("Injected Pick Failure"),
        py_trees.composites.Sequence(
            name="Recover & Retry",
            memory=True,
            children=[
                SkillNode("Recovery Home", runtime, "home"),
                SkillNode(
                    "Retry Pick cube",
                    runtime,
                    "pick",
                    object_id="cube",
                    grasp_hint="top",
                ),
            ],
        ),
    ],
)

root = py_trees.composites.Sequence(
    name="Recovery Test",
    memory=True,
    children=[
        recovery_pick,
        SkillNode(
            "Place cube",
            runtime,
            "place",
            target=target,
            mode="stable",
        ),
        SkillNode("Final Home", runtime, "home"),
    ],
)

print("\nInitial tree:")
print(py_trees.display.unicode_tree(root))

status = BTExecutor(root).run()

model.refresh_object("cube")
final = model.require("cube").pose.position
error = float(np.linalg.norm(final - target.position))

print("\n============================")
print(f"BT RESULT: {status.name}")
print("============================")
print(f"Start:  {start}")
print(f"Target: {target.position}")
print(f"Final:  {final}")
print(f"Error:  {error:.4f} m")

if status == py_trees.common.Status.SUCCESS and error <= 0.05:
    print("\nRECOVERY BT TEST: SUCCESS")
else:
    print("\nRECOVERY BT TEST: FAILED")

for _ in range(180):
    world.step(render=True)

simulation_app.close()