from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from isaacsim.robot.manipulators.examples.franka import Franka

from backends.isaac.franka_backend import IsaacFrankaBackend
from backends.isaac.ground_truth_provider import IsaacGroundTruthProvider
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


ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "scenes" / "playground.usd"

if not open_stage(str(SCENE)):
    simulation_app.close()
    raise RuntimeError(f"Could not open {SCENE}")

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

provider = IsaacGroundTruthProvider({"cube": cube})
world_model = WorldModel(state_provider=provider)
world_model.register(
    WorldObject(
        object_id="cube",
        size=np.array([0.05, 0.05, 0.05]),
        graspable=True,
    )
)
world_model.refresh_all()

grasp_planner = TopDownGraspPlanner(
    approach_height=0.10,
    default_lift_height=0.12,
    grasp_z_offset=0.0,
)

registry = SkillRegistry()
registry.register(MoveToPoseSkill(backend))
registry.register(HomeSkill(backend))
registry.register(PickSkill(backend, world_model, grasp_planner))
registry.register(PlaceSkill(backend, world_model))

runtime = RobotRuntime(registry)

commands: Queue[str] = Queue()


def console_loop():
    while True:
        try:
            commands.put(input("robot> ").strip())
        except EOFError:
            commands.put("quit")
            return


def print_result(result):
    print(f"\n[{result.status.value.upper()}] {result.message}")
    if result.failure_code:
        print(f"Failure: {result.failure_code.value}")
    if result.details:
        for key, value in result.details.items():
            print(f"  {key}: {value}")
    print()


def help_text():
    print(
        """
Commands:
  pick <object>
  place <x> <y> <z>
  move <x> <y> <z>
  home
  status
  objects
  skills
  help
  quit
"""
    )


def handle_command(command: str) -> bool:
    if not command:
        return True

    args = command.split()
    cmd = args[0].lower()

    try:
        if cmd in ("quit", "exit"):
            return False

        if cmd == "help":
            help_text()

        elif cmd == "skills":
            print("Skills:", ", ".join(runtime.available_skills()))

        elif cmd == "objects":
            world_model.refresh_all()
            for object_id in world_model._objects:
                obj = world_model.require(object_id)
                print(f"{object_id}: {None if obj.pose is None else obj.pose.position}")

        elif cmd == "status":
            world_model.refresh_all()
            ee = backend.get_end_effector_pose()
            print(f"EE:   {ee.position}")
            print(f"Held: {world_model.held_object_id}")

        elif cmd == "pick" and len(args) == 2:
            print_result(runtime.execute("pick", object_id=args[1]))

        elif cmd == "place" and len(args) == 4:
            target = Pose(np.array([float(v) for v in args[1:4]]))
            print_result(runtime.execute("place", target=target))

        elif cmd == "move" and len(args) == 4:
            current = backend.get_end_effector_pose()
            target = Pose(
                np.array([float(v) for v in args[1:4]]),
                current.orientation,
            )
            print_result(runtime.execute("move_to_pose", target=target))

        elif cmd == "home":
            print_result(runtime.execute("home"))

        else:
            print("Invalid command. Type 'help'.")

    except ValueError:
        print("Invalid numeric arguments.")

    return True


Thread(target=console_loop, daemon=True).start()

print("\nRobot runtime ready.")
help_text()

running = True

while simulation_app.is_running() and running:
    world.step(render=True)

    try:
        command = commands.get_nowait()
    except Empty:
        continue

    running = handle_command(command)

simulation_app.close()