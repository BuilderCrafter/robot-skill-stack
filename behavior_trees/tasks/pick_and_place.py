import py_trees

from behavior_trees.nodes.skill_node import SkillNode
from core.types import Pose
from runtime.robot_runtime import RobotRuntime


def create_pick_and_place_tree(
    runtime: RobotRuntime,
    object_id: str,
    target: Pose,
    return_home: bool = True,
):
    children = [
        SkillNode(
            f"Pick {object_id}",
            runtime,
            "pick",
            object_id=object_id,
            grasp_hint="top",
        ),
        SkillNode(
            f"Place {object_id}",
            runtime,
            "place",
            target=target,
            mode="stable",
        ),
    ]

    if return_home:
        children.append(
            SkillNode(
                "Home",
                runtime,
                "home",
            )
        )

    return py_trees.composites.Sequence(
        name="Pick & Place",
        memory=True,
        children=children,
    )