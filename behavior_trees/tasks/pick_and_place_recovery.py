import py_trees

from behavior_trees.nodes.skill_node import SkillNode
from core.types import Pose
from runtime.robot_runtime import RobotRuntime


def create_pick_and_place_recovery_tree(
    runtime: RobotRuntime,
    object_id: str,
    target: Pose,
    return_home: bool = True,
):
    first_pick = SkillNode(
        f"Pick {object_id}",
        runtime,
        "pick",
        object_id=object_id,
        grasp_hint="top",
    )

    retry_pick = SkillNode(
        f"Retry Pick {object_id}",
        runtime,
        "pick",
        object_id=object_id,
        grasp_hint="top",
    )

    recovery = py_trees.composites.Sequence(
        name="Recover & Retry",
        memory=True,
        children=[
            SkillNode("Recovery Home", runtime, "home"),
            retry_pick,
        ],
    )

    pick_with_recovery = py_trees.composites.Selector(
        name="Pick With Recovery",
        memory=False,
        children=[
            first_pick,
            recovery,
        ],
    )

    children = [
        pick_with_recovery,
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
            SkillNode("Final Home", runtime, "home")
        )

    root = py_trees.composites.Sequence(
        name="Pick & Place With Recovery",
        memory=True,
        children=children,
    )

    return root