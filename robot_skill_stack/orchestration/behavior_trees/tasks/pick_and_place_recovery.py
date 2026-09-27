import py_trees

from robot_skill_stack.common.types import Pose
from robot_skill_stack.orchestration.behavior_trees.nodes.skill_node import (
    SkillNode,
)
from robot_skill_stack.runtime.runtime import RobotRuntime


def create_pick_and_place_recovery_tree(
    runtime: RobotRuntime,
    object_id: str,
    target: Pose,
    return_home=True,
    skill_node_cls=SkillNode,
):
    first = skill_node_cls(
        f"Pick {object_id}",
        runtime,
        "pick",
        object_id=object_id,
        grasp_hint="top",
    )
    retry = skill_node_cls(
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
            skill_node_cls("Recovery Home", runtime, "home"),
            retry,
        ],
    )
    selector = py_trees.composites.Selector(
        name="Pick With Recovery",
        memory=False,
        children=[first, recovery],
    )
    children = [
        selector,
        skill_node_cls(
            f"Place {object_id}",
            runtime,
            "place",
            target=target,
            mode="stable",
        ),
    ]
    if return_home:
        children.append(
            skill_node_cls("Final Home", runtime, "home")
        )
    return py_trees.composites.Sequence(
        name="Pick & Place With Recovery",
        memory=True,
        children=children,
    )
