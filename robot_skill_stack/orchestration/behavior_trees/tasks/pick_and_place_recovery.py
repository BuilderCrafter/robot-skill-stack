import py_trees

from robot_skill_stack.common.types import Pose
from robot_skill_stack.orchestration.behavior_trees.nodes.skill_node import (
    SkillNode,
)
from robot_skill_stack.runtime.runtime import RobotRuntime
from robot_skill_stack.runtime.skill import FailureCode


class RecoveryAllowed(py_trees.behaviour.Behaviour):
    """Never Home/retry a Pick that reached closure or reported grip loss."""
    def __init__(self, first):
        super().__init__('Recovery allowed')
        self.first = first

    def update(self):
        result = self.first.result
        allowed = (result is not None and not result.details.get('gripper_engaged') and
                   result.failure_code not in (FailureCode.GRASP_LOST, FailureCode.GRASP_UNCERTAIN,
                                               FailureCode.INTERNAL_ERROR, FailureCode.BUSY))
        return py_trees.common.Status.SUCCESS if allowed else py_trees.common.Status.FAILURE


def create_pick_and_place_recovery_tree(
    runtime: RobotRuntime,
    object_id: str,
    target: Pose,
    return_home=True,
):
    first = SkillNode(
        f"Pick {object_id}",
        runtime,
        "pick",
        object_id=object_id,
        grasp_hint="top",
    )
    retry = SkillNode(
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
            RecoveryAllowed(first),
            SkillNode("Recovery Home", runtime, "home"),
            retry,
        ],
    )
    selector = py_trees.composites.Selector(
        name="Pick With Recovery",
        memory=True,
        children=[first, recovery],
    )
    children = [
        selector,
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
    return py_trees.composites.Sequence(
        name="Pick & Place With Recovery",
        memory=True,
        children=children,
    )
