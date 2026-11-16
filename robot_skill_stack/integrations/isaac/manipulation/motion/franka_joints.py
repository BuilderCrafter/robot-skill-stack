"""Franka-specific joint-name validation for the Isaac global-motion adapter."""
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure, map_joints


FRANKA_ARM_JOINTS = frozenset(f"panda_joint{i}" for i in range(1, 8))


def map_arm_joints(active_names, articulation_names) -> tuple[int, ...]:
    """Retain Lula's seven-arm-joint order while excluding the finger joints."""
    names = tuple(active_names)
    if len(names) != len(FRANKA_ARM_JOINTS) or set(names) != FRANKA_ARM_JOINTS:
        raise PlanningFailure("joint_mapping_failure", "Planner must name exactly the seven Panda arm joints.")
    return map_joints(names, articulation_names)
