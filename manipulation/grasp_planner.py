from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from core.types import Pose
from world_model.entities import WorldObject

@dataclass
class GraspPlan:
    """
    Simple manipulation trajectory expressed as three semantic Cartesian poses.
    """

    pre_grasp: Pose
    grasp: Pose
    lift: Pose


class TopDownGraspPlanner:
    """
    Simple V1 grasp planner

    This is deliberately NOT a general grasp planner

    Assumptions:
        - object is small
        - object can be grasped from above
        - parallel Franko gripper
        - object pose is known
        - right_gripper frame targets the graspe center

    Later this class can be replaced by:
        - authored grasp database
        - geometry-based planner
        - ML grasp predictor
    """

    def __init__(
            self,
            *,
            approach_height: float = 0.10,
            default_lift_height: float = 0.12,
            grasp_z_offset: float = 0.10,
            grasp_orientation = None
    ) -> None:

        self.approach_height = approach_height
        self.default_lift_height = default_lift_height
        self.grasp_z_offset = grasp_z_offset

        # -------------------------------------------------------------
        # Default downward-facing Franka orientation.
        #
        # Quaternion convention:
        # [w, x, y, z]
        #
        # Equivalent to approximately 180° rotation about X.
        # -------------------------------------------------------------

        if grasp_orientation is None:
            grasp_orientation = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float)

        self.grasp_orientation = np.asarray(grasp_orientation, dtype=float)

    def plan(self, obj: WorldObject, *, lift_height: float | None = None, grasp_hint: str | None) -> GraspPlan | None:
        """ Generate a top-down grasp plan. """
        if obj.pose is None:
            return None

        if not obj.graspable:
            return None

        # V1 has only one grasp strategy

        if grasp_hint not in (None, "top", "top_down"):
            return None

        if lift_height is None:
            lift_height = self.default_lift_height

        # Grasp center
        grasp_position = obj.pose.position.copy()
        grasp_position[2] += self.grasp_z_offset
        grasp = Pose(position=grasp_position, orientation=self.grasp_orientation.copy(), frame=obj.pose.frame)

        # Pre-grasp pose
        pre_grasp = grasp.translated([0.0, 0.0, self.approach_height])

        # Lift pose
        lift = grasp.translated([0.0, 0.0, lift_height])

        return GraspPlan(pre_grasp=pre_grasp, grasp=grasp, lift=lift)
