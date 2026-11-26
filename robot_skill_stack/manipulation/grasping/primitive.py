from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np
from numpy.typing import ArrayLike

from robot_skill_stack.common.rotations import quat_matrix
from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.grasping.clearance import GraspSafetyConfig, check_obstacles, tcp_height
from robot_skill_stack.manipulation.grasping.types import GraspFailureReason as Reason
from robot_skill_stack.manipulation.grasping.types import GraspPlan, GraspResult, ParallelJawGripperSpec
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.primitives import PrimitiveGeometry, PrimitiveShape
from robot_skill_stack.world.model.world_model import WorldModel


def _top_q(yaw: float) -> np.ndarray:
    """Top-down right_gripper quaternion: local +Z down, local Y jaw closing."""
    return np.array([0., -np.sin(yaw/2), np.cos(yaw/2), 0.])


class SimplePrimitiveGraspPlanner:
    """Rank top-down primitive grasps by wrist rotation, then test feasibility."""

    def __init__(
        self, *, approach_height: float = .10, default_lift_height: float = .12,
        grasp_z_offset: float = 0., gripper: ParallelJawGripperSpec | None = None,
        grasp_orientation: ArrayLike | None = None, safety: GraspSafetyConfig | None = None,
        support_plane_z: float | None = None, world_model: WorldModel | None = None,
        current_pose: Callable[[], Pose | None] | None = None,
        pose_reachable: Callable[[Pose], bool] | None = None,
    ) -> None:
        self.approach_height = float(approach_height)
        self.default_lift_height = float(default_lift_height)
        self.grasp_z_offset = float(grasp_z_offset)
        if not all(math.isfinite(x) for x in (self.approach_height, self.default_lift_height, self.grasp_z_offset)):
            raise ValueError("Grasp offsets must be finite")
        if self.approach_height <= 0 or self.default_lift_height <= 0:
            raise ValueError("Approach and lift distances must be positive")
        if support_plane_z is not None and not math.isfinite(support_plane_z):
            raise ValueError("support_plane_z must be finite")
        self.gripper = gripper or ParallelJawGripperSpec(.075, min_width=.005, open_width=.080)
        self.safety = safety or GraspSafetyConfig()
        self.support_plane_z = support_plane_z
        self.world_model = world_model
        self.current_pose = current_pose
        self.pose_reachable = pose_reachable
        self.grasp_orientation = (_top_q(0) if grasp_orientation is None
                                  else Pose([0, 0, 0], grasp_orientation).orientation)
        if not np.allclose(quat_matrix(self.grasp_orientation)[:, 2], [0, 0, -1], atol=1e-5):
            raise ValueError("Primitive planner requires a top-down grasp_orientation")

    def _geometry(
        self, obj: WorldObject,
    ) -> tuple[str, float, bool, list[tuple[float, float | None, float]]] | GraspResult:
        """Return strategy, height, rounded contact and (width, yaw, span) options."""
        geometry = obj.geometry
        if geometry is None and obj.class_name == "cube" and obj.size is not None:
            # Manually registered cube records may provide only axis-aligned size.
            geometry = PrimitiveGeometry(PrimitiveShape.CUBE, 1., box_size=obj.size, yaw=0.)
        if geometry is None or geometry.shape == PrimitiveShape.UNKNOWN:
            return GraspResult.failure(Reason.NO_FEASIBLE_GRASP, "Unknown primitive: refusing to guess a grasp.")
        try:
            if geometry.shape == PrimitiveShape.CUBE:
                dims = np.asarray(geometry.box_size if geometry.box_size is not None else obj.size, float)
                if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
                    raise ValueError("Box dimensions must be finite and positive")
                if geometry.yaw is None:
                    raise ValueError("Box yaw is unknown")
                yaw = float(geometry.yaw)
                if not math.isfinite(yaw):
                    raise ValueError("Box yaw is not finite")
                if obj.pose.orientation is not None and abs(quat_matrix(obj.pose.orientation)[2, 2]) < .995:
                    return GraspResult.failure(
                        Reason.UNSUPPORTED_ORIENTATION, "Tilted boxes are outside the top-down planner's scope.")
                return (geometry.shape.value, float(dims[2]), False,
                        [(dims[1], yaw, dims[0]), (dims[0], yaw+np.pi/2, dims[1])])
            radius = float(geometry.radius)
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError("Radius must be finite and positive")
            if geometry.shape == PrimitiveShape.SPHERE:
                return "sphere", 2*radius, True, [(2*radius, None, 2*radius)]
            length, axis = float(geometry.length), np.asarray(geometry.axis, float)
            if not math.isfinite(length) or length <= 0 or axis.shape != (3,) or not np.isfinite(axis).all():
                raise ValueError("Cylinder requires a finite positive length and a finite axis")
            if np.linalg.norm(axis) == 0:
                raise ValueError("Cylinder axis cannot be zero")
            axis = axis/np.linalg.norm(axis)
            vertical_extent = abs(axis[2])*length + 2*radius*np.sqrt(max(0., 1-axis[2]**2))
            if abs(axis[2]) >= .995:
                return "upright_cylinder", float(vertical_extent), False, [(2*radius, None, 2*radius)]
            if abs(axis[2]) <= .05:
                return ("lying_cylinder", float(vertical_extent), True,
                        [(2*radius, float(np.arctan2(axis[1], axis[0])), length)])
            return GraspResult.failure(Reason.UNSUPPORTED_ORIENTATION, "Leaning cylinders require a different grasp planner.")
        except (ValueError, TypeError) as exc:
            return GraspResult.failure(Reason.INVALID_GEOMETRY, str(exc))

    def plan(self, obj: WorldObject, *, lift_height: float | None = None,
             grasp_hint: str | None = None) -> GraspResult:
        """Choose a feasible grasp, preserving rejection evidence for each candidate."""
        if obj.pose is None:
            return GraspResult.failure(Reason.OBJECT_POSE_UNKNOWN, "Object pose is unknown.")
        if obj.pose.frame != "world" or not np.isfinite(obj.pose.position).all():
            return GraspResult.failure(Reason.INVALID_GEOMETRY, "A finite world-frame object pose is required.")
        if not obj.graspable:
            return GraspResult.failure(Reason.OBJECT_NOT_GRASPABLE, "Object is marked non-graspable.")
        if grasp_hint not in (None, "top", "top_down"):
            return GraspResult.failure(Reason.UNSUPPORTED_HINT, f"Unsupported grasp hint '{grasp_hint}'.")
        geometry = self._geometry(obj)
        if isinstance(geometry, GraspResult):
            return geometry
        strategy, height, rounded, options = geometry
        gripper, safety = self.gripper, self.safety
        lift = self.default_lift_height if lift_height is None else float(lift_height)
        if not math.isfinite(lift) or lift <= 0:
            return GraspResult.failure(Reason.INVALID_GEOMETRY, "lift_height must be finite and positive.")
        if height > safety.max_object_height:
            return GraspResult.failure(
                Reason.OBJECT_TOO_TALL, "Object exceeds the configured top-grasp height limit.",
                object_height_m=height, max_object_height_m=safety.max_object_height)
        if all(width > gripper.max_width for width, _, _ in options):
            return GraspResult.failure(
                Reason.OBJECT_TOO_LARGE, "Object width exceeds usable gripper width.",
                required_width_m=float(min(w for w, _, _ in options)), gripper_max_width_m=gripper.max_width)
        bottom = obj.pose.position[2]-height/2
        if self.support_plane_z is not None and bottom < self.support_plane_z-safety.clearance:
            return GraspResult.failure(
                Reason.INSUFFICIENT_CLEARANCE, "Object estimate intersects the support plane.",
                object_bottom_z_m=bottom, support_plane_z_m=self.support_plane_z)
        z, details = tcp_height(obj.pose.position[2], height, rounded, gripper, safety,
                                self.support_plane_z, obj.pose.position[2]+self.grasp_z_offset)
        if z is None:
            return GraspResult.failure(
                Reason.INSUFFICIENT_CLEARANCE,
                "No contact height clears the palm/table while retaining enough finger contact.", **details)
        current = self.current_pose() if self.current_pose is not None else None
        reference = (current.orientation if current is not None and current.orientation is not None
                     else self.grasp_orientation)
        try:
            reference = Pose([0, 0, 0], reference).orientation
            quat_matrix(reference)
        except ValueError as exc:
            return GraspResult.failure(Reason.INVALID_GEOMETRY, str(exc))
        preferred_yaw = -2*float(np.arctan2(reference[1], reference[2]))
        candidates, rejected = [], []
        for width, yaw, span in options:
            if yaw is None:
                yaws = preferred_yaw + np.arange(8)*np.pi/4
            else:
                yaws = [yaw, yaw+np.pi]
            for angle in yaws:
                rotation_error = 2*np.arccos(np.clip(abs(np.dot(reference, _top_q(angle))), 0, 1))
                candidates.append((float(rotation_error), float(width), float(angle), float(span)))
        for rotation_error, width, yaw, span in sorted(candidates):
            evidence = dict(details, strategy=strategy, required_width_m=width,
                            gripper_max_width_m=gripper.max_width, open_aperture_m=gripper.aperture,
                            clearance_m=safety.clearance, wrist_yaw_rad=yaw,
                            grasp_height_shift_m=z-obj.pose.position[2],
                            support_check=self.support_plane_z is not None,
                            axial_contact_span_m=min(span, gripper.finger_depth))
            if width > gripper.max_width:
                reason, message = Reason.OBJECT_TOO_LARGE, "Object width exceeds usable gripper width."
            elif width < gripper.min_width:
                reason, message = Reason.OBJECT_TOO_SMALL, "Object is narrower than reliable gripper feedback allows."
            elif min(span, gripper.finger_depth) < safety.min_contact_overlap:
                reason, message = Reason.OBJECT_TOO_SMALL, "Too little side surface overlaps the finger pads."
            elif width+2*safety.clearance > gripper.aperture+1e-9:
                reason, message = Reason.INSUFFICIENT_CLEARANCE, "Open fingers lack the required side clearance."
            else:
                grasp = Pose([*obj.pose.position[:2], z], _top_q(yaw), obj.pose.frame)
                pre_z = max(z+self.approach_height,
                            details["object_top_z_m"]+gripper.fingertip_offset+safety.approach_clearance)
                plan = GraspPlan(grasp.translated([0, 0, pre_z-z]), grasp, grasp.translated([0, 0, lift]))
                blocker, obstacle_details = check_obstacles(obj, plan, yaw, width, gripper, safety, self.world_model)
                evidence.update(obstacle_details)
                if blocker is not None:
                    reason, message = Reason.APPROACH_BLOCKED, f"Grasp path is blocked by '{blocker}'."
                elif self.pose_reachable is not None and not all(
                        self.pose_reachable(p) for p in (plan.pre_grasp, plan.grasp, plan.lift)):
                    reason, message = Reason.NO_FEASIBLE_GRASP, "Required grasp/lift poses have no IK solution."
                else:
                    return GraspResult.success(plan, **evidence, feasibility_checked=True,
                                               orientation_change_rad=float(rotation_error), rejected_candidates=rejected)
            rejected.append(dict(reason=reason.value, message=message, **evidence))
        failure = rejected[0]
        return GraspResult.failure(
            Reason(failure["reason"]), failure["message"], rejected_candidates=rejected,
            **{k: v for k, v in failure.items() if k not in ("reason", "message")})
