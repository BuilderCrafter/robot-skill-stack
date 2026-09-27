from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from robot_skill_stack.world.model.primitives import PrimitiveShape


@dataclass(frozen=True)
class GraspSafetyConfig:
    clearance: float = 0.005
    min_contact_overlap: float = 0.008
    max_object_height: float = 0.20
    approach_clearance: float = 0.020

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")


def tcp_height(center_z, height, rounded, gripper, safety, support_z, preferred):
    """Intersect palm/table clearance and usable contact-pad height intervals."""
    bottom, top = center_z-height/2, center_z+height/2
    overlap = safety.min_contact_overlap
    if rounded:
        # Both pads must straddle the equator; do not pinch the upper cap.
        low = center_z + overlap/2 - gripper.pad_above_tcp
        high = center_z - overlap/2 + gripper.pad_below_tcp
    else:
        low = bottom + overlap - gripper.pad_above_tcp
        high = top - overlap + gripper.pad_below_tcp
    low = max(low, top+safety.clearance-gripper.tcp_to_palm)
    if support_z is not None:
        low = max(low, support_z+safety.clearance+gripper.fingertip_offset)
    details = dict(object_bottom_z_m=bottom, object_top_z_m=top,
                   tcp_height_interval_m=[low, high], object_height_m=height)
    if height < overlap or gripper.pad_above_tcp+gripper.pad_below_tcp < overlap or low > high+1e-9:
        return None, details
    z = float(np.clip(preferred, low, high))
    details.update(grasp_tcp_z_m=z, palm_clearance_m=z+gripper.tcp_to_palm-top,
                   fingertip_z_m=z-gripper.fingertip_offset,
                   contact_band_z_m=[z-gripper.pad_below_tcp, z+gripper.pad_above_tcp],
                   support_clearance_m=None if support_z is None else z-gripper.fingertip_offset-support_z)
    return z, details


def object_aabb(obj):
    if obj.pose is None or not np.isfinite(obj.pose.position).all():
        return None
    sizes = []
    if obj.size is not None:
        sizes.append(np.asarray(obj.size, float))
    g = obj.geometry
    if g is not None:
        if g.shape == PrimitiveShape.CUBE and g.box_size is not None:
            d = np.asarray(g.box_size, float)
            c, s = abs(np.cos(g.yaw or 0)), abs(np.sin(g.yaw or 0))
            if d.shape == (3,):
                sizes.append(np.array([c*d[0]+s*d[1], s*d[0]+c*d[1], d[2]]))
        elif g.shape == PrimitiveShape.SPHERE and g.radius is not None:
            sizes.append(np.full(3, 2*g.radius))
        elif g.shape == PrimitiveShape.CYLINDER and all(v is not None for v in (g.radius, g.length, g.axis)):
            a = np.asarray(g.axis, float)
            if a.shape == (3,) and np.isfinite(a).all() and np.linalg.norm(a) > 0:
                a = a/np.linalg.norm(a)
                sizes.append(np.abs(a)*g.length + 2*g.radius*np.sqrt(np.maximum(0, 1-a*a)))
    sizes = [d for d in sizes if d.shape == (3,) and np.isfinite(d).all() and (d > 0).all()]
    return None if not sizes else (obj.pose.position.copy(), np.max(sizes, axis=0)/2)


def _overlap(box, obstacle, margin):
    # Separating-axis test: upright hand OBB against conservative object AABB.
    center, half, yaw = box
    other, other_half = obstacle
    delta = other-center
    if abs(delta[2]) >= half[2]+other_half[2]+margin:
        return False
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    for axis in (*np.eye(2), *rotation.T):
        radius = half[:2] @ np.abs(rotation.T @ axis) + other_half[:2] @ np.abs(axis)
        if abs(delta[:2] @ axis) >= radius+margin:
            return False
    return True


def swept_boxes(obj, plan, yaw, width, gripper):
    xy = plan.grasp.position[:2]
    z = plan.grasp.position[2]
    pre, lift = plan.pre_grasp.position[2], plan.lift.position[2]
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])

    def box(xhalf, y0, y1, z0, z1):
        center = np.r_[xy + rotation @ [0, (y0+y1)/2], (z0+z1)/2]
        return center, np.array([xhalf, (y1-y0)/2, (z1-z0)/2]), yaw

    g = gripper
    boxes = [("palm", box(g.palm_depth/2, -g.palm_width/2, g.palm_width/2,
                         z+g.tcp_to_palm, max(pre, lift)+g.tcp_to_palm+g.palm_height))]
    for sign in (-1, 1):
        y0, y1 = sorted((sign*g.aperture/2, sign*(g.aperture/2+g.finger_thickness)))
        boxes.append(("open_finger_descent", box(g.finger_depth/2, y0, y1,
                                                z-g.fingertip_offset, pre+g.tcp_to_palm)))
        y0, y1 = sorted((sign*width/2, sign*(g.aperture/2+g.finger_thickness)))
        boxes.append(("finger_closing_and_lift", box(g.finger_depth/2, y0, y1,
                                                    z-g.fingertip_offset, lift+g.tcp_to_palm)))
    bounds = object_aabb(obj)
    if bounds is not None:
        center, half = bounds
        delta = max(0, lift-z)
        boxes.append(("held_object_lift", (center+[0, 0, delta/2], half+[0, 0, delta/2], 0.)))
    return boxes


def check_obstacles(obj, plan, yaw, width, gripper, safety, world_model):
    if world_model is None:
        return None, {"obstacle_check": "not_configured"}
    boxes = swept_boxes(obj, plan, yaw, width, gripper)
    checked, unknown, cached = [], [], []
    for other in world_model.objects():
        if other.object_id == obj.object_id:
            continue
        bounds = object_aabb(other)
        if bounds is None or other.pose.frame != obj.pose.frame:
            unknown.append(other.object_id)
            continue
        checked.append(other.object_id)
        if not other.visible:
            cached.append(other.object_id)
        for part, box in boxes:
            if _overlap(box, bounds, safety.clearance):
                return other.object_id, {"blocking_object_id": other.object_id, "blocked_part": part,
                                         "blocking_object_visible": other.visible}
    details = {"obstacle_check": "known_object_swept_envelopes", "checked_object_ids": checked,
               "cached_obstacle_ids": cached, "uncheckable_object_ids": unknown}
    # A visible object with missing geometry is not evidence of free space.
    visible_unknown = [key for key in unknown if world_model.require(key).visible]
    if visible_unknown:
        details.update(blocking_object_id=visible_unknown[0], blocked_part="unknown_obstacle_geometry")
        return visible_unknown[0], details
    return None, details
