"""Readiness fixtures and read-only planner scoring; never a runtime input."""
from dataclasses import replace
import copy

import numpy as np

from robot_skill_stack.common.rotations import quat_matrix
from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.grasping import ParallelJawGripperSpec, SimplePrimitiveGraspPlanner
from robot_skill_stack.manipulation.skills.pick import PickSkill
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from robot_skill_stack.world.model.world_model import WorldModel
from tests.support.multiview_iteration2 import (
    RobustnessScene, _blocker, box, cylinder, footprint_half_extent, sphere,
)
from tests.support.multiview_research import ResearchObject


# Independent of both iteration-two partitions, which are selection data now.
SEEDS = {'selection': 618033989, 'holdout': 141421357}
EYES = {
    'selection': ((1.49, -.15, 1.13), (.19, -.74, .66)),
    'holdout': ((1.58, .18, 1.02), (.29, -.81, .59)),
}


def readiness_holdout_scenes(split='holdout'):
    scenes = []
    for partition in ('selection', 'holdout') if split == 'both' else (split,):
        rng = np.random.default_rng(SEEDS[partition])
        for gap_mm in (2, 4, 6, 10):
            for kind in ('box_box', 'box_cylinder', 'cylinder_cylinder', 'sphere_sphere'):
                seed = int(rng.integers(1, 2**31))
                sample = np.random.default_rng(seed)
                middle = np.array([.445, -.015])+sample.uniform(-.035, .035, 2)
                yaw = float(sample.uniform(-.95, .95))
                direction = np.array([np.cos(yaw), np.sin(yaw)])
                def make_box(side):
                    size = sample.uniform([.036, .030, .034], [.069, .053, .065])
                    return box(side, middle, size, yaw)
                def make_cylinder(side):
                    return cylinder(side, middle, float(sample.uniform(.015, .028)),
                                    float(sample.uniform(.051, .097)))
                if kind == 'box_box':
                    objects = (make_box('A'), make_box('B'))
                elif kind == 'box_cylinder':
                    objects = (make_box('A'), make_cylinder('B'))
                elif kind == 'cylinder_cylinder':
                    objects = (make_cylinder('A'), make_cylinder('B'))
                else:
                    radius = float(sample.uniform(.019, .030))
                    objects = (sphere('A', middle, radius),
                               sphere('B', middle, radius*float(sample.uniform(.96, 1.04))))
                distance = sum(footprint_half_extent(obj, direction) for obj in objects)+gap_mm/1000
                if kind == 'sphere_sphere':
                    distance = float(np.sqrt(distance**2-(objects[1].center[2]-objects[0].center[2])**2))
                objects = tuple(replace(obj, name=f'readiness_{partition}_{kind}_{gap_mm}_{side}',
                    center=obj.center+np.r_[direction*distance*sign/2, 0.])
                    for obj, side, sign in zip(objects, ('A', 'B'), (-1, 1)))
                scenes.append(RobustnessScene(f'gap_{gap_mm}_{kind}', partition, seed, objects,
                    gap_mm=float(gap_mm), note='Independent final-readiness seed, dimensions, arrangement and viewpoints.'))
        seed = int(rng.integers(1, 2**31))
        isolated = (box(f'{partition}_box', (.325, .17), (.061, .043, .046), -.38),
                    sphere(f'{partition}_sphere', (.565, .17), .0275),
                    cylinder(f'{partition}_upright', (.325, -.19), .0215, .083),
                    cylinder(f'{partition}_lying', (.565, -.19), .0175, .107, -.51))
        if partition == 'holdout':
            isolated = tuple(replace(obj, center=obj.center+np.array([.009, -.007, 0.])) for obj in isolated)
        scenes.extend((RobustnessScene('isolated', partition, seed, isolated),
                       RobustnessScene('empty', partition, seed+1, ())))
        unknown = ResearchObject(f'{partition}_ellipsoid', 'ellipsoid', np.array([.465, -.025, .029]),
                                 np.array([.087, .043, .058]))
        scenes.append(RobustnessScene('unknown', partition, seed+2, (unknown,)))
        for camera, eye in zip(('A', 'B'), EYES[partition]):
            for severity in ('partial', 'strong', 'one_view'):
                target = cylinder(f'{partition}_{severity}_{camera}_target', (.455, -.015), .0215, .083)
                panel = _blocker(target, eye, partial=severity == 'partial', single=severity == 'one_view')
                panel = replace(panel, name=f'{partition}_{severity}_{camera}_panel')
                scenes.append(RobustnessScene(f'{severity}_{camera}', partition, seed+len(scenes),
                    (target,), (panel,), note='Physical occlusion; visibility is verified from measured depth.'))
        link = cylinder(f'{partition}_link', (.465, -.025), .0135, .145, -.48)
        link = replace(link, center=np.array([.465, -.025, .096]))
        link_objects = (box(f'{partition}_link_box', (.41, .075), (.057, .039, .052), .37),
                        sphere(f'{partition}_link_sphere', (.525, -.095), .026))
        scenes.append(RobustnessScene('link_clutter', partition, seed+30, link_objects, (link,)))
        temporal = (sphere(f'{partition}_moving_sphere', (.425, -.09), .0265),
                    box(f'{partition}_moving_box', (.49, .105), (.058, .041, .051), -.33))
        scenes.append(RobustnessScene('temporal', partition, seed+31, temporal, temporal=True))
    return scenes


def grasp_width_scenes(split='selection'):
    """Single targets isolate dimension-policy errors from neighbor clearance."""
    rng = np.random.default_rng(SEEDS[split]+729)
    if split == 'selection':
        box_widths = (4, 5, 6, 7.5, 69, 70, 71, 74, 75, 76, 80, 90)
        cylinder_widths = (68, 69, 70, 71, 72, 74, 75, 76, 78, 82)
        sphere_widths = (60, 62, 64, 68, 72, 74, 75, 76, 78, 82)
        lying_widths = (60, 70, 74, 76)
    else:
        box_widths = (4.4, 5.4, 6.4, 68.6, 69.6, 70.6, 74.6, 75.4, 81)
        cylinder_widths = (68.6, 69.6, 70.6, 74.6, 75.4, 79)
        sphere_widths = (59.5, 61.5, 62.5, 69.5, 74.6, 75.4, 81)
        lying_widths = (59.5, 69.5, 75.4)
    scenes = []
    for kind, widths in (('box', box_widths), ('upright', cylinder_widths),
                         ('sphere', sphere_widths), ('lying', lying_widths)):
        for width_mm in widths:
            seed = int(rng.integers(1, 2**31))
            center = np.array([.445, -.01])+rng.uniform(-.025, .025, 2)
            width = width_mm/1000
            name = f'readiness_{split}_{kind}_{width_mm:g}'.replace('.', 'p')
            if kind == 'box':
                # Both horizontal edges exceed 75 mm in the oversized fixtures;
                # otherwise the second axis must not offer an easier grasp.
                size = (.051 if width < .01 else max(.086, width+.007), width, .047)
                obj = box(name, center, size, float(rng.uniform(-.85, .85)))
            elif kind == 'sphere':
                obj = sphere(name, center, width/2)
            else:
                obj = cylinder(name, center, width/2, .079 if kind == 'upright' else .105,
                               None if kind == 'upright' else float(rng.uniform(-.75, .75)))
            scenes.append(RobustnessScene(name, split, seed, (obj,),
                note='Read-only planner width fixture; GT planner determines full feasibility, not width alone.'))
    unknown = ResearchObject(f'readiness_{split}_unsupported', 'ellipsoid', np.array([.45, .0, .027]),
                             np.array([.085, .041, .054]))
    scenes.append(RobustnessScene(f'readiness_{split}_unknown', split, SEEDS[split]+1, (unknown,)))
    pair = next(scene for scene in readiness_holdout_scenes(split) if scene.name == 'gap_2_sphere_sphere')
    scenes.append(replace(pair, name=f'readiness_{split}_ambiguous_spheres'))
    ordinary = box(f'readiness_{split}_ordinary', (.44, .015), (.05, .05, .05), .27)
    scenes.append(RobustnessScene(f'readiness_{split}_ordinary', split, SEEDS[split]+2, (ordinary,)))
    return scenes


def truth_world_object(obj, object_id=None):
    shape = 'unknown' if obj.shape == 'ellipsoid' else obj.shape
    geometry = PrimitiveGeometry(shape, confidence=1. if shape != 'unknown' else 0.)
    if shape == 'cube':
        geometry.box_size, geometry.yaw = obj.size.copy(), obj.yaw
        c, s = abs(np.cos(obj.yaw)), abs(np.sin(obj.yaw))
        size = np.array([c*obj.size[0]+s*obj.size[1], s*obj.size[0]+c*obj.size[1], obj.size[2]])
    elif shape == 'sphere':
        geometry.radius, size = obj.radius, np.full(3, 2*obj.radius)
    elif shape == 'cylinder':
        geometry.radius, geometry.length, geometry.axis = obj.radius, obj.length, np.asarray(obj.axis, float)
        size = abs(geometry.axis)*obj.length+2*obj.radius*np.sqrt(np.maximum(0, 1-geometry.axis**2))
    else:
        size = obj.size.copy()
    orientation = None if shape != 'cube' else np.array([np.cos(obj.yaw/2), 0., 0., np.sin(obj.yaw/2)])
    return WorldObject(object_id or obj.name, shape if shape != 'unknown' else None,
        Pose(obj.center.copy(), orientation), size, geometry, graspable=shape != 'unknown', visible=True)


class QueryOnlyBackend:
    """Expose queries while making any accidental command fail immediately."""
    def __init__(self, backend=None):
        self.backend, self.commands, self.ik_queries = backend, [], 0

    def get_end_effector_pose(self):
        return None if self.backend is None else self.backend.get_end_effector_pose()

    def check_reachability(self, pose):
        self.ik_queries += 1
        if self.backend is None:
            raise RuntimeError('No native IK backend was constructed')
        return self.backend.check_reachability(pose)

    def __getattr__(self, name):
        def forbidden(*args, **kwargs):
            self.commands.append(name)
            raise RuntimeError(f'Readiness evaluation forbids backend command: {name}')
        return forbidden


def make_planner(model, backend=None, *, use_ik=False):
    return SimplePrimitiveGraspPlanner(world_model=model, support_plane_z=0.,
        gripper=ParallelJawGripperSpec(.075, min_width=.005, open_width=.080),
        current_pose=None if backend is None or backend.backend is None else backend.get_end_effector_pose,
        pose_reachable=backend.check_reachability if use_ik and backend is not None and backend.backend is not None else None)


def pick_start_guard(model, object_id, planner, backend=None):
    """Call only the existing synchronous precondition query, never a Pick request."""
    skill = PickSkill(backend or QueryOnlyBackend(), model, planner)
    obj, failure = skill._start(object_id)
    return dict(allowed=failure is None, failure_code=None if failure is None else failure.failure_code.value,
                message=None if failure is None else failure.message)


def plan_payload(result):
    def pose(value):
        return dict(position=value.position.tolist(), orientation=None if value.orientation is None else value.orientation.tolist(),
                    frame=value.frame)
    return dict(feasible=result.ok, failure_reason=None if result.failure_reason is None else result.failure_reason.value,
        message=result.message, details=result.details,
        poses=None if result.plan is None else {key: pose(getattr(result.plan, key)) for key in ('pre_grasp', 'grasp', 'lift')})


def evaluate_grasp(truth, observed, model, all_truth, *, backend=None):
    truth_model = WorldModel()
    for obj in all_truth:
        truth_model.register(truth_world_object(obj))
    truth_obj = truth_model.require(truth.name)
    gt_plan = make_planner(truth_model, backend).plan(truth_obj)
    planner = make_planner(model, backend)
    estimated = None if observed is None else planner.plan(observed)
    guard = None if observed is None else pick_start_guard(model, observed.object_id, planner, backend)
    predicted_feasible = estimated is not None and estimated.ok and guard['allowed']
    proposed = None
    if estimated is not None and estimated.ok:
        grasp = estimated.plan.grasp
        closing = quat_matrix(grasp.orientation)[:, 1]
        if truth.shape == 'cube':
            c, s = np.cos(truth.yaw), np.sin(truth.yaw)
            true_width = float(abs(closing @ [c, s, 0])*truth.size[0]+abs(closing @ [-s, c, 0])*truth.size[1])
        elif truth.shape in ('sphere', 'cylinder'):
            if truth.shape == 'sphere' or abs(truth.axis[2]) > .99:
                true_width = 2*truth.radius
            else:
                dot = float(abs(closing @ truth.axis))
                true_width = dot*truth.length+2*truth.radius*np.sqrt(max(0., 1-dot*dot))
        else:
            true_width = None
        if true_width is not None:
            true_width = float(true_width)
        true_height = truth_obj.size[2]
        true_top = truth.center[2]+true_height/2
        g = planner.gripper
        proposed = dict(true_required_width_m=true_width,
            true_palm_clearance_m=float(grasp.position[2]+g.tcp_to_palm-true_top),
            true_support_clearance_m=float(grasp.position[2]-g.fingertip_offset),
            backend_width_feasible=None if true_width is None else bool(g.min_width <= true_width <= g.max_width),
            aperture_clearance_feasible=None if true_width is None else bool(true_width+2*planner.safety.clearance <= g.aperture+1e-9),
            ik=None if backend is None or backend.backend is None else {
                key: bool(backend.check_reachability(getattr(estimated.plan, key))) for key in ('pre_grasp', 'grasp', 'lift')})
        proposed['clearance_violation'] = min(proposed['true_palm_clearance_m'], proposed['true_support_clearance_m']) < planner.safety.clearance-1e-9
        proposed['physical_overlap'] = min(proposed['true_palm_clearance_m'], proposed['true_support_clearance_m']) < 0
        proposed['width_violation'] = true_width is not None and not (
            proposed['backend_width_feasible'] and proposed['aperture_clearance_feasible'])
    return dict(object=truth.name, truth_shape=truth.shape,
        detected=observed is not None, object_id=None if observed is None else observed.object_id,
        truth_plan=plan_payload(gt_plan), estimated_plan=None if estimated is None else plan_payload(estimated),
        pick_start_guard=guard, estimated_feasible=predicted_feasible, truth_feasible=gt_plan.ok,
        false_feasible=bool(predicted_feasible and not gt_plan.ok),
        false_infeasible=bool(gt_plan.ok and not predicted_feasible), proposed_truth_checks=proposed,
        estimated_geometry_feasible=predicted_feasible,
        endpoint_ik_all_reachable=None if proposed is None or proposed['ik'] is None else all(proposed['ik'].values()),
        readonly_readiness_feasible=bool(predicted_feasible and proposed is not None and
            not proposed['clearance_violation'] and not proposed['width_violation'] and
            (proposed['ik'] is None or all(proposed['ik'].values()))),
        estimated_class=None if observed is None else observed.class_name,
        confidence=None if observed is None else observed.confidence,
        center_error_mm=None if observed is None else float(np.linalg.norm(observed.pose.position-truth.center)*1000))


def invalid_geometry_queries(world_object, model, backend=None):
    """Fault injections copied from measured geometry; never fed to the source."""
    planner = make_planner(model, backend)
    results = {}
    unknown = copy.deepcopy(world_object)
    unknown.geometry = PrimitiveGeometry('unknown')
    unknown.class_name, unknown.graspable = None, False
    results['explicit_unknown'] = plan_payload(planner.plan(unknown))
    invalid = copy.deepcopy(world_object)
    invalid.graspable = True
    invalid.pose.position[0] = np.nan
    results['nonfinite_pose'] = plan_payload(planner.plan(invalid))
    invalid = copy.deepcopy(world_object)
    invalid.graspable = True
    invalid.geometry = PrimitiveGeometry('sphere', .9, radius=np.nan)
    results['nonfinite_radius'] = plan_payload(planner.plan(invalid))
    return results
