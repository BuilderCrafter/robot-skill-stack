#!/usr/bin/env python3
"""Offline surface audit of actual cooked Franka hulls versus Lula spheres.

No SimulationApp, articulation command, scene mutation or vendor-file edit is
used. Surface fractions are deterministic equal-area quadrature estimates, not
solid-volume coverage or continuous collision certificates. Raw coordinates
remain in the supplied ignored artifact directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from robot_skill_stack.common.rotations import quat_matrix
from robot_skill_stack.integrations.isaac.manipulation.motion.lula_global import adjacent_self_collision_pairs
from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import (
    PlaneDomain, StaticObstacle, StaticWorldSnapshot,
)


ROOT = Path(__file__).resolve().parents[1]
FINGERS = ("panda_leftfinger", "panda_rightfinger")
REFINED_LINKS = ("panda_hand", *FINGERS)
DEFAULT_CASES = tuple(ROOT / "outputs/motion-planning-v4/native/primary" / name / "result.json"
    for name in ("global-static-seed123456", "global-narrow-seed123456", "global-large-seed123456"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def body_mesh_vertices(vertices, body_from_mesh):
    points, transform = np.asarray(vertices, float), np.asarray(body_from_mesh, float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or transform.shape != (4, 4)
            or not np.isfinite(points).all() or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or abs(np.linalg.det(transform[:3, :3])) < 1e-15):
        raise ValueError("Finite nondegenerate physical mesh-to-body transform required")
    return points@transform[:3, :3].T+transform[:3, 3]


def surface_quadrature(vertices, triangles, subdivisions=4):
    """One centroid per equal-area subtriangle; n² points per input triangle."""
    points, faces = np.asarray(vertices, float), np.asarray(triangles)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
            or not len(faces) or np.any(faces < 0) or np.any(faces >= len(points))
            or type(subdivisions) is not int or not 1 <= subdivisions <= 32):
        raise ValueError("Finite vertices, integer triangle indices and bounded subdivision are required")
    barycentric = []
    for first in range(subdivisions):
        for second in range(subdivisions-first):
            barycentric.append(((first+1/3)/subdivisions, (second+1/3)/subdivisions))
            if first+second < subdivisions-1:
                barycentric.append(((first+2/3)/subdivisions, (second+2/3)/subdivisions))
    uv = np.asarray(barycentric)
    assert len(uv) == subdivisions**2
    corners = points[faces]
    side1, side2 = corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0]
    area = np.linalg.norm(np.cross(side1, side2), axis=1)/2
    valid = area > 0.
    samples = (corners[valid, 0, None] + uv[None, :, 0, None]*side1[valid, None]
               + uv[None, :, 1, None]*side2[valid, None]).reshape(-1, 3)
    weights = np.repeat(area[valid]/len(uv), len(uv))
    if not len(samples) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Cooked reference must have nonzero finite surface area")
    return samples, weights


def sphere_union_gaps(points, centers, radii):
    """Signed distance to the nearest sphere surface, chunked to bound memory."""
    points, centers, radii = np.asarray(points, float), np.asarray(centers, float), np.asarray(radii, float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or centers.ndim != 2
            or centers.shape[1:] != (3,) or radii.shape != (len(centers),)
            or not np.isfinite(points).all() or not np.isfinite(centers).all()
            or not np.isfinite(radii).all() or np.any(radii <= 0)):
        raise ValueError("Finite points, sphere centers and positive matching radii are required")
    if not len(centers):
        return np.full(len(points), np.inf)
    result = []
    for start in range(0, len(points), 4096):
        difference = points[start:start+4096, None]-centers[None]
        result.append((np.linalg.norm(difference, axis=2)-radii).min(axis=1))
    return np.concatenate(result) if result else np.empty(0)


def coverage_statistics(gaps, weights):
    gaps, weights = np.asarray(gaps, float), np.asarray(weights, float)
    if gaps.shape != weights.shape or not len(gaps) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Positive surface quadrature weights must match gap samples")
    covered = gaps <= 1e-9  # A numerical inclusion tolerance, never geometry padding.
    finite = gaps[np.isfinite(gaps)]
    return dict(samples=len(gaps), surface_area_m2=float(weights.sum()),
        estimated_covered_surface_fraction=float(weights[covered].sum()/weights.sum()),
        uncovered_samples=int(np.count_nonzero(~covered)),
        maximum_sampled_outside_gap_m=None if not len(finite) else max(0., float(finite.max())),
        minimum_sampled_signed_gap_m=None if not len(finite) else float(finite.min()))


def pose_matrix_xyzw(pose, units=1.):
    values = np.asarray(pose, float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Recorded body pose must be finite xyz/xyzw")
    result = np.eye(4)
    result[:3, :3] = quat_matrix(values[[6, 3, 4, 5]])
    result[:3, 3] = values[:3]*units
    return result


def unique_state_indices(samples, count=5):
    if type(count) is not int or not 1 <= count <= 21:
        raise ValueError("State count must be bounded from1 through21")
    unique = []
    seen = set()
    for index, sample in enumerate(samples):
        key = sample["physics_step"], sample["simulation_time"]
        if key not in seen:
            seen.add(key)
            unique.append(index)
    if not unique:
        raise ValueError("No actual recorded physics states")
    chosen = np.rint(np.linspace(0, len(unique)-1, min(count, len(unique)))).astype(int)
    return [unique[index] for index in chosen]


def load_native_model(isaac_root):
    # Direct native Lula import; no Isaac foundation shims or SimulationApp.
    import yaml
    installed = Path(isaac_root)
    native = installed / "exts/isaacsim.robot_motion.lula/pip_prebundle"
    sys.path.insert(0, str(native))
    import lula
    motion = installed / "exts/isaacsim.robot_motion.motion_generation"
    descriptor = motion / "motion_policy_configs/franka/rmpflow/robot_descriptor.yaml"
    urdf = motion / "motion_policy_configs/franka/lula_franka_gen.urdf"
    document = yaml.safe_load(descriptor.read_text())
    robot = lula.load_robot(str(descriptor), str(urdf))
    world = lula.create_world()
    inspector = lula.create_robot_world_inspector(robot, world.add_world_view())
    inspector.update_world_view()
    sphere_entries = [(name, item) for group in document["collision_spheres"]
                      for name, rows in group.items() for item in rows]
    if len(sphere_entries) != len(inspector.collision_sphere_radii()):
        raise ValueError("Installed descriptor and native collision-sphere counts differ")
    return dict(lula=lula, robot=robot, world=world, inspector=inspector, kinematics=robot.kinematics(),
        descriptor=descriptor, urdf=urdf, document=document, spheres=sphere_entries,
        exclusions=adjacent_self_collision_pairs(urdf))


def _native_pose(kinematics, q, frame):
    pose = kinematics.pose(np.asarray(q)[:, None], frame)
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = pose.rotation.matrix(), pose.translation
    return result


def _summary(rows):
    values = [row["estimated_covered_surface_fraction"] for row in rows]
    gaps = [row["maximum_sampled_outside_gap_m"] for row in rows
            if row["maximum_sampled_outside_gap_m"] is not None]
    return dict(states=len(rows), sampled_points=sum(row["samples"] for row in rows),
        surface_area_m2=rows[0]["surface_area_m2"],
        estimated_covered_surface_fraction_minimum=min(values),
        estimated_covered_surface_fraction_mean=float(np.mean(values)),
        estimated_covered_surface_fraction_maximum=max(values),
        maximum_sampled_outside_gap_m=max(gaps, default=None))


def clear_numeric_scene(model):
    for handle in model.pop("scene_handles", []):
        model["world"].remove_obstacle(handle)
    model["inspector"].update_world_view()


def register_numeric_scene(model, snapshot, base):
    """Copy the original recorded planning world, not a physical USD scene."""
    clear_numeric_scene(model)
    handles = model["scene_handles"] = []
    lula = model["lula"]
    for item in snapshot.obstacles:
        kind = {"cuboid": lula.Obstacle.Type.CUBE, "sphere": lula.Obstacle.Type.SPHERE,
                "capsule": lula.Obstacle.Type.CYLINDER}[item.kind]
        obstacle = lula.create_obstacle(kind)
        if item.kind == "cuboid":
            obstacle.set_attribute(lula.Obstacle.Attribute.SIDE_LENGTHS, np.asarray(item.dimensions))
        else:
            obstacle.set_attribute(lula.Obstacle.Attribute.RADIUS, item.radius)
            if item.kind == "capsule":
                obstacle.set_attribute(lula.Obstacle.Attribute.HEIGHT, item.height)
        rotation = base[:3, :3].T@quat_matrix(item.orientation)
        position = base[:3, :3].T@(np.asarray(item.position)-base[:3, 3])
        handles.append(model["world"].add_obstacle(obstacle,
            lula.Pose3(lula.Rotation3(rotation), position)))
    model["inspector"].update_world_view()


def audit_case(path, model, *, state_count=5, subdivisions=4, refinement=8):
    path = Path(path)
    folder = path.parent
    payload = json.loads(path.read_text())
    result = payload["metrics"]
    if payload["status"] != "PASS" or not result["geometry_reference"]["cooked_only"]:
        raise ValueError("A valid, actually cooked native reference is required")
    trace_path, plan_path, geometry_path = folder/"trace.json", folder/"plan_trace.json", folder/"geometry_reference.npz"
    trace, plan = json.loads(trace_path.read_text()), json.loads(plan_path.read_text())
    snapshot_data = plan["snapshot"]
    snapshot = StaticWorldSnapshot(**dict(snapshot_data,
        obstacles=tuple(StaticObstacle(**value) for value in snapshot_data["obstacles"]),
        plane_domains=tuple(PlaneDomain(**value) for value in snapshot_data.get("plane_domains", []))))
    units = float(plan["snapshot"]["meters_per_unit"])
    if units != 1.:
        raise ValueError("This recorded-evidence audit is validated only for meter stages")
    expected = result["strategy_diagnostics"]["sdk_configuration_sha256"]
    if expected["robot_description_path"] != digest(model["descriptor"]) or expected["urdf_path"] != digest(model["urdf"]):
        raise ValueError("Installed collision model differs from the measured native reference")
    body_paths = tuple(result["contact_reference"]["sensor_paths"])
    body_indices = {name: index for index, name in enumerate(body_paths)}
    if len(body_indices) != len(body_paths):
        raise ValueError("Ambiguous physical body paths")
    names = tuple(plan["joint_names"])
    native_names = tuple(model["robot"].c_space_coord_name(i) for i in range(model["robot"].num_c_space_coords()))
    if names != native_names:
        raise ValueError("Recorded and installed arm joint orders differ")
    selected = unique_state_indices(trace["samples"], state_count)
    parts = []
    with np.load(geometry_path, allow_pickle=False) as archive:
        if len(archive["body_poses"]) != len(trace["samples"]):
            raise ValueError("Cooked reference poses and physical trace do not align")
        body_poses = archive["body_poses"].copy()
        index = 0
        for record in result["geometry_reference"]["cooking"]:
            if record["body"] not in body_indices:
                raise ValueError("Physical cooked body missing from contact/body reference")
            for hull in range(record["cooked_hulls"]):
                transform = archive[f"body_from_mesh_{index}"]
                vertices = body_mesh_vertices(archive[f"vertices_{index}"], transform)
                faces = archive[f"triangles_{index}"]
                samples, weights = surface_quadrature(vertices, faces, subdivisions)
                refined = (surface_quadrature(vertices, faces, refinement)
                    if record["body"].rsplit("/", 1)[-1] in REFINED_LINKS else None)
                parts.append(dict(body=record["body"], path=record["path"]+f"#convex{hull}",
                    samples=samples, weights=weights, refined=refined, hull_index=index))
                index += 1
        if index != result["geometry_reference"]["meshes"]:
            raise ValueError("Not every copied cooked hull was audited")
    own_counts = Counter(name for name, _ in model["spheres"])
    per_body, detailed, witnesses = defaultdict(list), [], []
    alignment, finger_states, configuration_checks, render = [], [], [], None
    first_base = pose_matrix_xyzw(body_poses[selected[0], body_indices["/Franka/panda_link0"]], units)
    register_numeric_scene(model, snapshot, first_base)
    limits = [model["kinematics"].c_space_coord_limits(i) for i in range(7)]
    low, high = np.array([x.lower for x in limits]), np.array([x.upper for x in limits])
    for sample_index in selected:
        row = trace["samples"][sample_index]
        poses = np.asarray(row["poses"])
        if not np.array_equal(poses, body_poses[sample_index]):
            raise ValueError("Raw JSON and NPZ physical poses differ")
        q = np.asarray(row["joint_feedback"]["positions"])[plan["joint_indices"]]
        physical = {name: pose_matrix_xyzw(poses[index], units) for name, index in body_indices.items()}
        base = physical["/Franka/panda_link0"]
        if np.max(np.abs(base-first_base)) > 1e-8:
            raise ValueError("Robot base moved relative to the frozen recorded world")
        centers_base = np.asarray(model["inspector"].collision_sphere_positions(q))
        radii = np.asarray(model["inspector"].collision_sphere_radii())
        centers = centers_base@base[:3, :3].T+base[:3, 3]
        snapshot.assert_within_domains(centers, radii)
        joint_limits = bool(np.all(q >= low-1e-8) and np.all(q <= high+1e-8))
        external = bool(model["inspector"].in_collision_with_obstacle(q))
        self_pairs = [tuple(pair) for pair in model["inspector"].frames_in_self_collision(q)
            if frozenset(pair) not in model["exclusions"]]
        configuration_valid = joint_limits and not external and not self_pairs
        configuration_checks.append(dict(physics_step=row["physics_step"], joint_limits=joint_limits,
            original_world_external_collision=external, unmasked_self_pairs=self_pairs,
            within_original_plane_domains=True, sampled_configuration_valid=configuration_valid))
        by_frame = defaultdict(list)
        for name, item in model["spheres"]:
            transform = base@_native_pose(model["kinematics"], q, name)
            by_frame[name].append((transform[:3, :3]@np.asarray(item["center"])+transform[:3, 3], item["radius"]))
        # This also verifies that parsing the descriptor did not silently change model geometry.
        parsed_centers = np.array([center for values in by_frame.values() for center, _ in values])
        if np.max(np.linalg.norm(parsed_centers[:, None]-centers[None], axis=2).min(axis=1)) > 1e-8:
            raise ValueError("Descriptor sphere positions do not match native Lula")
        frame_errors = []
        for path_name, body in physical.items():
            frame = path_name.rsplit("/", 1)[-1]
            expected_pose = base@_native_pose(model["kinematics"], q, frame)
            error = dict(frame=frame, translation_error_m=float(np.linalg.norm(expected_pose[:3, 3]-body[:3, 3])),
                rotation_matrix_max_error=float(np.max(np.abs(expected_pose[:3, :3]-body[:3, :3]))))
            frame_errors.append(error)
            if frame not in FINGERS and (error["translation_error_m"] > 2e-5 or error["rotation_matrix_max_error"] > 2e-5):
                raise ValueError(f"Lula FK and physical frame differ at {frame}, physics step {row['physics_step']}: {error}")
        alignment.extend(frame_errors)
        finger_states.append(np.asarray(row["joint_feedback"]["positions"])[7:9].tolist())
        plotted_parts = []
        for part in parts:
            body = physical[part["body"]]
            points = part["samples"]@body[:3, :3].T+body[:3, 3]
            gaps = sphere_union_gaps(points, centers, radii)
            frame = part["body"].rsplit("/", 1)[-1]
            own = by_frame.get(frame, [])
            own_gaps = sphere_union_gaps(points, np.asarray([v[0] for v in own]).reshape(-1, 3), np.asarray([v[1] for v in own]))
            all_stats, own_stats = coverage_statistics(gaps, part["weights"]), coverage_statistics(own_gaps, part["weights"])
            item = dict(physics_step=row["physics_step"], simulation_time=row["simulation_time"],
                body=part["body"], hull_index=part["hull_index"], own_spheres=own_counts[frame],
                all_spheres=all_stats, own_sphere_coverage=own_stats)
            if part["refined"] is not None:
                fine, fine_weights = part["refined"]
                fine = fine@body[:3, :3].T+body[:3, 3]
                fine_gaps = sphere_union_gaps(fine, centers, radii)
                item["refined_all_spheres"] = coverage_statistics(fine_gaps, fine_weights)
                plotted_parts.append(dict(body=part["body"], points=fine, gaps=fine_gaps))
                if frame in FINGERS and configuration_valid:
                    candidate = int(np.argmax(fine_gaps))
                    gap = float(fine_gaps[candidate])
                    if gap > 1e-5:
                        point = fine[candidate]
                        radius = min(.002, gap/4)
                        ball = model["lula"].create_obstacle(model["lula"].Obstacle.Type.SPHERE)
                        ball.set_attribute(model["lula"].Obstacle.Attribute.RADIUS, radius)
                        local_point = base[:3, :3].T@(point-base[:3, 3])
                        handle = model["world"].add_obstacle(ball, model["lula"].Pose3(
                            model["lula"].Rotation3(np.eye(3)), local_point))
                        try:
                            model["inspector"].update_world_view()
                            native_collision = bool(model["inspector"].in_collision_with_obstacle(q))
                        finally:
                            model["world"].remove_obstacle(handle)
                            model["inspector"].update_world_view()
                        if native_collision:
                            raise ValueError("Proposed outside-model witness unexpectedly collides in native Lula")
                        witnesses.append(dict(body=part["body"], physics_step=row["physics_step"],
                            physical_cooked_surface_center=point.tolist(), native_q=q.tolist(),
                            radius_m=radius, point_outside_all_spheres_gap_m=gap,
                            native_sphere_model_collision=False,
                            original_world_and_joint_self_checks_passed=True,
                            real_convex_hull_intersection_by_construction=True,
                            scope="Analysis-only hypothetical ball centered on actual cooked-hull surface sample; never authored or simulated. This is a configuration-level model gap, not an executed collision or complete-path certificate."))
            detailed.append(item)
            per_body[part["body"]].append(item)
        if render is None:
            hand = physical["/Franka/panda_hand"]
            render = dict(hand_pose=hand, centers=centers, radii=radii, parts=plotted_parts,
                hand_centers=np.asarray([value[0] for value in by_frame["panda_hand"]]),
                hand_radii=np.asarray([value[1] for value in by_frame["panda_hand"]]),
                case=folder.name, physics_step=row["physics_step"])
    summaries = []
    for body, rows in per_body.items():
        frame = body.rsplit("/", 1)[-1]
        summary = dict(body=body, own_sphere_count=own_counts[frame],
            all_sphere_surface=_summary([row["all_spheres"] for row in rows]),
            own_sphere_surface=_summary([row["own_sphere_coverage"] for row in rows]))
        if "refined_all_spheres" in rows[0]:
            summary["refined_all_sphere_surface"] = _summary([row["refined_all_spheres"] for row in rows])
        summaries.append(summary)
    arm_errors = [row for row in alignment if row["frame"] not in FINGERS]
    finger_errors = [row["translation_error_m"] for row in alignment if row["frame"] in FINGERS]
    qf = np.asarray(finger_states)
    brief = dict(case=folder.name, status="PASS_SURFACE_AND_FRAME_AUDIT", source_result_status=payload["status"],
        measured_configuration=result["configuration"], states=len(selected), cooked_hulls=len(parts),
        descriptor_sphere_count=len(model["spheres"]), sphere_counts=dict(own_counts), bodies=summaries,
        configuration_checks=configuration_checks,
        original_static_world=dict(obstacles=len(snapshot.obstacles),
            source_kinds=dict(Counter(value.source_kind for value in snapshot.obstacles)),
            paths=[value.path for value in snapshot.obstacles],
            plane_domains_checked=len(snapshot.plane_domains)),
        frame_alignment=dict(arm_and_hand_translation_error_max_m=max(x["translation_error_m"] for x in arm_errors),
            arm_and_hand_rotation_matrix_error_max=max(x["rotation_matrix_max_error"] for x in arm_errors),
            fixed_finger_translation_mismatch_range_m=[min(finger_errors), max(finger_errors)]),
        actual_finger_positions_range_m=np.column_stack((qf.min(0), qf.max(0))).tolist(),
        descriptor_fixed_finger_positions_m=[rule["value"] for rule in model["document"]["cspace_to_urdf_rules"]],
        model_gap_witnesses=len(witnesses),
        minimum_witness_sphere_radius_m=min((x["radius_m"] for x in witnesses), default=None),
        maximum_witness_gap_m=max((x["point_outside_all_spheres_gap_m"] for x in witnesses), default=None),
        source_artifacts={p.name: dict(path=str(p), sha256=digest(p)) for p in (path, trace_path, plan_path, geometry_path)},
        observed_native_contacts=result["contacts"], observed_native_physical_contact=result["contact_severity"])
    # Only scalar native outcomes belong in the compact summary.
    severity = result["contact_severity"]
    brief["observed_native_physical_contact"] = dict(
        physical_classification=severity.get("physical_classification"),
        observed_physical_contact_steps=severity.get("observed_physical_contact_steps"),
        maximum_penetration_m=severity.get("depth_m", {}).get("maximum"),
        maximum_normal_pair_impulse_ns=severity.get("normal_pair_impulse_ns", {}).get("maximum"))
    return brief, dict(case=folder.name, states=detailed, witnesses=witnesses), render


def plot_coverage(cases, render, destination):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"outputs/motion-planning-v4-1/matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    case = cases[0]
    bodies = case["bodies"]
    figure = plt.figure(figsize=(14, 6.7), constrained_layout=True)
    left = figure.add_subplot(1, 2, 1)
    labels = [row["body"].rsplit("/", 1)[-1].replace("panda_", "") for row in bodies]
    index = np.arange(len(labels))
    own = [100*row["own_sphere_surface"]["estimated_covered_surface_fraction_mean"] for row in bodies]
    union = [100*row["all_sphere_surface"]["estimated_covered_surface_fraction_mean"] for row in bodies]
    left.barh(index-.17, own, .32, color="#7e9fc0", label="Spheres on the body's own frame")
    left.barh(index+.17, union, .32, color="#285f91", label="Union of all 55 planning spheres")
    left.set(yticks=index, yticklabels=labels, xlim=(0, 100), xlabel="Estimated cooked-triangle surface coverage (%)",
        title="Actual physical robot references\nMean over 5 recorded static-case states; 16 centroids/triangle")
    left.invert_yaxis()
    left.grid(axis="x", alpha=.2)
    handles, labels = left.get_legend_handles_labels()
    figure.legend(handles, labels, fontsize=8, loc="lower center", bbox_to_anchor=(.25, -.08))
    right = figure.add_subplot(1, 2, 2, projection="3d")
    hand = render["hand_pose"]
    rotation, origin = hand[:3, :3], hand[:3, 3]
    shown = 0
    for part in render["parts"]:
        points = (part["points"]-origin)@rotation
        mask = part["gaps"] <= 1e-9
        stride = max(1, int(math.ceil(len(points)/1800)))
        points, mask = points[::stride], mask[::stride]
        shown += len(points)
        for covered, color in ((True, "#286da8"), (False, "#c4463c")):
            selected = points[mask == covered]
            right.scatter(*selected.T, s=1.8, alpha=.55, color=color, depthshade=False)
    theta, phi = np.linspace(0, 2*np.pi, 18), np.linspace(0, np.pi, 11)
    x, y = np.outer(np.cos(theta), np.sin(phi)), np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.ones_like(theta), np.cos(phi))
    for center, radius in zip(render["hand_centers"], render["hand_radii"]):
        local = (center-origin)@rotation
        right.plot_wireframe(local[0]+radius*x, local[1]+radius*y, local[2]+radius*z,
            rstride=3, cstride=3, color="#7b8d9c", linewidth=.4, alpha=.35)
    right.set(xlabel="Actual hand-frame X (m)", ylabel="Actual hand-frame Y (m)", zlabel="Actual hand-frame Z (m)",
        title="Actual cooked hand/finger samples versus 18 hand spheres\nBlue coverage uses the whole 55-sphere union")
    right.set_box_aspect((1., 1.5, 1.))
    right.view_init(elev=17, azim=-53)
    right.legend(handles=[Line2D([], [], color="#286da8", marker=".", linestyle="", label="Inside planning sphere union"),
        Line2D([], [], color="#c4463c", marker=".", linestyle="", label="Outside planning sphere union")],
        fontsize=8, loc="upper left")
    figure.suptitle("Measured physical collision-model coverage · not solid-volume or continuous-safety certification", fontsize=13)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=165, bbox_inches="tight")
    plt.close(figure)
    return dict(path=str(destination), sha256=digest(destination), selected_case=render["case"],
        physics_step=render["physics_step"], plotted_surface_samples=shown,
        source_artifacts=case["source_artifacts"],
        scope="Measured cooked hand/finger triangle centroids and actual installed planning spheres; blue/red indicates sampled union membership. Wireframes are the18 hand-frame spheres, not an added collider. No runtime/model or physical-scene changes.")


def audit_home_self_clearance(reference, traces, model, state_count=6):
    """Independent cooked-solid distances at bounded actual Home body states."""
    from scripts.obstacle_representation_v3_cylinder import convex_distance_details
    reference = Path(reference)
    capture = json.loads(reference.read_text())
    geometry_path = reference.with_name("geometry_reference.npz")
    parts = []
    with np.load(geometry_path, allow_pickle=False) as archive:
        index = 0
        for record in capture["metrics"]["geometry_reference"]["cooking"]:
            for _ in range(record["cooked_hulls"]):
                vertices = body_mesh_vertices(archive[f"vertices_{index}"], archive[f"body_from_mesh_{index}"])
                parts.append((record["body"], vertices))
                index += 1
    results = []
    for path in traces:
        path = Path(path)
        trace, source = json.loads(path.read_text()), json.loads(path.with_name("result.json").read_text())
        scene = "scenes/multiview_perception_final.usda"
        if source["hashes"].get(scene) != capture["hashes"].get(scene):
            raise ValueError("Actual Home and cooked references use different physical scene sources")
        body_paths = tuple(trace["metadata"]["body_paths"])
        body_indices = {name: index for index, name in enumerate(body_paths)}
        samples = [row for row in trace["samples"] if row.get("phase") == "home"]
        selected = unique_state_indices(samples, state_count)
        minima, errors = {}, []
        queries, max_certificate = 0, 0.
        for index in selected:
            row = samples[index]
            physical = {name: pose_matrix_xyzw(row["poses"][slot]) for name, slot in body_indices.items()}
            q = np.asarray(row["feedback"]["positions"][:7])
            base = physical["/Franka/panda_link0"]
            for name in physical:
                frame = name.rsplit("/", 1)[-1]
                if frame in FINGERS:
                    continue
                expected = base@_native_pose(model["kinematics"], q, frame)
                actual = physical[name]
                errors.append(dict(frame=frame, translation_m=float(np.linalg.norm(expected[:3, 3]-actual[:3, 3])),
                    rotation_max=float(np.max(np.abs(expected[:3, :3]-actual[:3, :3])))))
            world = [(body, points@physical[body][:3, :3].T+physical[body][:3, 3]) for body, points in parts]
            for first, (body1, vertices1) in enumerate(world):
                for body2, vertices2 in world[first+1:]:
                    pair = body1.rsplit("/", 1)[-1], body2.rsplit("/", 1)[-1]
                    if body1 == body2 or frozenset(pair) in model["exclusions"]:
                        continue
                    key = tuple(sorted(pair))
                    previous = minima.get(key, dict(distance_m=math.inf))
                    lower1, upper1 = vertices1.min(0), vertices1.max(0)
                    lower2, upper2 = vertices2.min(0), vertices2.max(0)
                    bound = float(np.linalg.norm(np.maximum(np.maximum(lower1-upper2, lower2-upper1), 0.)))
                    if bound >= previous["distance_m"]:
                        continue
                    value = convex_distance_details(vertices1, vertices2)
                    queries += 1
                    max_certificate = max(max_certificate, value["upper_bound_m"]-value["lower_bound_m"])
                    if value["distance_m"] < previous["distance_m"]:
                        minima[key] = dict(distance_m=value["distance_m"], physics_step=row["physics_step"])
        rows = [dict(pair=list(pair), **value) for pair, value in sorted(minima.items())]
        results.append(dict(case=path.parent.parent.name, states=len(selected), nonadjacent_pairs=len(rows),
            physical_self_collision_feedback=trace["metadata"].get("initial_dynamics", {}).get("get_enabled_self_collisions"),
            physical_self_collision_feedback_provenance="Installed articulation-view public getter reads the USD-authored self-collision setting; not independently verified through a native PhysX getter or collision response.",
            minimum_cooked_solid_distance_m=min(row["distance_m"] for row in rows),
            zero_clearance_pairs=[row for row in rows if row["distance_m"] == 0.],
            pair_minima=rows, convex_distance_queries=queries,
            maximum_numerical_certificate_width_m=max_certificate,
            arm_hand_fk_translation_error_max_m=max(row["translation_m"] for row in errors),
            arm_hand_fk_rotation_matrix_error_max=max(row["rotation_max"] for row in errors),
            source_artifacts={str(p): digest(p) for p in (path, path.with_name("result.json"))}))
    return dict(schema_version=1, status="MEASURED_BOUNDED_ACTUAL_SELF_GEOMETRY", cases=results,
        cooked_reference=dict(path=str(geometry_path), sha256=digest(geometry_path),
            result_path=str(reference), result_sha256=digest(reference)),
        urdf_sha256=digest(model["urdf"]), producer_sha256=digest(__file__),
        exclusion_policy="The same URDF-derived rigid/adjacent-frame mask as the planner; no exclusions learned from these poses.",
        limitation="Actual copied cooked convex solids and physical body poses; six finite Home states per case, not a continuous sweep or contact-force measurement. Zero clearance does not distinguish touching from penetration. These distances do not alter self-collision filtering or the physical simulation.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--subdivisions", type=int, default=4)
    parser.add_argument("--hand-refinement", type=int, default=8)
    parser.add_argument("--isaac-root", type=Path, default=Path(os.environ.get("ISAAC_SIM", str(Path.home()/"isaacsim"))))
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT/"outputs/motion-planning-v4-1/collision-audit")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--self-traces", nargs="*", type=Path)
    parser.add_argument("--self-result", type=Path)
    args = parser.parse_args(argv)
    if args.result.exists() or (args.artifacts_dir/"full_audit.json").exists():
        raise ValueError("Refusing to overwrite a recorded collision audit")
    began = time.perf_counter()
    model = load_native_model(args.isaac_root)
    brief, raw, first_render = [], [], None
    for path in args.cases:
        try:
            compact, details, render = audit_case(path, model, state_count=args.states,
                subdivisions=args.subdivisions, refinement=args.hand_refinement)
        finally:
            clear_numeric_scene(model)
        brief.append(compact)
        raw.append(details)
        if first_render is None:
            first_render = render
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.artifacts_dir/"full_audit.json"
    raw_path.write_text(json.dumps(raw, separators=(",", ":"), allow_nan=False)+"\n")
    output = dict(schema_version=1, status="PASS_BOUNDED_COLLISION_MODEL_AUDIT", cases=brief,
        coverage_assessment="INCOMPLETE_PHYSICAL_COLLISION_COVERAGE; no conservative solid-coverage claim",
        runtime_model_changed=False,
        method=dict(surface_quadrature="One centroid per equal-area barycentric subtriangle; area-weighted finite-sampling estimate, not exact area or solid-volume coverage.",
            subdivisions=args.subdivisions, hand_refinement=args.hand_refinement, states_per_case=args.states,
            sphere_inclusion_numerical_tolerance_m=1e-9,
            witness="Native Lula query against an analysis-only <=2mm ball centered on a real cooked-finger surface sample; no physical scene authored or motion commanded."),
        source_sha256={str(model[key]): digest(model[key]) for key in ("descriptor", "urdf")},
        producer_sha256=digest(__file__), raw_artifact=dict(path=str(raw_path), sha256=digest(raw_path)),
        coverage_compute_seconds=time.perf_counter()-began,
        limitations=["Finite surface samples do not establish conservative solid coverage or continuous-time collision safety.",
            "A configuration model-gap witness is not a complete validated route or an observed contact in the original scenes.",
            "Only original recorded static/narrow/large seed123456 states are sampled; no global robot-model redesign or runtime change.",
            "No force/impulse or position-drive behavior is inferred from geometric coverage."])
    physical_names = {row["body"].rsplit("/", 1)[-1] for row in brief[0]["bodies"]}
    output["self_collision_policy"] = dict(method="Existing URDF fixed-joint collapse and immediately adjacent links; never learned from a measured pose.",
        excluded_physical_frame_pairs=sorted(sorted(pair) for pair in model["exclusions"] if set(pair) <= physical_names),
        limitation="No finger-frame spheres exist, so this sphere inspector does not independently establish physical finger self-collision coverage.")
    if args.figure:
        output["figure"] = plot_coverage(brief, first_render, args.figure)
    if args.self_traces:
        if args.self_result is None or args.self_result.exists():
            raise ValueError("Fresh --self-result required with --self-traces")
        self_report = audit_home_self_clearance(args.cases[0], args.self_traces, model)
        args.self_result.parent.mkdir(parents=True, exist_ok=True)
        args.self_result.write_text(json.dumps(self_report, indent=2, allow_nan=False)+"\n")
        output["home_self_geometry"] = dict(path=str(args.self_result), sha256=digest(args.self_result))
    output["wall_seconds_including_optional_plot_and_self_audit"] = time.perf_counter()-began
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status=output["status"], cases=len(brief),
        seconds=output["wall_seconds_including_optional_plot_and_self_audit"])))


if __name__ == "__main__":
    main()
