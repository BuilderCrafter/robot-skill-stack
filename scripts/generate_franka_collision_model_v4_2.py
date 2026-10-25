#!/usr/bin/env python3
"""Generate project-owned Lula spheres from copied PhysX-cooked convex solids.

Generation is offline: no SimulationApp, USD mutation or native Lula import.
SciPy and PyYAML are generation tools, never production planner dependencies.
Every moving-body convex cell is contained in one sphere, which certifies its
solid interior by convexity. Surface quadrature is not used as a certificate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ISAAC_ROOT = Path('/home/etfrobotics/isaacsim')
REFERENCE = ROOT/'outputs/motion-planning-v4-1/native/global-static-seed123456/result.json'
DESTINATION = ROOT/'config/motion/franka_enhanced_v4_2.yaml'
FINGER_NAMES = ('panda_finger_joint1', 'panda_finger_joint2')
FINGER_BODIES = ('panda_leftfinger', 'panda_rightfinger')
FINGER_RANGE = (.001, .005)
# Explicit allowance for measured submicrometre URDF/PhysX frame disagreement
# and floating-point clipping/fitting. This is not a tracking-error envelope.
CONTAINMENT_MARGIN = 1e-5
STEPS = dict(arm=(.060,), hand=(.060, .050), finger=(.020,))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()


def vertices_array(value):
    points = np.asarray(value, dtype=float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 4
            or not np.isfinite(points).all() or np.linalg.matrix_rank(points-points.mean(0)) != 3):
        raise ValueError('A finite three-dimensional convex solid is required')
    return points


def cooked_body_vertices(vertices, body_from_mesh):
    points = vertices_array(vertices)
    transform = np.asarray(body_from_mesh, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
            or abs(np.linalg.det(transform[:3, :3])) < 1e-15):
        raise ValueError('A finite invertible affine mesh-to-body transform is required')
    return points@transform[:3, :3].T+transform[:3, 3]


def clip_convex_vertices(vertices, normal, bound, *, upper):
    """Vertices of a convex solid intersected with one closed halfspace.

    Every clipped-polytope vertex is either an original retained vertex or an
    intersection of the plane with an original edge. Triangulated hull edges
    include every polytope edge; extra face diagonals add only interior points.
    """
    from scipy.spatial import ConvexHull

    points = vertices_array(vertices)
    normal = np.asarray(normal, dtype=float)
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or np.linalg.norm(normal) < 1e-15 or not np.isfinite(bound)):
        raise ValueError('A finite plane is required')
    hull = ConvexHull(points)
    faces = hull.simplices
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                              faces[:, [2, 0]])), axis=1), axis=0)
    projected = points@normal
    kept = points[projected <= bound+1e-12 if upper else projected >= bound-1e-12]
    first, last = projected[edges[:, 0]], projected[edges[:, 1]]
    crossings = edges[((first < bound) & (last > bound)) | ((last < bound) & (first > bound))]
    weights = (bound-projected[crossings[:, 0]])/(projected[crossings[:, 1]]-projected[crossings[:, 0]])
    extra = points[crossings[:, 0]]+weights[:, None]*(points[crossings[:, 1]]-points[crossings[:, 0]])
    result = np.unique(np.concatenate((kept, extra)), axis=0)
    if len(result) < 4 or np.linalg.matrix_rank(result-result.mean(0)) < 3:
        return None  # Empty or measure-zero cells do not add solid volume.
    return result


def principal_axes(vertices):
    values, axes = np.linalg.eigh(np.cov(vertices_array(vertices).T))
    axes = axes[:, np.argsort(values)[::-1]]
    for index in range(3):
        if axes[np.argmax(np.abs(axes[:, index])), index] < 0:
            axes[:, index] *= -1
    return axes


def convex_cells(vertices, widths):
    """Partition one convex hull by a fixed principal-coordinate grid."""
    points = vertices_array(vertices)
    axes = principal_axes(points)
    if not widths or len(widths) > 3 or not np.isfinite(widths).all() or min(widths) <= 0:
        raise ValueError('One to three positive finite cell widths are required')
    cells = [points]
    for dimension, width in enumerate(widths):
        axis = axes[:, dimension]
        projected = points@axis
        cuts = np.linspace(projected.min(), projected.max(), int(np.ceil(np.ptp(projected)/width))+1)
        result = []
        for cell in cells:
            for low, high in zip(cuts[:-1], cuts[1:]):
                clipped = clip_convex_vertices(cell, axis, low, upper=False)
                if clipped is not None:
                    clipped = clip_convex_vertices(clipped, axis, high, upper=True)
                if clipped is not None:
                    result.append(clipped)
        cells = result
    return cells, axes


def enclosing_sphere(vertices, margin=CONTAINMENT_MARGIN):
    """Minimax fit plus independently checked frame/numerical allowance."""
    from scipy.optimize import minimize

    points = vertices_array(vertices)
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError('A positive finite containment allowance is required')
    center = (points.min(0)+points.max(0))/2
    initial = np.r_[center, np.sum((points-center)**2, axis=1).max()]
    result = minimize(lambda value: value[3], initial, jac=lambda value: np.array([0., 0., 0., 1.]),
        constraints={'type': 'ineq',
                     'fun': lambda value: value[3]-np.sum((points-value[:3])**2, axis=1),
                     'jac': lambda value: np.c_[2*(points-value[:3]), np.ones(len(points))]},
        method='SLSQP', options={'ftol': 1e-12, 'maxiter': 100})
    if result.success and np.isfinite(result.x).all():
        center = result.x[:3]
    radius = float(np.linalg.norm(points-center, axis=1).max()+margin)
    return center, radius


def certify_cell(vertices, center, radius):
    """All vertices must fit the SAME sphere; union vertex coverage is weaker."""
    points = vertices_array(vertices)
    center = np.asarray(center, float)
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError('A finite sphere is required')
    maximum = float(np.linalg.norm(points-center, axis=1).max())
    if maximum > radius:
        raise ValueError('Convex cell vertices are not contained in one sphere')
    return dict(vertices=len(points), vertices_sha256=array_digest(points),
                maximum_vertex_distance_m=maximum, radius_m=float(radius),
                minimum_containment_margin_m=float(radius-maximum))


def finger_transforms(urdf_path, midpoint):
    """Read the actual finger URDF attachment; never assume nominal opening."""
    document = ET.parse(urdf_path).getroot()
    result = {}
    for joint_name, body in zip(FINGER_NAMES, FINGER_BODIES):
        elements = [entry for entry in document.findall('joint') if entry.get('name') == joint_name]
        if len(elements) != 1:
            raise ValueError('Unique finger joints are required')
        entry = elements[0]
        if (entry.get('type') != 'prismatic' or entry.find('parent').get('link') != 'panda_hand'
                or entry.find('child').get('link') != body):
            raise ValueError('Unexpected finger attachment')
        origin = entry.find('origin')
        translation = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        axis = np.fromstring(entry.find('axis').get('xyz'), sep=' ')
        if (translation.shape != (3,) or rpy.shape != (3,) or axis.shape != (3,)
                or not np.isfinite(np.r_[translation, rpy, axis]).all()
                or not np.allclose(rpy, 0, atol=1e-12, rtol=0)
                or not np.isclose(np.linalg.norm(axis), 1, atol=1e-12, rtol=0)):
            raise ValueError('This frozen model requires its measured unrotated finger attachments')
        transform = np.eye(4)
        transform[:3, 3] = translation+midpoint*axis
        result[body] = (transform, axis)
    return result


@dataclass
class CookedPart:
    body: str
    hull_index: int
    vertices: np.ndarray


def load_reference(result_path):
    result_path = Path(result_path)
    payload = json.loads(result_path.read_text())
    result = payload['metrics']
    if payload['status'] != 'PASS' or not result['geometry_reference']['cooked_only']:
        raise ValueError('A valid independent PhysX-cooked reference is required')
    geometry = result_path.parent/'geometry_reference.npz'
    parts = []
    with np.load(geometry, allow_pickle=False) as archive:
        for record in result['geometry_reference']['cooking']:
            for _ in range(record['cooked_hulls']):
                index = len(parts)
                parts.append(CookedPart(record['body'].rsplit('/', 1)[-1], index,
                    cooked_body_vertices(archive[f'vertices_{index}'], archive[f'body_from_mesh_{index}'])))
    if len(parts) != result['geometry_reference']['meshes']:
        raise ValueError('Every actual cooked hull must be represented')
    return parts, dict(result_sha256=sha256(result_path), geometry_reference_sha256=sha256(geometry),
        result_path=str(result_path.relative_to(ROOT)) if result_path.is_relative_to(ROOT) else str(result_path),
        geometry_path=str(geometry.relative_to(ROOT)) if geometry.is_relative_to(ROOT) else str(geometry))


def generate(stock_document, parts, urdf_path):
    from scipy.spatial import ConvexHull

    document = deepcopy(stock_document)
    groups = {name: rows for entry in document['collision_spheres'] for name, rows in entry.items()}
    stock_counts = {name: len(rows) for name, rows in groups.items()}
    stock_count = sum(map(len, groups.values()))
    midpoint = sum(FINGER_RANGE)/2
    sweep_padding = (FINGER_RANGE[1]-FINGER_RANGE[0])/2
    transforms = finger_transforms(urdf_path, midpoint)
    for rule in document['cspace_to_urdf_rules']:
        if rule['name'] in FINGER_NAMES:
            rule['value'] = midpoint
    certificates = []
    base = None
    finger_intervals = {}
    for part in parts:
        if part.body == 'panda_link0':
            if base is not None:
                raise ValueError('The frozen base certificate expects one cooked hull')
            low, high = part.vertices.min(0), part.vertices.max(0)
            center, radius = enclosing_sphere(part.vertices)
            base = dict(base_link=part.body,
                base_enclosing_sphere=dict(center=center.tolist(), radius=radius),
                base_aabb_corners_m=[list(corner) for corner in itertools.product(*zip(low, high))],
                base_aabb_minimum_m=low.tolist(), base_aabb_maximum_m=high.tolist(),
                base_sphere_certificate=certify_cell(part.vertices, center, radius),
                base_convex_volume_m3=float(ConvexHull(part.vertices).volume),
                base_aabb_certificate='Every cooked vertex lies in these coordinate bounds; convexity contains the whole solid. Bounds are uninflated and conditional on the captured mounted-base frame alignment; no penetration allowance is introduced below the support plane.')
            continue
        is_finger = part.body in FINGER_BODIES
        category = 'finger' if is_finger else 'hand' if part.body == 'panda_hand' else 'arm'
        cells, axes = convex_cells(part.vertices, STEPS[category])
        original_volume = float(ConvexHull(part.vertices).volume)
        partition_volume = float(sum(ConvexHull(cell).volume for cell in cells))
        relative_volume_error = abs(partition_volume-original_volume)/original_volume
        if relative_volume_error > 1e-8:
            raise ValueError('Clipped-cell partition does not preserve the source solid volume')
        target_frame = part.body
        rows = groups.setdefault(target_frame, [])
        cell_proofs = []
        before = len(rows)
        for cell in cells:
            local = cell
            # Reuse a sphere only when it covers this complete cell (including
            # the declared finger sweep), never from vertex-union coverage.
            covering = next((index for index, row in enumerate(rows)
                if np.linalg.norm(local-np.asarray(row['center']), axis=1).max()+
                    (sweep_padding if is_finger else 0.) <= row['radius']), None)
            if covering is None:
                center, radius = enclosing_sphere(local)
                if is_finger:
                    radius += sweep_padding
                rows.append(dict(center=center.tolist(), radius=float(radius)))
                covering = len(rows)-1
            row = rows[covering]
            proof = certify_cell(local, row['center'], row['radius']-(sweep_padding if is_finger else 0.))
            proof.update(sphere_frame=target_frame, sphere_index_on_frame=covering,
                swept_axis_padding_m=sweep_padding if is_finger else 0.)
            cell_proofs.append(proof)
        if is_finger:
            transform, axis = transforms[part.body]
            nominal = part.vertices@transform[:3, :3].T+transform[:3, 3]
            finger_intervals[part.body] = [float(nominal[:, 1].min()-sweep_padding*abs(axis[1])),
                                          float(nominal[:, 1].max()+sweep_padding*abs(axis[1]))]
        certificates.append(dict(body=part.body, source_hull_index=part.hull_index,
            source_body_vertices_sha256=array_digest(part.vertices),
            partition_widths_m=list(STEPS[category]), principal_axes=axes.tolist(),
            original_convex_volume_m3=original_volume, cell_volume_sum_m3=partition_volume,
            partition_relative_volume_error=relative_volume_error,
            convex_cells=len(cells), added_spheres=len(rows)-before, cells=cell_proofs))
    if base is None or set(finger_intervals) != set(FINGER_BODIES):
        raise ValueError('Base and both independently cooked fingers are required')
    separating_gap = finger_intervals['panda_leftfinger'][0]-finger_intervals['panda_rightfinger'][1]
    if separating_gap <= 0:
        raise ValueError('The operating range does not establish positive physical finger separation')
    document['collision_spheres'] = [{name: rows} for name, rows in groups.items() if rows]
    count = sum(map(len, groups.values()))
    metadata = dict(schema_version=1, model='franka_enhanced_v4_2', sphere_count=count,
        stock_sphere_count=stock_count, added_sphere_count=count-stock_count,
        sphere_groups=[dict(frame=name, stock_spheres=stock_counts.get(name, 0),
            certified_added_spheres=len(rows)-stock_counts.get(name, 0),
            role='finger' if name in FINGER_BODIES else 'hand' if name == 'panda_hand'
                else 'mounted_base' if name == 'panda_link0' else 'arm')
            for name, rows in groups.items() if rows],
        finger_joint_names=list(FINGER_NAMES), finger_range_m=list(FINGER_RANGE),
        descriptor_fixed_finger_m=midpoint, finger_sweep_padding_m=sweep_padding,
        numerical_containment_margin_m=CONTAINMENT_MARGIN,
        measured_frame_and_numerical_allowance_m=CONTAINMENT_MARGIN,
        finger_hand_y_intervals_m=finger_intervals,
        minimum_physical_finger_separating_projection_m=separating_gap,
        finger_separation_scope='Positive separation of both cooked finger solids along hand Y for every independent joint position in the declared range; excludes contact offsets.',
        certified_self_collision_exclusions=[dict(frames=list(FINGER_BODIES),
            minimum_separating_projection_m=separating_gap, numerical_tolerance_m=1e-9,
            urdf_sha256=sha256(urdf_path), finger_range_m=list(FINGER_RANGE),
            cooked_finger_vertices_sha256={part.body: array_digest(part.vertices)
                for part in parts if part.body in FINGER_BODIES},
            proof='The left cooked solid projects above the right cooked solid along hand Y over all independent finger positions in the declared interval. Common arm/hand rigid transforms preserve this separation. The conservative sphere overlap is therefore an approximation artifact.')],
        solid_coverage=dict(status='FINITE_CONVEX_CELL_CONTAINMENT',
            method='Each closed convex grid cell has every vertex inside one assigned sphere. Sphere convexity contains the entire cell; cells partition the source convex solid. Finger range follows the triangle inequality.',
            numerical_scope='Floating-point certificate conditional on the copied cooked vertex solids, unchanged source URDF, and the measured link-frame alignment remaining within the explicit allowance. This is not a formal all-world physical safety guarantee.',
            maximum_partition_relative_volume_error=max(x['partition_relative_volume_error'] for x in certificates),
            moving_body_count=len(certificates), convex_cell_count=sum(x['convex_cells'] for x in certificates),
            certificates=certificates),
        limitations=['Native Lula retains its original base sphere; the conservative base bound requires a separate frozen-base world preflight.',
            'Only the mutually overlapping finger-sphere pair receives an extra self exclusion, justified by the positive full-range cooked-solid separation certificate. Existing URDF fixed/adjacent exclusions remain.',
            'Added spheres include a 10 micrometre measured-frame/numerical allowance. The mounted-base AABB remains uninflated; its exact-corner support-plane check is conditional on the captured frame alignment.',
            'Certificates cover the captured cooked convex solids in aligned link frames, not visuals, contact/rest offsets, future asset revisions, tracking error or continuous swept trajectories.'], **base)
    return document, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--isaac-root', type=Path, default=ISAAC_ROOT)
    parser.add_argument('--output', type=Path, default=DESTINATION)
    args = parser.parse_args(argv)
    import yaml
    motion = args.isaac_root/'exts/isaacsim.robot_motion.motion_generation'
    descriptor = motion/'motion_policy_configs/franka/rmpflow/robot_descriptor.yaml'
    urdf = motion/'motion_policy_configs/franka/lula_franka_gen.urdf'
    parts, sources = load_reference(args.reference)
    document, metadata = generate(yaml.safe_load(descriptor.read_text()), parts, urdf)
    if metadata['stock_sphere_count'] != 55 or metadata['sphere_count'] != 102:
        raise ValueError('Frozen model source/partition changed; expected 55 stock plus 47 added spheres')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = '# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n# Stock descriptor fields retained; project-owned additions generated from independently captured PhysX hulls.\n# Regenerate with scripts/generate_franka_collision_model_v4_2.py; see sibling JSON certificate.\n'
    args.output.write_text(header+yaml.safe_dump(document, sort_keys=False))
    metadata.update(descriptor_sha256=sha256(args.output), stock_descriptor_sha256=sha256(descriptor),
        urdf_sha256=sha256(urdf), generator_sha256=sha256(__file__), source_artifacts=sources)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(sphere_count=metadata['sphere_count'],
        convex_cells=metadata['solid_coverage']['convex_cell_count'],
        minimum_finger_separation_m=metadata['minimum_physical_finger_separating_projection_m'],
        descriptor=str(args.output), metadata=str(args.output.with_suffix('.json')))))


if __name__ == '__main__':
    main()
