"""Evaluate registered voxel cuboids against captured physical actors.

This is an offline diagnostic. Fixture geometry never enters runtime planning.
Coverage samples refer to complete physical box surfaces, not the percentage
of camera-visible evidence retained (which requires full input captures).
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import tomllib
import numpy as np


def rotation_xyzw(quaternion):
    q = np.asarray(quaternion, float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Physical actor rotation needs a finite nonzero XYZW quaternion')
    x, y, z, w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def surface_samples(dimensions, pitch):
    dimensions = np.asarray(dimensions, float)
    if dimensions.shape != (3,) or np.any(dimensions <= 0) or pitch <= 0:
        raise ValueError('Surface sampling requires positive XYZ dimensions and pitch')
    coordinates = [(np.arange(int(np.ceil(size/pitch)))+.5)/int(np.ceil(size/pitch))*size-size/2
                   for size in dimensions]
    faces = []
    for fixed in range(3):
        other = [axis for axis in range(3) if axis != fixed]
        first, second = np.meshgrid(coordinates[other[0]], coordinates[other[1]], indexing='ij')
        for sign in (-1, 1):
            face = np.zeros((first.size, 3))
            face[:, fixed] = sign*dimensions[fixed]/2
            face[:, other[0]], face[:, other[1]] = first.ravel(), second.ravel()
            faces.append(face)
    return np.concatenate(faces)


def decode_registry(registrations, lower, upper, voxel_size):
    shape = np.ceil((upper-lower)/voxel_size-1e-10).astype(int)
    boxes = []
    for identity in registrations:
        values = identity.split(':')
        if len(values) != 6:
            raise ValueError('Registry identity is not an exact voxel-bounds descriptor: '+identity)
        bounds = np.asarray([int(value) for value in values])
        first, stop = bounds[[0, 2, 4]], bounds[[1, 3, 5]]
        if np.any(first < 0) or np.any(stop <= first) or np.any(stop > shape):
            raise ValueError('Registry bounds lie outside the configured occupancy grid')
        boxes.append(dict(id=identity, first=first, stop=stop,
            lower=lower+first*voxel_size, upper=np.minimum(lower+stop*voxel_size, upper),
            voxel_count=int(np.prod(stop-first))))
    return boxes


def point_union_membership(points, boxes):
    result = np.zeros(len(points), bool)
    for box in boxes:
        result |= ((points >= box['lower']-1e-9) & (points <= box['upper']+1e-9)).all(1)
    return result


def physical_coverage(boxes, dimensions, physical_pose, lower, voxel_size, surface_pitch):
    center = np.asarray(physical_pose[:3], float)
    rotation = rotation_xyzw(physical_pose[3:])
    half = np.asarray(dimensions)/2
    points = surface_samples(dimensions, surface_pitch) @ rotation.T+center
    covered = point_union_membership(points, boxes)
    # A physical surface point can quantize to a voxel whose center lies up to
    # half the cell diagonal outside the solid. This is an evaluation envelope,
    # never padding of runtime obstacle geometry.
    margin = np.sqrt(3)*voxel_size/2+1e-6
    actor_extent = np.abs(rotation)@half
    actor_low, actor_high = center-actor_extent-margin, center+actor_extent+margin
    associated, intersecting = set(), 0
    for box in boxes:
        first = np.maximum(box['first'], np.floor((actor_low-lower)/voxel_size).astype(int))
        stop = np.minimum(box['stop'], np.ceil((actor_high-lower)/voxel_size).astype(int))
        if np.any(stop <= first):
            continue
        indices = np.stack(np.meshgrid(*(np.arange(a, b) for a, b in zip(first, stop)), indexing='ij'), -1).reshape(-1, 3)
        centers = lower+(indices+.5)*voxel_size
        local = (centers-center)@rotation
        distance = np.linalg.norm(np.maximum(np.abs(local)-half, 0.), axis=1)
        chosen = indices[distance <= margin]
        if len(chosen):
            intersecting += 1
            associated.update(map(tuple, chosen.tolist()))
    return dict(complete_surface_sample_count=len(points), covered_surface_sample_count=int(covered.sum()),
        complete_surface_sample_coverage=float(covered.mean()),
        proxies_associated_with_actor=intersecting,
        represented_voxel_centers_inside_or_within_half_diagonal=len(associated)), associated


def summarize_case(folder):
    folder = Path(folder)
    raw = json.loads((folder/'result.json').read_text())
    metrics = raw['metrics']
    trace = json.loads((folder/'trace.json').read_text())
    config = tomllib.loads((folder/'runtime_profile.toml').read_text())
    mapping = config['spatial_mapping']
    lower, upper = np.array(mapping['bounds_min']), np.array(mapping['bounds_max'])
    size = float(mapping['voxel_size'])
    plane = config.get('scene', {}).get('support_plane_z', 0.)
    samples = trace['samples']
    times = [sample['at'] for sample in samples]
    bodies = metrics['capture']['body_paths']
    objects = metrics['configuration']['objects']
    rows = []
    for update in trace['sdk_updates']:
        diagnostics = update.get('diagnostics')
        if not diagnostics:
            continue
        registrations = diagnostics.get('registry', {}).get('registrations', {})
        boxes = decode_registry(registrations, lower, upper, size)
        index = max(0, bisect.bisect_right(times, update['at'])-1)
        sample = samples[index]
        floors = [box for box in boxes if box['upper'][2] <= plane+size+1e-12]
        row = dict(generation=update['generation'], revision=update['revision'],
            original_timestamp=update['original_timestamp'], map_age_seconds=update['map_age_seconds'],
            registered_proxy_count=len(boxes), floor_only_proxy_count=len(floors),
            above_plane_proxy_count=len(boxes)-len(floors),
            represented_source_cell_count=sum(box['voxel_count'] for box in boxes),
            physical_sample_time=sample['at'], physical_sample_simulation_time=sample['simulation_time'],
            physical_sample_time_error_ms=abs(sample['at']-update['at'])*1000, objects={})
        useful = set()
        for actor in objects:
            body = next(path for path in bodies if path.endswith('/'+actor['name']))
            pose = sample['poses'][bodies.index(body)]
            scores, cells = physical_coverage(boxes, actor['dimensions'], pose, lower, size, size)
            row['objects'][actor['name']] = dict(actual_physical_pose_xyzw=pose, **scores)
            useful.update(cells)
        row['represented_source_cells_associated_with_any_fixture_actor'] = len(useful)
        rows.append(row)
    return dict(schema_version=1, status='OFFLINE_PHYSICAL_PROXY_COVERAGE', case_dir=str(folder),
        result_sha256=hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),
        source=metrics['configuration']['mapping_source'], strategy=metrics['configuration']['strategy'],
        registry_capture_applicable=metrics['configuration']['strategy']=='reactive',
        direct_qualification='Direct has no runtime obstacle product/registrations; empty coverage rows are N/A, not zero camera evidence.' if metrics['configuration']['strategy']=='direct' else None,
        physical_pose_units='The fixed V2 scene has metersPerUnit=1; captured actor translations and configured fixture dimensions are meters.',
        scene=metrics['configuration'].get('objects'), surface_sample_pitch_m=size,
        qualifications=[
            'The full registry ID set is available only when the diagnostic records it; rows do not certify continuous coverage between captured changes.',
            'Physical poses are actual captured body transforms at or immediately preceding registry consumption, with matching time error reported. They are not substituted into runtime perception.',
            'Complete physical surface coverage includes hidden/unseen faces and is not a camera-visible evidence-retention fraction.',
            'The half-voxel-diagonal envelope labels source cells that quantize a physical surface. It adds no padding, occupied cells or UNKNOWN geometry to planning.',
            'Expiry may change registrations within an unchanged map revision. Report only the registry snapshots actually captured.'], registry_snapshots=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    result = summarize_case(args.case_dir)
    path = Path(args.result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
