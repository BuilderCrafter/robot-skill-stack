"""Measured-box numerical reference, extracted unchanged from V5 research.

Only the module documentation differs; the original research copy stays frozen.
"""
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from robot_skill_stack.world.perception_multiview.clouds import (
    crop_workspace, project_depth, voxelize,
)
from robot_skill_stack.world.perception_multiview.fitting import (
    _box_distance, _box_search, _hull_area,
)
from robot_skill_stack.world.perception_multiview.observability import (
    _DEPTH_TOLERANCE, _EDGE_POSITIONS, _EDGE_SAMPLES, _PIXEL_NEIGHBORS,
    _RIBBON_OFFSETS, _unique_pixels,
)
from robot_skill_stack.world.perception_multiview.segmentation import cluster_points

TOL = .0015  # Existing SurfaceFitter tolerance.
DEPTH = _DEPTH_TOLERANCE
FIT_LIMIT = 1200
MODEL_LIMIT = 4
FACE_KEYS = [(a, s) for a in range(3) for s in (-1, 1)]
CORNERS = np.array(list(product((-1., 1.), repeat=3))) / 2
POLICY = dict(model='single convex yaw box; no support-height prior',
              fit_tolerance_m=TOL, score_min=.65, depth_tolerance_m=DEPTH,
              cloud_voxel_m=.002, cloud_max=12000, stride=2,
              candidate_max=16, model_max=MODEL_LIMIT, fit_point_max=FIT_LIMIT,
              robot_envelope_margin_m=.005, direct_face_min=6,
              direct_face_span_fraction=.25, boundary_bins=3,
              edge_samples=_EDGE_SAMPLES,
              hidden_bound='must have current calibrated occupied/free edge witnesses')


def plain(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False)+'\n')


def robot_envelopes(frames: Sequence, snapshots: Mapping) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Conservative component rejection, not a claim of exact robot masking."""
    assert set(snapshots) == {f.camera_id for f in frames}
    bounds = []
    for f in frames:
        s = snapshots[f.camera_id]
        if s.frame_token != f.token or abs(s.simulation_time-f.token[1]) > .01:
            raise ValueError('robot/frame synchronization mismatch')
        for part in s.parts:
            p = part.mesh.vertices @ part.world_from_local[:3, :3].T + part.world_from_local[:3, 3]
            bounds.append((p.min(0)-.005, p.max(0)+.005))
    return tuple(bounds)


def robot_overlap(points: np.ndarray, bounds: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    return np.logical_or.reduce([((points >= lo) & (points <= hi)).all(1)
                                 for lo, hi in bounds]) if bounds else np.zeros(len(points), bool)


def project_clouds(frames: Sequence, scene) -> np.ndarray:
    """Fuse workspace points from calibrated depth frames for box reconstruction."""
    clouds = [voxelize(crop_workspace(project_depth(f, stride=2,
              bounds=(scene.workspace_min, scene.workspace_max))[0], scene), .002, 12000)
              for f in frames]
    return voxelize(np.concatenate(clouds), .002, 12000)


def hypothesis(points: np.ndarray) -> dict | None:
    """Existing yaw search, with Z extrema used ONLY to propose free-box bounds."""
    p = points[np.linspace(0, len(points)-1, min(len(points), FIT_LIMIT), dtype=int)]
    high = float(np.percentile(p[:, 2], 99.8))
    upper = p[p[:, 2] > high-.010]
    levels, counts = np.unique(np.floor(upper[:, 2]/TOL).astype(int), return_counts=True)
    top = float(np.median(upper[np.floor(upper[:, 2]/TOL).astype(int) == levels[np.argmax(counts)], 2]))
    side = p[p[:, 2] < top-3*TOL]
    top_points = p[abs(p[:, 2]-top) < TOL]
    if len(side) < 15 or len(top_points) < 15:
        return None
    yaw, rotation, _, _ = _box_search(p, side, np.deg2rad(np.arange(0, 90, 5.)))
    for width, step in ((5., .5), (.5, .05)):
        yaw, rotation, _, _ = _box_search(p, side, yaw+np.deg2rad(np.arange(-width, width+step/2, step)))
    local = points @ rotation
    lo, hi = local.min(0), local.max(0)
    size = hi-lo
    if min(size) < .007:
        return None
    center = ((lo+hi)/2) @ rotation.T
    residual = float(np.percentile(_box_distance(p, center, rotation, size), 80))
    fill = _hull_area(top_points[:, :2])/max(float(np.prod(size[:2])), 1e-12)
    score = float(np.exp(-.5*(residual/TOL)**2)*np.clip((fill-.78)/.16, 0, 1))
    return dict(center=center, dimensions=size, yaw_rad=float(yaw), rotation=rotation,
                residual_p80_m=residual, top_fill=float(fill), score=score)


def face_support(local: np.ndarray, size: np.ndarray, axis: int, sign: int) -> dict:
    """Check whether visible points directly support a candidate box face."""
    other = [a for a in range(3) if a != axis]
    selected = ((abs(local[:, axis]-sign*size[axis]/2) <= TOL) &
                (abs(local[:, other]) < .35*size[other]).all(1))
    p = local[selected]
    out = dict(points=len(p), supported=False, interval_local_m=None, spread_m=None)
    if len(p) < 6 or np.any(np.ptp(p[:, other], axis=0) < .25*size[other]):
        return out
    design = np.c_[p[:, other], np.ones(len(p))]
    plane, _, rank, _ = np.linalg.lstsq(design, p[:, axis], rcond=None)
    spread = float(np.percentile(abs(design@plane-p[:, axis]), 95))
    if rank != 3 or spread > DEPTH or np.sum(abs(plane[:2])*size[other]) > 2*DEPTH:
        return dict(out, spread_m=spread)
    return dict(out, supported=True, spread_m=spread,
                plane_local_m=float(plane[-1]),
                interval_local_m=[float(p[:, axis].min()-DEPTH), float(p[:, axis].max()+DEPTH)])


def boundary(model: dict, frame, target: tuple[int, int], carrier: tuple[int, int],
             carrier_support: dict) -> dict:
    """Existing bounded ribbon method generalized to an observed carrier plane.

    The target's hidden face is inferred only under the single-box model.
    A nearer occluder or invalid pixel supplies no free-space witness.
    """
    a, sign = target
    b, carrier_sign = carrier
    c = next(x for x in range(3) if x not in (a, b))
    size, center, r = model['dimensions'], model['center'], model['rotation']
    origin = (frame.world_from_camera[:3, 3]-center) @ r
    result = dict(camera_id=frame.camera_id, carrier=f'{"xyz"[b]}{carrier_sign:+d}',
                  bounded=False, bins_supported=0, transitions=0, free=0, occupied=0,
                  foreground=0, missing=0, outer_local_m=None, witnesses=[])
    plane = carrier_support['plane_local_m']
    if carrier_sign*(origin[b]-plane) <= 0:
        return dict(result, reason='carrier_back_facing')
    edge = np.zeros((_EDGE_SAMPLES, 3))
    edge[:, a], edge[:, b], edge[:, c] = sign*size[a]/2, plane, _EDGE_POSITIONS*size[c]
    world = center+edge@r.T
    camera = (world-frame.world_from_camera[:3, 3]) @ frame.world_from_camera[:3, :3]
    if np.any(camera[:, 2] <= 0):
        return dict(result, reason='out_of_frame')
    k = frame.intrinsics
    projected = camera[:, :2]/camera[:, 2, None]*[k[0, 0], k[1, 1]]+k[:2, 2]-.5
    pixels = _unique_pixels((np.rint(projected).astype(int)[:, None]+_RIBBON_OFFSETS).reshape(-1, 2))
    h, w = frame.depth.shape
    pixels = pixels[((pixels >= 0) & (pixels < [w, h])).all(1)]
    rays = np.c_[(pixels[:, 0]+.5-k[0, 2])/k[0, 0],
                 (pixels[:, 1]+.5-k[1, 2])/k[1, 1], np.ones(len(pixels))]
    directions = rays @ frame.world_from_camera[:3, :3].T @ r
    with np.errstate(divide='ignore', invalid='ignore'):
        expected = (plane-origin[b])/directions[:, b]
        q = origin+directions*expected[:, None]
    keep = ((expected > 0) & np.isfinite(q).all(1) & (abs(q[:, c]) <= .38*size[c]))
    pixels, q, directions, expected = [v[keep] for v in (pixels, q, directions, expected)]
    if not len(pixels):
        return dict(result, reason='no_usable_ribbon')
    measured = frame.depth[pixels[:, 1], pixels[:, 0]]
    valid = np.isfinite(measured) & (measured > 0)
    free = valid & (measured > expected+DEPTH)
    foreground = valid & (measured < expected-DEPTH)
    occupied = valid & ~free & ~foreground
    result.update(free=int(free.sum()), occupied=int(occupied.sum()),
                  foreground=int(foreground.sum()), missing=int((~valid).sum()))
    occupied &= sign*q[:, a] <= size[a]/2+DEPTH
    ids = pixels[:, 1]*w+pixels[:, 0]
    order = np.argsort(ids)
    free_ids = np.flatnonzero(free)
    if not len(free_ids) or not occupied.any():
        return dict(result, reason='no_occupied_free_pair')
    neighbors = pixels[free_ids, None]+_PIXEL_NEIGHBORS
    neighbor_ids = neighbors[..., 1]*w+neighbors[..., 0]
    positions = np.searchsorted(ids[order], neighbor_ids)
    selected = order[np.minimum(positions, len(order)-1)]
    pairs = ((positions < len(order)) & (ids[selected] == neighbor_ids) & occupied[selected] &
             (sign*(q[selected, a]-q[free_ids, None, a]) < 0) &
             ((neighbors >= 0) & (neighbors < [w, h])).all(2))
    witness = free_ids[pairs.any(1)]
    result['transitions'] = len(witness)
    outward = sign*q[witness, a]+DEPTH*(1+abs(directions[witness, a]/directions[witness, b]))
    bins, saved = [], []
    for low, high in ((-.38, -.1267), (-.1267, .1267), (.1267, .38)):
        selected = np.flatnonzero((q[witness, c] >= low*size[c]) & (q[witness, c] <= high*size[c]))
        if len(selected):
            best = selected[np.argmin(outward[selected])]
            bins.append(float(outward[best]))
            saved.append(dict(pixel_xy=pixels[witness[best]].tolist(),
                              intersection_local_m=q[witness[best]].tolist()))
    result.update(bins_supported=len(bins), witnesses=saved)
    if len(bins) == 3:
        result.update(bounded=True, outer_local_m=float(sign*max(size[a]/2, max(bins))),
                      reason='three_separated_occupied_free_transition_bins')
    else:
        result['reason'] = 'missing_boundary_bins'
    return result


def free_ray_check(model: dict, frames: Sequence) -> dict:
    """Fixed small surface grid; check current pixels without integrating rays."""
    size, center, r = model['dimensions'], model['center'], model['rotation']
    samples = []
    for a, sign in FACE_KEYS:
        other = [x for x in range(3) if x != a]
        for u, v in product((-.25, 0., .25), repeat=2):
            q = np.zeros(3)
            q[a], q[other[0]], q[other[1]] = sign*size[a]/2, u*size[other[0]], v*size[other[1]]
            samples.append(q)
    world = center+np.array(samples)@r.T
    out = dict(tested_pixels=0, contradictions=0, witnesses=[])
    # Test strict interior; the unchanged 2 mm measurement band is not FREE.
    half = size/2-DEPTH
    for f in frames:
        camera = (world-f.world_from_camera[:3, 3])@f.world_from_camera[:3, :3]
        camera = camera[camera[:, 2] > 0]
        k = f.intrinsics
        pixels = np.unique(np.rint(camera[:, :2]/camera[:, 2, None]*[k[0, 0], k[1, 1]]+k[:2, 2]-.5).astype(int), axis=0)
        h, w = f.depth.shape
        pixels = pixels[((pixels >= 0) & (pixels < [w, h])).all(1)]
        rays = np.c_[(pixels[:, 0]+.5-k[0, 2])/k[0, 0],
                     (pixels[:, 1]+.5-k[1, 2])/k[1, 1], np.ones(len(pixels))]
        origin = (f.world_from_camera[:3, 3]-center)@r
        directions = rays@f.world_from_camera[:3, :3].T@r
        with np.errstate(divide='ignore', invalid='ignore'):
            first, second = (-half-origin)/directions, (half-origin)/directions
        enter, leave = np.minimum(first, second).max(1), np.maximum(first, second).min(1)
        measured = f.depth[pixels[:, 1], pixels[:, 0]]
        valid = np.isfinite(measured) & (measured > 0) & (leave > np.maximum(enter, 0))
        conflicts = valid & (measured > enter+DEPTH)
        out['tested_pixels'] += int(valid.sum())
        out['contradictions'] += int(conflicts.sum())
        out['witnesses'].extend(dict(camera_id=f.camera_id, pixel_xy=pixels[i].tolist())
                                for i in np.flatnonzero(conflicts)[:3])
    return out


def qualify(points: np.ndarray, frames: Sequence,
            envelopes: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Reject robot-connected or weak box hypotheses and qualify measured bounds."""
    overlap = robot_overlap(points, envelopes)
    result = dict(points=len(points), robot_envelope_points=int(overlap.sum()),
                  status='rejected', reason=None, hypothesis=None, geometry=None, faces={})
    if overlap.any():
        return dict(result, reason='possible_robot_component')
    if len(points) < 30:
        return dict(result, reason='insufficient_points')
    model = hypothesis(points)
    if model is None:
        return dict(result, reason='insufficient_box_surfaces')
    result['hypothesis'] = model
    local = (points-model['center'])@model['rotation']
    supports = {key: face_support(local, model['dimensions'], *key) for key in FACE_KEYS}
    bounds = np.c_[-model['dimensions']/2, model['dimensions']/2]
    for key in FACE_KEYS:
        a, sign = key
        support = supports[key]
        edges = []
        if not support['supported']:
            edges = [boundary(model, f, key, carrier, supports[carrier])
                     for carrier in FACE_KEYS if carrier[0] != a and supports[carrier]['supported']
                     for f in frames]
        qualified = support['supported'] or any(e['bounded'] for e in edges)
        face = dict(qualified=bool(qualified), direct=support, boundary_trials=edges,
                    source='direct_face' if support['supported'] else 'boundary' if qualified else 'unqualified',
                    outer_local_m=None)
        if support['supported']:
            face['outer_local_m'] = support['interval_local_m'][int(sign > 0)]
        elif qualified:
            face['outer_local_m'] = float(sign*min(sign*e['outer_local_m'] for e in edges if e['bounded']))
        if qualified:
            bounds[a, int(sign > 0)] = face['outer_local_m']
        result['faces'][f'{"xyz"[a]}{sign:+d}'] = face
    result['free_rays'] = free_ray_check(model, frames)
    if result['free_rays']['contradictions']:
        return dict(result, reason='contradictory_current_free_rays')
    if model['score'] < .65:
        return dict(result, reason='unsupported_or_merged_box_surface')
    if not all(face['qualified'] for face in result['faces'].values()):
        return dict(result, status='unqualified', reason='unqualified_hidden_bound')
    center = model['center']+bounds.mean(1)@model['rotation'].T
    geometry = dict(center=center, dimensions=np.diff(bounds, axis=1).ravel(),
                    rotation=model['rotation'], yaw_rad=model['yaw_rad'],
                    approximation='rgbd_inferred_box', occupancy_completion=False)
    return dict(result, status='qualified', reason='all_six_empirical_model_bounds', geometry=geometry)


def reconstruct(frames: Sequence, scene, snapshots: Mapping) -> tuple[dict, list, list]:
    """Reconstruct qualified box geometry from synchronized frames and robot poses."""
    start = time.perf_counter()
    envelopes = robot_envelopes(frames, snapshots)
    envelope_end = time.perf_counter()
    cloud = project_clouds(frames, scene)
    clusters = cluster_points(cloud)
    projected_end = time.perf_counter()
    results, fits = [], 0
    for i, points in enumerate(sorted(clusters, key=len, reverse=True)[:16]):
        if robot_overlap(points, envelopes).any():
            q = qualify(points, frames, envelopes)
        elif fits >= MODEL_LIMIT:
            q = dict(points=len(points), status='rejected', reason='hypothesis_budget_exhausted')
        else:
            fits += 1
            q = qualify(points, frames, envelopes)
        results.append(dict(component=i, **q))
    end = time.perf_counter()
    return dict(policy=POLICY, source_timestamp=min(f.acquired_at for f in frames),
                source_frames=[dict(camera_id=f.camera_id, token=f.token, acquired_at=f.acquired_at,
                                    shape=f.depth.shape, intrinsics=f.intrinsics,
                                    world_from_camera=f.world_from_camera) for f in frames],
                total_cloud_points=len(cloud), cluster_count=len(clusters), numerical_hypotheses=fits,
                components=results,
                timing_ms=dict(robot_envelopes=(envelope_end-start)*1000,
                               existing_projection_fusion_clustering=(projected_end-envelope_end)*1000,
                               qualification=(end-projected_end)*1000, total=(end-start)*1000)), clusters, envelopes
