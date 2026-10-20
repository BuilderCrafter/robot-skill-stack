"""Reproducible iteration-two fixtures/scoring; never imported by perception.

Truth is used only to construct scenes and score/export diagnostics after the
automatic providers run. Selection and holdout use disjoint seeds/viewpoints.
"""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from tests.support.multiview_research import ResearchObject, metric_row


SEEDS = {'selection': 271828, 'holdout': 314159}
EYES = {
    'selection': ((1.65896, -.05281, 1.10358), (.22, -.78, .60)),
    'holdout': ((1.55, .12, 1.04), (.30, -.67, .72)),
}


@dataclass(frozen=True)
class RobustnessScene:
    name: str
    split: str
    seed: int
    objects: tuple
    clutter: tuple = ()
    gap_mm: float | None = None
    note: str = ''
    temporal: bool = False

    @property
    def key(self):
        return f'{self.split}_{self.name}'


def box(name, center, size, yaw=0.):
    size = np.asarray(size, float)
    return ResearchObject(name, 'cube', np.array([*center, size[2]/2]), size, yaw)


def sphere(name, center, radius):
    return ResearchObject(name, 'sphere', np.array([*center, radius]), np.full(3, 2*radius), radius=radius)


def cylinder(name, center, radius, length, yaw=None):
    axis = (0., 0., 1.) if yaw is None else (np.cos(yaw), np.sin(yaw), 0.)
    size = np.array([2*radius, 2*radius, length]) if yaw is None else np.array([length, 2*radius, 2*radius])
    return ResearchObject(name, 'cylinder', np.array([*center, length/2 if yaw is None else radius]),
                          size, radius=radius, length=length, axis=axis)


def footprint_half_extent(obj, direction):
    direction = np.asarray(direction, float)
    if obj.shape == 'cube':
        c, s = np.cos(obj.yaw), np.sin(obj.yaw)
        return float(abs(direction @ [c, s])*obj.size[0]/2 + abs(direction @ [-s, c])*obj.size[1]/2)
    if obj.shape == 'sphere' or abs(obj.axis[2]) > .99:
        return obj.radius
    axis = np.asarray(obj.axis[:2])
    return float(abs(direction @ axis)*obj.length/2 + abs(np.cross(direction, axis))*obj.radius)


def close_pair(kind, gap, rng, *, holdout):
    """The closest support planes are exactly gap apart, even for yawed boxes."""
    middle = np.array([.44, .01]) + rng.uniform(-.035, .035, 2)
    yaw = float(rng.uniform(-.8, .8))
    direction = np.array([np.cos(yaw), np.sin(yaw)])
    scale = float(rng.uniform(.85, 1.13))
    box_yaw = yaw  # One face normal follows the separation direction.
    if kind == 'box_box':
        first = box('target_A', middle, np.array([.043, .050, .047])*scale, box_yaw)
        second = box('target_B', middle, np.array([.057, .034, .061])*scale, box_yaw)
    elif kind == 'box_cylinder':
        first = box('target_A', middle, np.array([.067, .041, .057])*scale, box_yaw)
        second = cylinder('target_B', middle, .019*scale, .079*scale)
    elif kind == 'cylinder_cylinder':
        first = cylinder('target_A', middle, .023*scale, .067*scale)
        second = cylinder('target_B', middle, .017*scale, .091*scale)
    elif kind == 'sphere_sphere':
        radius = (.024 if holdout else .021)*scale
        first = sphere('target_A', middle, radius)
        second = sphere('target_B', middle, radius)
    else:
        raise ValueError(kind)
    distance = footprint_half_extent(first, direction)+footprint_half_extent(second, direction)+gap
    first = replace(first, center=first.center-np.r_[direction*distance/2, 0.])
    second = replace(second, center=second.center+np.r_[direction*distance/2, 0.])
    return first, second


def _blocker(target, eye, *, partial=False, single=False):
    direction = np.asarray(eye[:2])-target.center[:2]
    direction /= np.linalg.norm(direction)
    across = np.array([-direction[1], direction[0]])
    center = target.center[:2]+direction*(.105 if single else .085)
    if partial:
        center += .026*across
    size = [.024, .047 if partial else (.19 if single else .14), .21]
    return box('OccludingPanel', center, size, np.arctan2(direction[1], direction[0]))


def robustness_scenes(split='both'):
    selected = ('selection', 'holdout') if split == 'both' else (split,)
    scenes = []
    for partition in selected:
        rng = np.random.default_rng(SEEDS[partition])
        holdout = partition == 'holdout'
        for gap in (2., 4., 6., 10.):
            for kind in ('box_box', 'box_cylinder', 'cylinder_cylinder', 'sphere_sphere'):
                seed = int(rng.integers(1, 2**31))
                objects = close_pair(kind, gap/1000, np.random.default_rng(seed), holdout=holdout)
                # Hydra must not reuse one USD prim path for different schema
                # types across scenes (e.g. Cube then Cylinder then Sphere).
                objects = tuple(replace(obj, name=f'{partition}_{kind}_gap_{int(gap)}_{side}')
                                for obj, side in zip(objects, ('A', 'B')))
                scenes.append(RobustnessScene(f'gap_{int(gap)}_{kind}', partition, seed, objects, gap_mm=gap,
                    note='Exact horizontal support-plane gap; no touching/intersecting solids.'))
        seed = int(rng.integers(1, 2**31))
        objects = (
            box('isolated_box', (.35, .19), (.068, .037, .051), -.47 if holdout else .31),
            sphere('isolated_sphere', (.52, .18), .029 if holdout else .024),
            cylinder('isolated_upright', (.35, -.18), .023, .077),
            cylinder('isolated_lying', (.54, -.18), .019, .093, -.63 if holdout else .42),
        )
        scenes.append(RobustnessScene('isolated', partition, seed, objects))
        scenes.append(RobustnessScene('empty', partition, seed+1, ()))
        unknown = ResearchObject('unsupported_ellipsoid', 'ellipsoid', np.array([.46, .015, .026]),
                                 np.array([.089, .039, .052]))
        scenes.append(RobustnessScene('unknown', partition, seed+2, (unknown,),
                                      note='Anisotropic ellipsoid: the expected primitive label is unknown.'))
        for camera, eye in zip(('A', 'B'), EYES[partition]):
            for severity in ('partial', 'strong', 'one_view'):
                target = cylinder('occluded_target', (.44, .025), .023, .077)
                panel = _blocker(target, eye, partial=severity == 'partial', single=severity == 'one_view')
                scenes.append(RobustnessScene(f'{severity}_{camera}', partition, seed+len(scenes), (target,), (panel,),
                    note='Physical panel occlusion; measured oracle visible-point counts are reported per camera.'))
        link = cylinder('ElevatedLinkSurrogate', (.47, -.005), .014, .15, -.60)
        link = replace(link, center=np.array([.47, -.005, .095]))
        link_targets = (box('near_link_box', (.42, .08), (.059, .036, .047), .29),
                        sphere('near_link_sphere', (.52, -.08), .024))
        scenes.append(RobustnessScene('link_clutter', partition, seed+20, link_targets, (link,),
                                      note='Elevated link-shaped distractor; no claim of full robot-link coverage.'))
        temporal = (sphere('moving_sphere', (.42, -.075), .024),
                    box('moving_box', (.48, .09), (.055, .039, .048), .27))
        scenes.append(RobustnessScene('temporal', partition, seed+21, temporal, temporal=True,
                                      note='Confirm, move, one-frame absence, reacquire; no historical geometry scoring.'))
    return scenes


def scene_objects(scene, frame_index):
    if not scene.temporal or frame_index < 3:
        return scene.objects
    shift = np.array([.010+max(0, frame_index-3)*.003, -.006, 0.])
    objects = tuple(replace(obj, center=obj.center+shift) for obj in scene.objects)
    return objects[1:] if frame_index == 4 else objects


def object_payload(obj):
    return dict(name=obj.name, shape=obj.shape, center=obj.center.tolist(), size=obj.size.tolist(),
                yaw=float(obj.yaw), radius=float(obj.radius), length=float(obj.length), axis=list(obj.axis))


def object_from_payload(row):
    return ResearchObject(**{**row, 'center': np.asarray(row['center']), 'size': np.asarray(row['size'])})


def truth_label(obj):
    return 'unknown' if obj.shape == 'ellipsoid' else obj.shape


def surface_distances(points, obj):
    p = np.asarray(points, float)-obj.center
    if obj.shape == 'sphere':
        return abs(np.linalg.norm(p, axis=1)-obj.radius)
    if obj.shape == 'ellipsoid':
        # Radial distance along the center-to-point ray; scoring approximation
        # only, never used to generate a fitting/segmentation input.
        lengths = np.linalg.norm(p, axis=1)
        scale = np.linalg.norm(p/(obj.size/2), axis=1)
        distance = abs(lengths-lengths/np.maximum(scale, 1e-12))
        return np.where(lengths < 1e-12, float(np.min(obj.size)/2), distance)
    if obj.shape == 'cube':
        c, s = np.cos(obj.yaw), np.sin(obj.yaw)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        q = abs(p @ rotation)-obj.size/2
    else:
        axis = np.asarray(obj.axis)
        along = p @ axis
        q = np.column_stack((np.linalg.norm(p-along[:, None]*axis, axis=1)-obj.radius,
                             abs(along)-obj.length/2))
    return abs(np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(1), 0))


def surface_memberships(points, objects, tolerance=.0015):
    """Exclusive nearest-surface assignments avoid counting 2-mm gaps twice."""
    points = np.asarray(points).reshape(-1, 3)
    labels = np.full(len(points), -1, int)
    if len(points) and objects:
        distances = np.column_stack([surface_distances(points, obj) for obj in objects])
        nearest = distances.argmin(1)
        valid = distances[np.arange(len(points)), nearest] <= tolerance
        labels[valid] = nearest[valid]
    return labels


def score_observations(objects, observations, clouds, *, method, trial, frame_index, previous_ids=None):
    pairs = sorted((float(np.linalg.norm(obs.pose.position-obj.center)), oi, di)
                   for oi, obj in enumerate(objects) for di, obs in enumerate(observations))
    matched, assigned = {}, set()
    for distance, oi, di in pairs:
        if distance < .04 and oi not in matched and di not in assigned:
            matched[oi] = di
            assigned.add(di)
    rows, supports, mixed, checked = [], np.zeros(len(objects), int), 0, 0
    cloud_details = []
    for obs, cloud in zip(observations, clouds):
        fractions = np.zeros(len(objects))
        if cloud is not None and len(cloud):
            checked += 1
            labels = surface_memberships(cloud, objects)
            counts = np.array([np.count_nonzero(labels == oi) for oi in range(len(objects))])
            fractions = counts/len(cloud)
            significant = (fractions >= .10) & (counts >= 8)
            supports += significant
            mixed += int(np.count_nonzero(significant) > 1)
        cloud_details.append(dict(object_id=obs.object_id, truth_surface_fractions=fractions.tolist(),
                                  points=0 if cloud is None else len(cloud)))
    id_switches, current_ids = 0, {}
    for oi, obj in enumerate(objects):
        extra = dict(method=method, trial=trial, frame_index=frame_index)
        if oi in matched:
            obs = observations[matched[oi]]
            fit = SimpleNamespace(geometry=obs.geometry, center=obs.pose.position, size=obs.size)
            if obj.shape == 'ellipsoid':
                geometry = obs.geometry
                row = dict(object=obj.name, truth='unknown', predicted=geometry.shape.value,
                    correct=geometry.shape.value == 'unknown', unknown=geometry.shape.value == 'unknown',
                    detected=True, confidence=float(geometry.confidence), scores=geometry.scores,
                    residuals_m=geometry.metadata.get('residuals_m', {}),
                    center_error_mm=float(np.linalg.norm(obs.pose.position-obj.center)*1000),
                    dimension_error_mm=None, axis_error_deg=None, estimated_center=obs.pose.position.tolist(),
                    estimated_size=obs.size.tolist(), truth_center=obj.center.tolist(),
                    estimated_dimensions=None, truth_dimensions=obj.size.tolist(), **extra)
            else:
                row = metric_row(obj, fit, detected=True, **extra)
            scores = sorted((float(value) for value in obs.geometry.scores.values()), reverse=True)
            row['score_margin'] = None if len(scores) < 2 else scores[0]-scores[1]
            row['radius_error_mm'] = (float(abs(obs.geometry.radius-obj.radius)*1000)
                if obj.shape in ('sphere', 'cylinder') and obs.geometry.radius is not None else None)
            row['length_error_mm'] = (float(abs(obs.geometry.length-obj.length)*1000)
                if obj.shape == 'cylinder' and obs.geometry.length is not None else None)
            row['box_yaw_error_deg'] = row.get('axis_error_deg') if obj.shape == 'cube' else None
            row['cylinder_axis_error_deg'] = row.get('axis_error_deg') if obj.shape == 'cylinder' else None
            row['object_id'] = obs.object_id
            row['contributing_views'] = obs.metadata.get('contributing_views')
            current_ids[obj.name] = obs.object_id
            if previous_ids and obj.name in previous_ids and previous_ids[obj.name] != obs.object_id:
                id_switches += 1
        else:
            row = dict(object=obj.name, truth=truth_label(obj), predicted='missing', correct=False,
                detected=False, unknown=False, center_error_mm=None, dimension_error_mm=None,
                axis_error_deg=None, radius_error_mm=None, length_error_mm=None, box_yaw_error_deg=None,
                cylinder_axis_error_deg=None, confidence=None, score_margin=None, object_id=None, **extra)
        rows.append(row)
    counts = dict(method=method, trial=trial, frame_index=frame_index, expected=len(objects),
        observed=len(observations), matched=len(matched), missing=len(objects)-len(matched),
        false_positives=len(observations)-len(assigned), mixed_accepted_clouds=mixed, mixed_checked=checked,
        false_splits=int(np.maximum(supports-1, 0).sum()), identity_switches=id_switches,
        duplicate_ids=len(observations)-len({obs.object_id for obs in observations}), clouds=cloud_details)
    return rows, counts, current_ids


class SnapshotCamera:
    def __init__(self, frame):
        self.push(frame)

    def push(self, frame): self.frame = frame
    def get_frame_token(self): return self.frame.token
    def get_depth(self): return self.frame.depth
    def get_rgb(self): return self.frame.rgb
    def get_intrinsics(self): return self.frame.intrinsics
    def get_world_from_camera_transform(self): return self.frame.world_from_camera
    def close(self): pass


class SnapshotRig:
    """One consumable snapshot per push; capture never retimestamps a replay."""
    def __init__(self):
        self.frames, self.consumed = None, True
        self.status = {'state': 'waiting'}

    def push(self, frames):
        self.frames, self.consumed = tuple(frames), False
        self.status = {'state': 'saved_native_snapshot'}

    def capture(self):
        if self.consumed:
            self.status = {'state': 'unchanged'}
            return None
        self.consumed = True
        return self.frames

    def close(self): pass


def perturb_frames(frames, variant, *, seed):
    """Deterministic measurement perturbations; original tokens/times survive."""
    result = []
    rng = np.random.default_rng(seed)
    for index, frame in enumerate(frames):
        depth, transform = frame.depth.copy(), frame.world_from_camera.copy()
        if variant == 'depth_noise':
            valid = np.isfinite(depth) & (depth > 0)
            depth[valid] += rng.normal(0., .0005, np.count_nonzero(valid))
        elif variant == 'calibration' and index == 1:
            angle = np.deg2rad(.05)
            c, s = np.cos(angle), np.sin(angle)
            transform[:3, :3] = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ transform[:3, :3]
            transform[:3, 3] += [.0005, -.0003, .0002]
        elif variant == 'sparse':
            depth[rng.random(depth.shape) < .35] = np.nan
        elif variant not in ('clean', 'calibration'):
            raise ValueError(variant)
        result.append(DepthFrame(frame.camera_id, frame.token, depth, frame.intrinsics.copy(),
                                 transform, frame.acquired_at, frame.rgb))
    return tuple(result)


def distribution(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return None if not values else dict(mean=float(np.mean(values)), p95=float(np.percentile(values, 95)),
                                        worst=float(np.max(values)), samples=len(values))
