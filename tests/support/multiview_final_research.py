"""Matched cube/sphere scenes, bounded camera layouts and truth-only audits.

These fixtures never enter production perception. Physical scenes do not depend
on camera choice. Native visibility is an outcome: a layout that reveals or
hides a supported object does not make an otherwise valid experiment invalid.
The holdout seed/design is declared here; generation requires an explicit
post-freeze authorization from the evaluation runner.
"""
from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception_multiview.clouds import project_depth, voxelize
from tests.support.multiview_iteration2 import box, footprint_half_extent, sphere, surface_memberships


SEEDS = {'coarse': 20261021, 'shortlist': 20261103, 'holdout': 20261219}
HOLDOUT_DRAW_SEEDS = (20261219, 20261220)
WORKSPACE_MIN = np.array([.20, -.40, 0.])
WORKSPACE_MAX = np.array([.70, .40, .22])
WORKSPACE_CENTER = (WORKSPACE_MIN+WORKSPACE_MAX)/2
WORKSPACE_AIM = (.45, 0., .04)
# Read from the authored/composed /Franka root, rather than a presumed world
# front convention: translate0, identity orient, identity world transform.
ROBOT_WORLD_FROM_BASE = np.eye(4)
VISIBILITY = dict(minimum_fused_source_points=30, minimum_surface_fraction=.18,
                  minimum_camera_surface_points=8, surface_tolerance_m=.0015,
                  table_clearance_m=.004, box_grid_side=12, sphere_grid_points=1024)

# Authored USD world matrices were inspected without starting Kit. Convert USD
# local +X/right,+Y/up,-Z/forward to ROS right/down/forward. These exact matrices
# retain original camera roll/pitch, unlike benchmark look-at overrides.
REFERENCE_WORLD_FROM_CAMERAS = (
    ((.015999969482422105, .6062406100373334, -.7951203202774557, 1.6589600172814285),
     (.9998406222443645, -.015999969482422105, .007920295863393323, -.05281348017761145),
     (-.007920295863393323, -.7951203202774557, -.6063999877929689, 1.1035808604242954),
     (0., 0., 0., 1.)),
    ((.9734171683, -.1311651672, .1877623906, .25),
     (-.2290393337, -.5574519604, .79799016, -.85),
     (0., -.8197822947, -.5726752913, .65), (0., 0., 0., 1.)),
)


@dataclass(frozen=True)
class CameraLayout:
    name: str
    eyes: tuple
    targets: tuple
    note: str = ''
    world_from_cameras: tuple | None = None


@dataclass(frozen=True)
class FinalScene:
    name: str
    split: str
    seed: int
    objects: tuple
    requirements: dict | None = None
    note: str = ''
    temporal: bool = False

    @property
    def key(self):
        return f'{self.split}_{self.name}'


def workspace_axes(world_from_base=ROBOT_WORLD_FROM_BASE):
    """Forward from the measured base toward the horizontal workspace center."""
    transform = np.asarray(world_from_base, float)
    origin, up = transform[:3, 3], np.array([0., 0., 1.])
    forward = WORKSPACE_CENTER-origin
    forward -= up*(forward @ up)
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError('Workspace center cannot coincide with robot base')
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    return origin.copy(), forward, left, up


def camera_layouts():
    """Ten predeclared external arrangements in robot/workspace-derived axes."""
    origin, forward, left, up = workspace_axes()
    world = lambda x, y, z: tuple(origin+x*forward+y*left+z*up)
    reference_eyes = tuple(tuple(np.asarray(t)[:3, 3]) for t in REFERENCE_WORLD_FROM_CAMERAS)
    reference_targets = tuple(tuple(np.asarray(t)[:3, 3]+np.asarray(t)[:3, 2])
                              for t in REFERENCE_WORLD_FROM_CAMERAS)
    layouts = [CameraLayout('original_reference', reference_eyes, reference_targets,
        'Exact composed historical USD poses; A approximately1.65896,-.05281,1.10358m; B.25,-.85,.65m.',
        REFERENCE_WORLD_FROM_CAMERAS)]
    for name, values, note in (
        ('front_symmetric', ((1.05, .70, .85), (1.05, -.70, .85)), 'Preferred front-left/front-right hypothesis.'),
        ('front_higher', ((1.05, .70, 1.15), (1.05, -.70, 1.15)), 'Higher symmetric oblique views.'),
        ('front_lower', ((1.05, .70, .55), (1.05, -.70, .55)), 'Lower oblique views emphasize side surfaces.'),
        ('front_wider', ((.95, .95, .80), (.95, -.95, .80)), 'Wider lateral baseline.'),
        ('front_narrower', ((1.15, .45, .80), (1.15, -.45, .80)), 'Narrower front corner baseline.'),
        ('asymmetric_high_left', ((1.10, .65, 1.05), (.90, -.80, .65)), 'High left with lower lateral right.'),
        ('asymmetric_high_right', ((.90, .80, .65), (1.10, -.65, 1.05)), 'Mirrored height asymmetry.'),
    ):
        layouts.append(CameraLayout(name, tuple(world(*eye) for eye in values), (WORKSPACE_AIM,)*2, note))
    for name, eye, note in (
        ('original_A_front_left_B', (.95, .80, .80), 'Preserve original A; move only B to front-left.'),
        ('original_A_side_left_B', (.18, .80, .65), 'Preserve original A; complementary left lateral B.'),
    ):
        layouts.append(CameraLayout(name, (reference_eyes[0], world(*eye)),
            (reference_targets[0], WORKSPACE_AIM), note, (REFERENCE_WORLD_FROM_CAMERAS[0], None)))
    return tuple(layouts)


def layout_payload(layout):
    return dict(name=layout.name, eyes=layout.eyes, targets=layout.targets,
                world_from_cameras=layout.world_from_cameras, note=layout.note)


def layout_digest(layout):
    return hashlib.sha256(json.dumps(layout_payload(layout), sort_keys=True).encode()).hexdigest()


def balanced_layout_order(layouts, scene_index, round_index=0):
    """Rotate first layout and alternate direction; never change physical scenes."""
    layouts = tuple(layouts)
    if not layouts:
        return ()
    offset = (scene_index+round_index) % len(layouts)
    ordered = layouts[offset:]+layouts[:offset]
    return ordered if round_index % 2 == 0 else ordered[::-1]


def _rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _pair(kind, middle, gap, rng, prefix):
    yaw = float(rng.uniform(-.65, .65))
    direction = np.array([np.cos(yaw), np.sin(yaw)])
    scale = float(rng.uniform(.88, 1.08))
    if kind == 'cube_cube':
        first = box(prefix+'_A', middle, np.array([.043, .050, .043])*scale, yaw)
        second = box(prefix+'_B', middle, np.array([.054, .035, .054])*scale, yaw)
    elif kind == 'sphere_sphere':
        first = sphere(prefix+'_A', middle, .021*scale)
        second = sphere(prefix+'_B', middle, .025*scale)
    elif kind == 'cube_sphere':
        first = box(prefix+'_A', middle, np.array([.052, .039, .053])*scale, yaw)
        second = sphere(prefix+'_B', middle, .022*scale)
    else:
        raise ValueError(kind)
    distance = sum(footprint_half_extent(obj, direction) for obj in (first, second))+gap
    if kind == 'sphere_sphere':
        distance = np.sqrt(distance**2-(second.center[2]-first.center[2])**2)
    return tuple(replace(obj, center=obj.center+np.r_[sign*direction*distance/2, 0.])
                 for obj, sign in ((first, -1), (second, 1)))


def final_scenes(split='coarse', *, holdout_authorized=False):
    """Matched scenes with no camera input; the coarse draw remains13/31.

    Shortlist adds supported one-view/full hiding and a second temporal cover:
    fifteen scenes/thirty-seven objects. Final holdout uses two independent
    draws under20261219/20261220, but requires source AND layout freeze first.
    """
    if split not in SEEDS:
        raise ValueError('Choose coarse,shortlist or frozen holdout')
    if split == 'holdout' and not holdout_authorized:
        raise ValueError('Freeze source and camera layout before generating holdout')
    if split == 'holdout':
        return tuple(scene for draw, seed in enumerate(HOLDOUT_DRAW_SEEDS)
                     for scene in _scene_draw(split, seed, f'draw{draw}_'))
    return _scene_draw(split, SEEDS[split])


def _scene_draw(split, seed, prefix=''):
    rng = np.random.default_rng(seed)
    scenes = []
    def add(name, objects, **extra):
        name = prefix+name
        # Unique per-scene names avoid Hydra schema reuse between shapes.
        objects = tuple(replace(obj, name=f'final_{split}_{name}_{index}') for index, obj in enumerate(objects))
        scenes.append(FinalScene(name, split, int(rng.integers(1, 2**31)), objects, **extra))
    xy = lambda x, y, spread=.02: np.array([x, y])+rng.uniform(-spread, spread, 2)
    scale = lambda: float(rng.uniform(.90, 1.08))
    add('single_square', (box('cube', xy(.43, -.08), np.full(3, .045*scale()), float(rng.uniform(-.7, .7))),))
    add('single_rectangle', (box('box', xy(.46, .08), np.array([.078, .038, .052])*scale(), float(rng.uniform(-.9, .9))),))
    add('near_square_rectangle', (box('box', xy(.445, .02), np.array([.050, .0495, .043])*scale(),
                                    float(rng.uniform(-.9, .9))),),
        requirements=dict(yaw_truth_scope='Near-square1.01aspect; precise long-axis heading is geometrically weak.'))
    add('single_small_sphere', (sphere('sphere', xy(.42, .07), .018*scale()),))
    add('single_large_sphere', (sphere('sphere', xy(.48, -.08), .031*scale()),))
    add('separated', (box('box', xy(.33, -.18), np.array([.058, .037, .046])*scale(), float(rng.uniform(-.7, .7))),
        box('cube', xy(.56, .18), np.full(3, .055*scale()), float(rng.uniform(-.7, .7))),
        sphere('sphere', xy(.33, .18), .023*scale()), sphere('sphere', xy(.56, -.18), .027*scale())))
    add('workspace_edges', (box('box', xy(.24, -.30, .003), np.array([.043, .034, .044])*scale(), float(rng.uniform(.2, .55))),
        box('box', xy(.66, .30, .003), np.array([.047, .033, .052])*scale(), float(rng.uniform(-.6, -.2))),
        sphere('sphere', xy(.24, .30, .003), .023*scale()),
        sphere('sphere', xy(.66, -.30, .003), .024*scale())),
        note='Objects remain physically inside the configured ROI; camera FOV/robot occlusion is a scored outcome.')
    for kind in ('cube_cube', 'sphere_sphere', 'cube_sphere'):
        add(kind+'_gap4', _pair(kind, xy(.45, .01), .004, rng, kind),
            requirements=dict(gap_mm=4., pair_kind=kind), note='Exact nearest support-face gap; sphere pair uses full3D center spacing.')
    pairs = tuple(obj for kind, middle in zip(('cube_cube', 'sphere_sphere', 'cube_sphere'),
                    ((.32, -.22), (.55, 0.), (.35, .22))) for obj in _pair(kind, xy(*middle, .006), .008, rng, kind))
    add('three_pairs_gap8', pairs, requirements=dict(gap_mm=8., pair_kinds=['cube_cube', 'sphere_sphere', 'cube_sphere']))
    offset = rng.uniform(-.012, .012, 2)
    occlusion_scale = float(rng.uniform(.95, 1.04))
    add('supported_occlusion', (box('cover', np.array([.515, 0.])+offset, np.array([.095, .135, .140])*occlusion_scale),
        box('lowbox', np.array([.40, .075])+offset, np.array([.041, .031, .034])*scale(), float(rng.uniform(-.2, .2))),
        sphere('lowsphere', np.array([.40, -.025])+offset, .0185*scale())),
        requirements=dict(visibility_outcome='Record achieved A/B/both hiding; do not invalidate a layout that defeats the occlusion.'),
        note='Only supported objects; tall box and low targets all remain scored objects.')
    offset = rng.uniform(-.012, .012, 2)
    motion = [float(rng.uniform(.008, .014)), float(rng.uniform(.032, .048)), 0.]
    add('temporal', (sphere('moving', np.array([.39, -.04])+offset, .0195*scale()),
        box('stationary', np.array([.35, .20])+offset, np.array([.047, .036, .043])*scale(), float(rng.uniform(-.6, .6))),
        box('cover', np.array([.58, .20])+offset, np.array([.095, .135, .135])*float(rng.uniform(.96, 1.04)))), temporal=True,
        requirements=dict(identity='Same truth names through movement,covered,revealed phases; actual visibility determines recovery scope.',
                          movement_delta_m=motion),
        note='Supported box changes rendered pose; no robot or gripper commands.')
    if split != 'coarse':
        # Keep the coarse RNG sequence and legacy three-object temporal fixture
        # unchanged. Later splits add camera-independent supported covers only.
        previous = scenes[-1]
        group, cover_scale = _supported_cover_group(rng, both=True)
        target, cover_a, cover_b = group
        covers = (cover_a, cover_b)
        destinations = [cover.center+np.asarray(motion) for cover in covers]
        starts = (target.center[:2]+np.array([.19, .285])*cover_scale,
                  target.center[:2]+np.array([.19, -.285])*cover_scale)
        objects = (target, previous.objects[1])+tuple(
            replace(cover, center=np.r_[xy, cover.center[2]]) for cover, xy in zip(covers, starts))
        objects = tuple(replace(obj, name=f'final_{split}_{previous.name}_{index}')
                        for index, obj in enumerate(objects))
        scenes[-1] = replace(previous, objects=objects,
            requirements={**previous.requirements, 'fixture_version': 2,
                'cover_destinations_m': [dict(object_index=index+2, center_m=center.tolist())
                                         for index, center in enumerate(destinations)],
                'visibility_outcome': 'Record actual clear,one-view,full hiding and return; all supported covers remain scored.'},
            note='Two supported boxes approach and retreat over three fresh-frame steps; no robot commands.')
        for name, both in (('supported_one_view', False), ('supported_full_hide', True)):
            objects, _ = _supported_cover_group(rng, both=both)
            add(name, objects, requirements=dict(fixture_version=2,
                intended_visibility='full_hiding' if both else 'positive_Y_view_hidden_negative_Y_clear',
                visibility_outcome='Audit actual native depths; layout-dependent hiding is an outcome, not fixture invalidity.'),
                note='Same physical scene across layouts; only table-supported scored cube/sphere objects.')
    return tuple(scenes)


def _supported_cover_group(rng, *, both):
    """Low sphere and supported covers, independent of any camera geometry."""
    offset = rng.uniform(-.008, .008, 2)
    scale = float(rng.uniform(.97, 1.03))
    target_xy = np.array([.39, 0.])+offset
    objects = (sphere('target', target_xy, .0195*scale),
        box('cover_a', target_xy+np.array([.085, .08])*scale, np.array([.06, .11, .16])*scale))
    if both:
        objects += (box('cover_b', target_xy+np.array([.085, -.09])*scale,
                        np.array([.06, .11, .16])*scale),)
    return objects, scale


def scene_frame_count(scene):
    return 12 if scene.temporal else 4


def scene_frame_content(scene, index):
    """Same supported objects/identities in every phase, including hidden ones."""
    if not 0 <= index < scene_frame_count(scene):
        raise ValueError('Frame index outside fixture sequence')
    if not scene.temporal:
        return scene.objects, 'static'
    objects = list(scene.objects)
    movement = np.asarray(scene.requirements['movement_delta_m'])
    if index < 3:
        phase = 'clear'
        cover_fraction = 0.
    elif index < 6:
        fraction = (index-2)/3
        objects[0] = replace(objects[0], center=objects[0].center+movement*fraction)
        phase = 'moving'
        cover_fraction = fraction
    elif index < 9:
        objects[0] = replace(objects[0], center=objects[0].center+movement)
        phase = 'covered'
        cover_fraction = 1.
    else:
        objects[0] = replace(objects[0], center=objects[0].center+movement+np.array([.01, .01, 0.]))
        phase = 'revealed'
        cover_fraction = (11-index)/3
    if scene.requirements.get('fixture_version') == 2:
        for specification in scene.requirements['cover_destinations_m']:
            slot = specification['object_index']
            destination = np.asarray(specification['center_m'])
            objects[slot] = replace(objects[slot],
                center=objects[slot].center*(1-cover_fraction)+destination*cover_fraction)
    else:
        # Manifest-reconstructed coarse/older three-object fixtures retain their
        # exact historical path; no new cover is synthesized during replay.
        destination = np.r_[scene.objects[0].center[:2]+movement[:2]+[.105, 0.], objects[2].center[2]]
        objects[2] = replace(objects[2], center=objects[2].center*(1-cover_fraction)+destination*cover_fraction)
    return tuple(objects), phase


def object_aabb(obj):
    if obj.shape == 'sphere':
        extent = np.full(3, obj.radius)
    else:
        extent = abs(_rotation(obj.yaw)) @ obj.size/2
    return obj.center-extent, obj.center+extent


def _overlap(first, second):
    if first.shape == second.shape == 'sphere':
        return np.linalg.norm(first.center-second.center) < first.radius+second.radius-1e-7
    if first.shape == 'sphere' or second.shape == 'sphere':
        ball, cube = (first, second) if first.shape == 'sphere' else (second, first)
        q = (ball.center-cube.center) @ _rotation(cube.yaw)
        return np.linalg.norm(np.maximum(abs(q)-cube.size/2, 0)) < ball.radius-1e-7
    # Ground-supported yaw boxes share a vertical interval. Separating axes
    # distinguish rotated physical footprints even when their world AABBs overlap.
    for direction in (*_rotation(first.yaw)[:2, :2].T, *_rotation(second.yaw)[:2, :2].T):
        distance = abs((first.center[:2]-second.center[:2]) @ direction)
        radii = sum(footprint_half_extent(obj, direction) for obj in (first, second))
        if distance >= radii-1e-7:
            return False
    return True


def fixture_geometry_errors(objects):
    errors = []
    for obj in objects:
        lo, hi = object_aabb(obj)
        if np.any(lo < WORKSPACE_MIN-1e-7) or np.any(hi > WORKSPACE_MAX+1e-7):
            errors.append(f'{obj.name}: outside workspace AABB')
        if abs(lo[2]) > 1e-7 or obj.shape not in ('cube', 'sphere'):
            errors.append(f'{obj.name}: not a supported cube/sphere')
    for i, first in enumerate(objects):
        for second in objects[i+1:]:
            if _overlap(first, second):
                errors.append(f'{first.name}/{second.name}: intersecting physical solids')
    return errors


def _surface_quadrature(obj, support_plane_z):
    """Area-weighted physical surface above the same4mm contact crop as source."""
    points, normals, weights, faces = [], [], [], []
    if obj.shape == 'sphere':
        count = VISIBILITY['sphere_grid_points']
        z = 1-2*(np.arange(count)+.5)/count
        theta = np.arange(count)*np.pi*(3-np.sqrt(5))
        unit = np.c_[np.sqrt(1-z*z)*np.cos(theta), np.sqrt(1-z*z)*np.sin(theta), z]
        points, normals = obj.center+obj.radius*unit, unit
        weights = np.full(count, 4*np.pi*obj.radius**2/count)
        faces = np.full(count, 'sphere')
    else:
        count = VISIBILITY['box_grid_side']
        grid = (np.arange(count)+.5)/count-.5
        for axis, sign, face in ((0, -1, '-x'), (0, 1, '+x'), (1, -1, '-y'), (1, 1, '+y'), (2, 1, '+z')):
            other = [a for a in range(3) if a != axis]
            local = np.zeros((count*count, 3))
            local[:, axis] = sign*obj.size[axis]/2
            local[:, other] = np.array([(a, b) for a in grid for b in grid])*obj.size[other]
            normal = np.zeros((len(local), 3)); normal[:, axis] = sign
            points.extend(obj.center+local @ _rotation(obj.yaw).T)
            normals.extend(normal @ _rotation(obj.yaw).T)
            weights.extend(np.full(len(local), np.prod(obj.size[other])/len(local)))
            faces.extend([face]*len(local))
        points, normals, weights, faces = map(np.asarray, (points, normals, weights, faces))
    keep = points[:, 2] > support_plane_z+VISIBILITY['table_clearance_m']
    return points[keep], normals[keep], weights[keep], faces[keep]


def audit_final_scene(scene, frames, *, index=3, config=None):
    """Native measurement audit; no algorithm/classification acceptance gate.

    Surface fraction is the weighted fraction of deterministic analytic-area
    samples that are front-facing,in-image,and project to a pixel whose current
    depth is exclusively associated with that physical object's surface within
    1.5mm. It is a sampled visibility estimate, not a point-count percentage or
    proof that every dimension/yaw is geometrically observable.
    """
    objects, phase = scene_frame_content(scene, index)
    cfg = config or load_scene_config('config/scenes/multiview_playground.toml').scene
    errors = fixture_geometry_errors(objects)
    if len(frames) != 2:
        errors.append('need_two_synchronized_views')
    views, clouds = [], []
    for ci, frame in enumerate(frames):
        if not np.any(np.isfinite(frame.depth) & (frame.depth > 0)):
            errors.append(f'camera_{ci}: no valid depth')
        points, pixels = project_depth(frame, stride=1, bounds=(cfg.workspace_min, cfg.workspace_max))
        keep = ((points >= cfg.workspace_min).all(1) & (points <= cfg.workspace_max).all(1) &
                (points[:, 2] > cfg.support_plane_z+VISIBILITY['table_clearance_m']))
        points, pixels = points[keep], pixels[keep]
        labels = surface_memberships(points, objects, tolerance=VISIBILITY['surface_tolerance_m'])
        label_image = np.full(frame.depth.shape, -1, dtype=np.int16)
        label_image[pixels[:, 0], pixels[:, 1]] = labels
        stride = (pixels[:, 0] % 2 == 0) & (pixels[:, 1] % 2 == 0)
        cloud = voxelize(points[stride], .002, 12000)
        source_labels = surface_memberships(cloud, objects, tolerance=VISIBILITY['surface_tolerance_m'])
        views.append((frame, labels, source_labels, label_image))
        clouds.append(cloud)
    fused = voxelize(np.concatenate(clouds) if clouds else np.empty((0, 3)), .002, 12000)
    fused_labels = surface_memberships(fused, objects, tolerance=VISIBILITY['surface_tolerance_m'])
    rows = []
    for oi, obj in enumerate(objects):
        surface, normals, weights, face_names = _surface_quadrature(obj, cfg.support_plane_z)
        masks, camera_rows = [], []
        for ci, (frame, labels, source_labels, label_image) in enumerate(views):
            camera = (surface-frame.world_from_camera[:3, 3]) @ frame.world_from_camera[:3, :3]
            front = np.einsum('ij,ij->i', normals, frame.world_from_camera[:3, 3]-surface) > 0
            with np.errstate(divide='ignore', invalid='ignore'):
                uv = camera[:, :2]/camera[:, 2, None]*[frame.intrinsics[0, 0], frame.intrinsics[1, 1]]+frame.intrinsics[:2, 2]-.5
            finite = np.isfinite(uv).all(1) & (camera[:, 2] > 0)
            pixels = np.rint(np.where(np.isfinite(uv), uv, -1e6)).astype(int)
            h, w = frame.depth.shape
            inside = finite & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
            visible = np.zeros(len(surface), bool)
            ids = np.flatnonzero(inside & front)
            visible[ids] = label_image[pixels[ids, 1], pixels[ids, 0]] == oi
            masks.append(visible)
            camera_rows.append(dict(camera_index=ci, raw_points=int(np.count_nonzero(labels == oi)),
                source_points=int(np.count_nonzero(source_labels == oi)),
                visible_surface_fraction=float(weights[visible].sum()/weights.sum()),
                front_facing_surface_fraction=float(weights[front].sum()/weights.sum()),
                in_image_surface_fraction=float(weights[inside].sum()/weights.sum())))
        union = np.logical_or.reduce(masks) if masks else np.zeros(len(surface), bool)
        fraction = float(weights[union].sum()/weights.sum())
        fused_count = int(np.count_nonzero(fused_labels == oi))
        seen = [row['raw_points'] >= VISIBILITY['minimum_camera_surface_points'] for row in camera_rows]
        faces = {str(face): dict(area_m2=float(weights[face_names == face].sum()),
            union_visible_fraction=float(weights[(face_names == face) & union].sum()/weights[face_names == face].sum()),
            per_camera_visible_fraction=[float(weights[(face_names == face) & mask].sum()/weights[face_names == face].sum()) for mask in masks])
            for face in np.unique(face_names)}
        rows.append(dict(object=obj.name, shape=obj.shape, cameras=camera_rows, faces=faces,
            union_visible_surface_fraction=fraction, sampled_surface_area_m2=float(weights.sum()),
            surface_quadrature_samples=len(surface), fused_source_points=fused_count,
            sufficiently_visible=bool(fused_count >= VISIBILITY['minimum_fused_source_points'] and
                                      fraction >= VISIBILITY['minimum_surface_fraction']),
            visible_in_only_one_camera=sum(seen) == 1, achieved_visible_cameras=[i for i, value in enumerate(seen) if value],
            fully_hidden=sum(row['raw_points'] for row in camera_rows) == 0))
    return dict(valid=not errors, errors=errors, frame_index=index, phase=phase, objects=rows,
        thresholds=dict(VISIBILITY), surface_fraction_definition='Area-weighted analytic physical surface above4mm table crop; current exclusive-depth pixel support, not a point-count ratio. No sphere orientation.',
        visibility_flag_scope='Evaluation denominator only; sufficient visibility does not establish complete dimensions or grasp eligibility.')
