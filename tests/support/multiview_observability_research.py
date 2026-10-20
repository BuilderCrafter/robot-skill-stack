"""Fixed visibility fixtures and post-capture truth audits, never runtime inputs.

The holdout generator is invoked explicitly only after the production freeze.
Ray/depth validity and fixture visibility are evaluated separately from detector
or planner outcomes. Invalid scenes remain in saved manifests and result rows.
"""
from dataclasses import dataclass, replace

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception_multiview.clouds import fuse_clouds, project_depth, voxelize
from tests.support.multiview_iteration2 import _blocker, box, cylinder, sphere, surface_memberships
from tests.support.multiview_readiness_research import EYES as PREVIOUS_EYES
from tests.support.multiview_research import ResearchObject


SEEDS = {'selection': 11235813, 'holdout': 223606797}
# Predeclared independent local viewing angles; fixed before holdout capture.
# The mandatory legacy regression intentionally retains its original exact rig.
RIG_EYES = {'selection': PREVIOUS_EYES['selection'],
            'holdout': ((1.54, -.12, 1.08), (.16, -.70, .70))}
CASES = ('clear_both', 'A_clear', 'B_clear', 'same_extent_hidden_both',
         'opposite_extents', 'partial_top', 'missing_sides', 'edge_adjacent_occluder',
         'near_image_boundary', 'alternative_views')
BASE_CENTER = np.array([.445, -.01, .025])


@dataclass(frozen=True)
class ObservabilityScene:
    name: str
    split: str
    seed: int
    objects: tuple
    clutter: tuple = ()
    eyes: tuple = PREVIOUS_EYES['selection']
    targets: tuple = ((.45, 0., .04), (.45, 0., .04))
    requirements: dict | None = None
    note: str = ''
    temporal: bool = False
    occluded_clutter: tuple = ()
    gap_mm: float | None = None

    @property
    def key(self):
        return f'{self.split}_{self.name}'


def _rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _canonical_blockers():
    target = box('canonical', BASE_CENTER[:2], (.09, .09, .05))
    return dict(
        A=(_blocker(target, PREVIOUS_EYES['selection'][0], partial=True),),
        B=(ResearchObject('roof_B', 'cube', np.array([.345, -.01, .085]), np.array([.15, .30, .014])),),
        both=(ResearchObject('roof_both', 'cube', np.array([.445, .04, .085]), np.array([.30, .17, .014])),
              ResearchObject('curtain_B', 'cube', np.array([.37, -.015, .105]), np.array([.012, .19, .21]))),
        top=(ResearchObject('top_patch', 'cube', np.array([.445, -.01, .085]), np.array([.045, .045, .014])),),
        adjacent=(box('adjacent', (.445, .065), (.14, .012, .13)),),
    )


def _transform_point(point, center, scale, rotation):
    return np.asarray(center)+(np.asarray(point)-BASE_CENTER) @ rotation.T*scale


def _transform_obj(obj, name, center, scale, yaw):
    return replace(obj, name=name, center=_transform_point(obj.center, center, scale, _rotation(yaw)),
                   size=obj.size*scale, radius=obj.radius*scale, length=obj.length*scale,
                   yaw=obj.yaw+yaw, axis=tuple(np.asarray(obj.axis) @ _rotation(yaw).T))


def _look_at(eye, target):
    forward = np.asarray(target)-np.asarray(eye)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack((right, down, forward))


def _near_boundary_target(eye, target, obj):
    """Aim using fixture corners/known native lens; actual pixels still audited."""
    rotation = _look_at(eye, target)
    signs = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = obj.center+(signs*obj.size/2) @ _rotation(obj.yaw).T
    lo, hi = 0., .6
    distance = np.linalg.norm(np.asarray(target)-eye)
    for _ in range(28):
        angle = (lo+hi)/2
        direction = rotation[:, 2]*np.cos(angle)+rotation[:, 0]*np.sin(angle)
        aim = np.asarray(eye)+direction*distance
        camera = (corners-eye) @ _look_at(eye, aim)
        u = camera[:, 0]/camera[:, 2]*554.25622559+319.5
        if u.min() > 10:
            lo = angle
        else:
            hi = angle
    return tuple(aim)


def _scene(name, split, seed, canonical_target, case, *, yaw=0., center=None):
    center = np.array([.445, -.01, canonical_target.size[2]/2]) if center is None else np.asarray(center)
    scale = canonical_target.size[1]/.09
    # Uniform camera/cover scaling preserves ray geometry without shearing boxes.
    target = replace(canonical_target, name=f'observability_{split}_{name}_target', center=center, yaw=yaw)
    transform = lambda point: tuple(_transform_point(point, center, scale, _rotation(yaw)))
    eyes = tuple(transform(eye) for eye in RIG_EYES[split])
    targets = tuple(transform((.45, 0., .04)) for _ in range(2))
    blockers = _canonical_blockers()
    if split == 'holdout' and case in ('B_clear', 'opposite_extents'):
        # Initial held-out camera angles let A see the fullY extent around the
        #47mm cover. A fixed67mm cover restores the predeclared partial-view
        #condition; targets/seeds/poses/cameras and valid scenes stay unchanged.
        # Original native captures and their failed visibility audit are kept.
        panel_size = blockers['A'][0].size.copy()
        panel_size[1] = .067
        blockers['A'] = (replace(blockers['A'][0], size=panel_size),)
    extra_x = (target.size[0]/scale-.09)/2
    # Keep the plate/curtain shadow relative to the target's farX side as the
    # horizontal aspect ratio varies. All boxes remain rigid (no affine shear).
    edge = .420-extra_x
    blockers['B'] = (replace(blockers['B'][0], center=np.array([(.27+edge)/2, -.01, .085]),
                             size=np.array([edge-.27, .30, .014])),)
    blockers['both'] = (blockers['both'][0], replace(blockers['both'][1],
        center=blockers['both'][1].center-np.array([extra_x, 0., 0.])))
    requirement = dict(minimum_target_points_per_view=30, full_views=[0, 1],
                       fused_full=True, case=case)
    selected = ()
    if case == 'A_clear':
        selected = blockers['B']
        requirement.update(full_views=[0], clipped=[[1, 0]])
    elif case == 'B_clear':
        selected = blockers['A']
        requirement.update(full_views=[1], clipped=[[0, 1]])
    elif case == 'same_extent_hidden_both':
        selected = blockers['both']
        requirement.update(full_views=[], clipped=[[0, 1], [1, 1]], fused_full=False,
                           fused_clipped_axes=[1])
    elif case == 'opposite_extents':
        selected = blockers['A']+blockers['B']
        requirement.update(full_views=[], clipped=[[0, 1], [1, 0]])
    elif case == 'partial_top':
        selected = blockers['top']
        requirement.update(top_occlusion=True)
    elif case == 'missing_sides':
        # Both origins lie above the footprint, so only top-face rays hit the
        # box. This is a physical view ablation, not an oracle depth mask.
        delta = 0. if split == 'selection' else .002
        # Higher vertical rays hit inherited stage foreground atZ≈0.46–0.59m
        # in the preserved first native capture. Place these fixture cameras
        # below that obstruction, keeping their XY origins inside the footprint.
        eyes = ((float(center[0]-.004*scale), float(center[1]+delta*scale), .35),
                (float(center[0]+.004*scale), float(center[1]-delta*scale), .40))
        targets = (tuple(center), tuple(center))
        requirement.update(missing_sides=True)
    elif case == 'edge_adjacent_occluder':
        selected = blockers['adjacent']
        requirement.update(occluder_visible=True)
    elif case == 'near_image_boundary':
        targets = (_near_boundary_target(np.asarray(eyes[0]), targets[0], target), targets[1])
        requirement.update(near_image_boundary_view=0)
    elif case == 'alternative_views':
        alternative = ((1.35, .65, 1.05), (.18, -.60, .72)) if split == 'selection' else (
                       (1.42, .54, .98), (.15, -.66, .76))
        eyes = tuple(transform(eye) for eye in alternative)
    clutter = tuple(_transform_obj(obj, f'observability_{split}_{name}_cover_{i}', center, scale, yaw)
                    for i, obj in enumerate(selected))
    return ObservabilityScene(name, split, seed, (target,), clutter, eyes, targets, requirement,
        note='Fixed selection/holdout geometry; actual visibility/depth validity audited independently of perception.')


def observability_scenes(split='selection'):
    """45 scenes/split; never call holdout until the production source is frozen."""
    if split not in SEEDS:
        raise ValueError('Choose selection or explicitly frozen holdout')
    rng = np.random.default_rng(SEEDS[split])
    scenes = []
    for band, bounds in enumerate(((.042, .049), (.061, .069), (.072, .079), (.082, .090))):
        width = float(rng.uniform(*bounds))
        aspect = float(rng.uniform(1.08, 1.24))
        height = float(rng.uniform(.0475, .0525))*width/.09
        yaw = float(rng.uniform(-.65, .65))
        target = box('target', BASE_CENTER[:2], (aspect*width, width, height))
        for case in CASES:
            name = f'band_{band}_{case}'
            scenes.append(_scene(name, split, int(rng.integers(1, 2**31)), target, case, yaw=yaw))
    # Exact mandatory failure geometry/cameras from the preserved earlier run.
    target = box('partial_box_both_target', BASE_CENTER[:2], (.09, .09, .05))
    clutter = tuple(replace(obj, name=f'partial_box_both_occluder_{i}')
                    for i, obj in enumerate(_canonical_blockers()['both']))
    scenes.append(ObservabilityScene('partial_box_both', split, SEEDS[split]+100, (target,), clutter,
        PREVIOUS_EYES['selection'], ((.45, 0., .04),)*2,
        dict(case='mandatory_legacy_90mm', minimum_target_points_per_view=30,
             full_views=[], clipped=[[0, 1], [1, 1]], fused_clipped_axes=[1], fused_full=False),
        note='Exact valid90×90×50mm both-camera hidden-extent regression; never replace by an easier fixture.'))
    controls = (sphere('sphere', (.445, -.01), float(rng.uniform(.024, .028))),
                cylinder('upright', (.445, -.01), float(rng.uniform(.021, .025)), float(rng.uniform(.072, .085))),
                cylinder('lying', (.445, -.01), float(rng.uniform(.017, .021)), float(rng.uniform(.095, .110)),
                         float(rng.uniform(.31, .55))))
    for obj in controls:
        obj = replace(obj, name=f'observability_{split}_{obj.name}_control')
        scenes.append(ObservabilityScene(obj.name, split, SEEDS[split]+len(scenes), (obj,),
            requirements=dict(case='primitive_control', minimum_target_points_per_view=30)))
    width = float(rng.uniform(.052, .058))
    target = box('temporal', BASE_CENTER[:2], (width*1.05, width, .05*width/.09))
    temporal = _scene('temporal_eligibility', split, SEEDS[split]+200, target, 'clear_both')
    incomplete = _scene('temporal_eligibility', split, temporal.seed, target, 'same_extent_hidden_both')
    scenes.append(replace(temporal, temporal=True, occluded_clutter=incomplete.clutter,
                          note='Same physical target/name/center: complete0–3,incomplete4–5,restored6–8; no state mutation.'))
    return scenes


def scene_frame_count(scene):
    return 9 if scene.temporal else 4


def scene_frame_content(scene, index):
    if not 0 <= index < scene_frame_count(scene):
        raise ValueError('Frame index outside fixed fixture sequence')
    if scene.temporal and 4 <= index <= 5:
        return scene.objects, scene.occluded_clutter, 'incomplete'
    return scene.objects, scene.clutter, 'restored' if scene.temporal and index >= 6 else 'complete'


def _extent(points, target):
    if not len(points):
        return dict(points=0, span_mm=None, lower_mm=None, upper_mm=None, side_points=0, top_points=0)
    local = (points-target.center) @ _rotation(target.yaw)
    return dict(points=len(points), span_mm=np.ptp(local, axis=0)*1000,
                lower_mm=local.min(0)*1000, upper_mm=local.max(0)*1000,
                side_points=int(np.count_nonzero((abs(abs(local[:, :2])-target.size[:2]/2) <= .0015).any(1) &
                                                 (local[:, 2] < target.size[2]/2-.002)))
                    if target.shape == 'cube' else None,
                top_points=int(np.count_nonzero(abs(local[:, 2]-target.size[2]/2) <= .0015)))


def _top_ray_evidence(frame, target):
    """Post-capture truth ray audit for physical top-cover preconditions only."""
    grid = np.array([(x, y, .5) for x in np.linspace(-.4, .4, 7) for y in np.linspace(-.4, .4, 7)])
    points = target.center+(grid*target.size) @ _rotation(target.yaw).T
    camera = (points-frame.world_from_camera[:3, 3]) @ frame.world_from_camera[:3, :3]
    uv = camera[:, :2]/camera[:, 2, None]*[frame.intrinsics[0, 0], frame.intrinsics[1, 1]]+frame.intrinsics[:2, 2]-.5
    ij = np.rint(uv).astype(int)
    h, w = frame.depth.shape
    valid = (camera[:, 2] > 0) & (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    observed = frame.depth[ij[valid, 1], ij[valid, 0]]
    predicted = camera[valid, 2]
    finite = np.isfinite(observed) & (observed > 0)
    return dict(samples=len(grid), valid=int(finite.sum()),
                foreground=int(np.count_nonzero(finite & (observed < predicted-.004))),
                surface=int(np.count_nonzero(finite & (abs(observed-predicted) <= .004))))


def audit_observability_scene(scene, frames, *, frame_index=3, phase=None, config=None):
    """Check actual raw-depth fixture requirements; never detector acceptance."""
    objects, clutter, actual_phase = scene_frame_content(scene, frame_index)
    if phase is not None and phase != actual_phase:
        raise ValueError('Phase does not match fixed fixture frame index')
    target, solids = objects[0], objects+clutter
    cfg = config or load_scene_config('config/scenes/multiview_playground.toml').scene
    requirement = dict(scene.requirements or {})
    if scene.temporal and actual_phase == 'incomplete':
        requirement.update(full_views=[], fused_full=False, clipped=[[0, 1], [1, 1]], fused_clipped_axes=[1])
    raw, clouds, errors = [], [], []
    for ci, frame in enumerate(frames):
        points, pixels = project_depth(frame, stride=1)
        keep = ((points >= cfg.workspace_min).all(1) & (points <= cfg.workspace_max).all(1) &
                (points[:, 2] > cfg.support_plane_z+.004))
        points, pixels = points[keep], pixels[keep]
        labels = surface_memberships(points, solids)
        selected = points[labels == 0]
        selected_pixels = pixels[labels == 0]
        entry = _extent(selected, target)
        entry['camera_index'] = ci
        entry['top_ray_evidence'] = _top_ray_evidence(frame, target) if target.shape == 'cube' else None
        entry['occluder_points'] = int(np.count_nonzero(labels > 0))
        h, w = frame.depth.shape
        entry['image_margin_pixels'] = None if not len(selected_pixels) else int(np.min(np.column_stack((
            selected_pixels[:, 0], selected_pixels[:, 1], h-1-selected_pixels[:, 0], w-1-selected_pixels[:, 1]))))
        entry['invalid_depth_fraction'] = float(np.mean(~np.isfinite(frame.depth) | (frame.depth <= 0)))
        if target.shape == 'cube':
            signs = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            corners = target.center+(signs*target.size/2) @ _rotation(target.yaw).T
            camera = (corners-frame.world_from_camera[:3, 3]) @ frame.world_from_camera[:3, :3]
            uv = camera[:, :2]/camera[:, 2, None]*[frame.intrinsics[0, 0], frame.intrinsics[1, 1]]+frame.intrinsics[:2, 2]-.5
            entry['physical_corners_inside_image'] = bool((camera[:, 2] > 0).all() and
                (uv >= 0).all() and (uv <= [w-1, h-1]).all())
        raw.append(entry)
        # Use stride2 cloud for the diagnostic fused extent, as production does.
        stride = (pixels[:, 0] % 2 == 0) & (pixels[:, 1] % 2 == 0)
        clouds.append(voxelize(points[stride], .002, 12000))
    fused = fuse_clouds(clouds, voxel=.002, max_points=12000)
    fused = _extent(fused[surface_memberships(fused, solids) == 0], target)
    if len(raw) != 2:
        errors.append('need_two_raw_native_views')
    elif any(view['points'] < requirement.get('minimum_target_points_per_view', 30) for view in raw):
        errors.append('target_missing_or_insufficient_raw_surface_samples')
    else:
        truth = target.size[:2]*1000
        for ci in requirement.get('full_views', []):
            if np.any(np.asarray(raw[ci]['span_mm'])[:2] < truth-5):
                errors.append(f'view_{ci}_required_full_extent_not_measured')
        for ci, axis in requirement.get('clipped', []):
            if truth[axis]-raw[ci]['span_mm'][axis] < max(5, truth[axis]*.12):
                errors.append(f'view_{ci}_axis_{axis}_required_extent_not_clipped')
        if requirement.get('fused_full') and (fused['span_mm'] is None or np.any(np.asarray(fused['span_mm'])[:2] < truth-5)):
            errors.append('fused_required_full_extent_not_measured')
        for axis in requirement.get('fused_clipped_axes', []):
            if fused['span_mm'] is None or truth[axis]-fused['span_mm'][axis] < max(5, truth[axis]*.12):
                errors.append(f'fused_axis_{axis}_required_extent_not_clipped')
        if requirement.get('top_occlusion') and not any(view['top_ray_evidence']['foreground'] >= 3 for view in raw):
            errors.append('physical_top_cover_not_measured')
        if requirement.get('missing_sides') and any(view['side_points'] > max(4, .05*view['points']) for view in raw):
            errors.append('side_face_measurements_not_missing')
        if requirement.get('occluder_visible') and max(view['occluder_points'] for view in raw) < 8:
            errors.append('adjacent_occluder_not_observed')
        if 'near_image_boundary_view' in requirement:
            edge_view = raw[requirement['near_image_boundary_view']]
            margin = edge_view['image_margin_pixels']
            if margin is None or not 1 <= margin <= 20 or not edge_view['physical_corners_inside_image']:
                errors.append('required_valid_near_image_boundary_silhouette_not_measured')
    entry = dict(frame_index=frame_index, phase=actual_phase, valid_fixture=not errors,
                 errors=errors, raw_cameras=raw, fused_voxel=fused, requirements=requirement)
    return dict(valid=not errors, errors=errors, measurements=[entry])
