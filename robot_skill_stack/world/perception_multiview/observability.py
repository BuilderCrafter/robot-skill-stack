"""Current calibrated-ray evidence for table-supported yaw-box dimensions.

A rectangular patch does not establish a physical box edge. A horizontal side
needs measured top-to-free-space transitions at separated positions along that
side. Nearer surfaces, invalid depth and unobserved pixels never prove free
space. Directly measured vertical faces may tighten the resulting interval.

The returned bounds are empirical, resolution-aware estimates under the same
single supported box assumption as the fitter, not universal certified bounds.
No historical geometry, object truth, gripper policy or simulator API is used.
"""
import numpy as np


_DEPTH_TOLERANCE = .002
_PIXEL_RADIUS = 3
_EDGE_SAMPLES = 48
_RIBBON_OFFSETS = np.array([(x, y) for x in range(-_PIXEL_RADIUS, _PIXEL_RADIUS+1)
                                for y in range(-_PIXEL_RADIUS, _PIXEL_RADIUS+1)])
_PIXEL_NEIGHBORS = np.array([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y])
_EDGE_POSITIONS = np.linspace(-.4, .4, _EDGE_SAMPLES)



def _unique_pixels(raw):
    low = raw.min(0)
    high = raw.max(0)
    width = int(high[0])-int(low[0])+1
    height = int(high[1])-int(low[1])+1
    if width*height > np.iinfo(np.int64).max:
        return np.unique(raw, axis=0)
    codes = (raw[:, 1]-low[1])*width+raw[:, 0]-low[0]
    codes = np.unique(codes)
    return np.column_stack((codes % width+low[0], codes // width+low[1]))

def _view_boundary(center, size, rotation, top_z, top_spread, frame, axis, sign):
    """Inspect a bounded pixel ribbon, retaining adjacent occupied/free pairs."""
    k, transform, depth = frame.intrinsics, frame.world_from_camera, frame.depth
    height, width = depth.shape
    other = 1-axis
    local = np.zeros((_EDGE_SAMPLES, 3))
    local[:, axis] = sign*size[axis]/2
    local[:, other] = _EDGE_POSITIONS*size[other]
    boundary = center+local @ rotation.T
    boundary[:, 2] = top_z
    camera = (boundary-transform[:3, 3]) @ transform[:3, :3]
    result = dict(camera_id=frame.camera_id, free=0, occupied=0, occluded=0,
                  missing=0, out_of_frame=0, transitions=0, bins_supported=0,
                  bounded=False, bound_local_m=None)
    if transform[2, 3] <= top_z or np.any(camera[:, 2] <= 0):
        result['out_of_frame'] = _EDGE_SAMPLES
        return result
    projected = camera[:, :2]/camera[:, 2, None]*[k[0, 0], k[1, 1]]+k[:2, 2]-.5
    offsets = _RIBBON_OFFSETS
    pixels = _unique_pixels((np.rint(projected).astype(int)[:, None]+offsets).reshape(-1, 2))
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
              (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    result['out_of_frame'] = int(np.count_nonzero(~inside))
    pixels = pixels[inside]
    rays = np.c_[(pixels[:, 0]+.5-k[0, 2])/k[0, 0],
                 (pixels[:, 1]+.5-k[1, 2])/k[1, 1], np.ones(len(pixels))]
    world_rays = rays @ transform[:3, :3].T
    with np.errstate(divide='ignore', invalid='ignore'):
        expected_depth = (top_z-transform[2, 3])/world_rays[:, 2]
    world = transform[:3, 3]+world_rays*expected_depth[:, None]
    q = (world-center) @ rotation
    keep = ((expected_depth > 0) & np.isfinite(q).all(1) &
            (abs(q[:, other]) <= .38*size[other]))
    pixels, q, world_rays, expected_depth = (value[keep] for value in
                                             (pixels, q, world_rays, expected_depth))
    if not len(pixels):
        return result
    measured = depth[pixels[:, 1], pixels[:, 0]]
    valid = np.isfinite(measured) & (measured > 0)
    free = valid & (measured > expected_depth+_DEPTH_TOLERANCE)
    foreground = valid & (measured < expected_depth-_DEPTH_TOLERANCE)
    occupied = valid & ~free & ~foreground
    for name, mask in (('free', free), ('occupied', occupied), ('occluded', foreground), ('missing', ~valid)):
        result[name] = int(np.count_nonzero(mask))
    # An anchor belongs to the measured interior, rather than a second surface
    # farther outside the candidate. Calibration/plane spread is explicit below.
    occupied &= sign*q[:, axis] <= size[axis]/2+top_spread
    free_ids = np.flatnonzero(free)
    if not len(free_ids) or not occupied.any():
        return result
    # Eight image neighbors, using bounded sorted lookups rather than a dense
    # all-pairs point or image search for every box side.
    neighbors = _PIXEL_NEIGHBORS
    ids = pixels[:, 1]*width+pixels[:, 0]
    order = np.argsort(ids)
    nearby = pixels[free_ids, None]+neighbors
    nearby_ids = nearby[..., 1]*width+nearby[..., 0]
    positions = np.searchsorted(ids[order], nearby_ids)
    selected = order[np.minimum(positions, len(order)-1)]
    paired = ((positions < len(order)) & (ids[selected] == nearby_ids) & occupied[selected] &
              (sign*(q[selected, axis]-q[free_ids, None, axis]) < 0))
    # Prevent row wrapping at the image border.
    paired &= ((nearby[..., 0] >= 0) & (nearby[..., 0] < width) &
               (nearby[..., 1] >= 0) & (nearby[..., 1] < height))
    witnesses = free_ids[paired.any(1)]
    result['transitions'] = len(witnesses)
    if not len(witnesses):
        return result
    outward = sign*q[witnesses, axis]
    # Plane-height uncertainty changes the lateral intersection of oblique rays.
    local_rays = world_rays[witnesses] @ rotation
    plane_margin = top_spread*abs(local_rays[:, axis]/world_rays[witnesses, 2])
    upper = outward+plane_margin
    bins = []
    for lower, higher in ((-.38, -.1267), (-.1267, .1267), (.1267, .38)):
        chosen = ((q[witnesses, other] >= lower*size[other]) &
                  (q[witnesses, other] <= higher*size[other]))
        if chosen.any():
            bins.append(float(upper[chosen].min()))
    result['bins_supported'] = len(bins)
    if len(bins) == 3:
        result.update(bounded=True, bound_local_m=float(max(size[axis]/2, max(bins))))
    return result


def _face_plane(local, size, axis, sign):
    """Measured side-plane support, excluding top and adjacent-face corners."""
    other = 1-axis
    selected = ((abs(local[:, axis]-sign*size[axis]/2) <= .0015) &
                (abs(local[:, other]) < .35*size[other]) &
                (abs(local[:, 2]) < .35*size[2]))
    points = local[selected]
    result = dict(points=len(points), supported=False)
    if (len(points) < 6 or np.ptp(points[:, 2]) < .25*size[2] or
            np.ptp(points[:, other]) < .3*size[other]):
        return result
    design = np.c_[points[:, other], np.ones(len(points))]
    slope, intercept = np.linalg.lstsq(design, sign*points[:, axis], rcond=None)[0]
    spread = float(np.percentile(abs(design @ [slope, intercept]-sign*points[:, axis]), 95))
    if spread <= _DEPTH_TOLERANCE:
        result.update(supported=True, slope=float(slope), intercept_m=float(intercept), spread_m=spread)
    return result


def _unbounded_status(views):
    totals = {name: sum(view[name] for view in views) for name in ('occluded', 'missing', 'out_of_frame')}
    if any(totals.values()):
        return max(totals, key=totals.get)
    return 'insufficient_boundary_evidence'


def assess(fit, points, frames, *, view_clouds=(), support_plane_z=0.):
    """Return box-side evidence and current conservative geometry, without edits.

    Non-box primitives remain governed by their existing fitter finalization.
    A caller must combine ``observable`` with that existing parameter decision.
    ``conservative_geometry`` is present only when every box dimension is
    supported. Its local axis intervals refer to the input fit's center/yaw.
    """
    geometry = fit.geometry
    shape = geometry.shape.value
    result = dict(method='current_box_boundary_transitions', applicable=shape == 'cube',
                  observable=shape not in ('cube', 'unknown'), reason='not_a_box', axes=[],
                  conservative_geometry=None,
                  measurement_resolution=dict(depth_tolerance_m=_DEPTH_TOLERANCE,
                      pixel_ribbon_radius=_PIXEL_RADIUS, edge_samples=_EDGE_SAMPLES),
                  assumptions='Single table-supported yaw box; empirical measured bounds, not certified hidden geometry')
    if shape != 'cube':
        if shape == 'unknown':
            result.update(observable=False, reason='unknown_geometry')
        return result
    if not np.isfinite(support_plane_z):
        raise ValueError('Support plane must be finite')
    p = np.asarray(points, float).reshape(-1, 3)
    p = p[np.isfinite(p).all(1)]
    size, center = np.asarray(geometry.box_size, float), np.asarray(fit.center, float)
    if (size.shape != (3,) or center.shape != (3,) or not np.isfinite([size, center]).all()
            or np.any(size <= 0) or len(p) < 30 or not frames):
        result.update(observable=False, reason='insufficient_current_box_evidence')
        return result
    c, s = np.cos(geometry.yaw), np.sin(geometry.yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    local = (p-center) @ rotation
    measured_fit = geometry.metadata.get('surface_fit', {})
    measured_size = measured_fit.get('box_size', size)
    measured_center = measured_fit.get('center', center)
    top_z = float(measured_center[2]+measured_size[2]/2)
    top = p[abs(p[:, 2]-top_z) <= _DEPTH_TOLERANCE]
    top_supported = (len(top) >= 12 and
                     np.all(np.ptp((top-center) @ rotation, axis=0)[:2] >= .25*size[:2]))
    top_local = (top-center) @ rotation
    core = top[(abs(top_local[:, :2]) < .35*size[:2]).all(1)]
    # Edge voxels average top and vertical-side samples. They are useful extent
    # evidence, but their height variation is not top-plane measurement noise.
    plane_samples = core if len(core) >= 6 else top
    top_spread = float(np.percentile(abs(plane_samples[:, 2]-top_z), 95)) if len(plane_samples) else 0.
    support = bool(p[:, 2].min() <= support_plane_z+.014 and
                   abs(center[2]-size[2]/2-support_plane_z) <= .004)
    result.update(current_points=len(p), top_samples=len(top), top_spread_m=top_spread,
                  contributing_views=[len(cloud) for cloud in view_clouds])
    limits = np.c_[-size/2, size/2]
    faces = []
    for axis in (0, 1):
        entry = dict(axis='xyz'[axis], measured_interval_m=limits[axis].tolist())
        for sign, label in ((-1, 'negative'), (1, 'positive')):
            views = [_view_boundary(center, size, rotation, top_z, top_spread, frame, axis, sign)
                     for frame in frames]
            bounds = [view['bound_local_m'] for view in views if view['bounded']]
            face = _face_plane(local, size, axis, sign)
            side = dict(status='bounded' if bounds else _unbounded_status(views),
                        bound_local_m=None, cameras=views, physical_face=face)
            if bounds:
                outer = min(bounds)
                limits[axis, int(sign > 0)] = sign*outer
                side.update(bound_local_m=sign*outer, bound_source='calibrated_top_ray_transition')
            entry[label] = side
            faces.append((axis, sign, side))
        result['axes'].append(entry)
    # A measured side face and its independently witnessed top/free transition
    # jointly constrain a physical edge more tightly than image sampling alone.
    # Fit plane slope accounts for small yaw error; spread bounds measured noise.
    for axis, sign, side in faces:
        face = side['physical_face']
        if side['status'] != 'bounded' or not face['supported']:
            continue
        other = limits[1-axis]
        upper = (face['intercept_m']+max(face['slope']*other)+2*face['spread_m'])
        outer = min(abs(side['bound_local_m']), max(size[axis]/2, upper))
        limits[axis, int(sign > 0)] = sign*outer
        side.update(bound_local_m=float(sign*outer), bound_source='measured_face_and_ray_transition')
    result['axes'].append(dict(axis='z', measured_interval_m=[-size[2]/2, size[2]/2],
        negative=dict(status='bounded' if support else 'missing', bound_local_m=-size[2]/2,
                      source='configured_support_plane_with_current_low_side_samples'),
        positive=dict(status='bounded' if top_supported else 'missing', bound_local_m=size[2]/2,
                      source='measured_top_plane')))
    horizontal = all(side['status'] == 'bounded' for _, _, side in faces)
    result['observable'] = bool(horizontal and support and top_supported)
    result['reason'] = ('current_dimensions_bounded' if result['observable'] else
                        'unbounded_box_side' if not horizontal else 'unobserved_top_or_support')
    if result['observable']:
        conservative_center = center+limits.mean(1) @ rotation.T
        result['conservative_geometry'] = dict(center=conservative_center.tolist(),
            box_size=(limits[:, 1]-limits[:, 0]).tolist(), yaw=float(geometry.yaw))
    return result
