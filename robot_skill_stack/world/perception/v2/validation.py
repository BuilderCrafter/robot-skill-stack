"""Depth evidence and post-cleaning duplicate checks; never use GT labels."""
import numpy as np
from robot_skill_stack.world.model.primitives import PrimitiveShape
from robot_skill_stack.world.perception.discovery import DepthObjectDiscoverer


def foreground(frame, instance, config, discovery, world, robot_mask):
    mask = instance.mask.copy()
    for _ in range(config.mask_erode_pixels):
        p = np.pad(mask, 1)
        mask = np.logical_and.reduce([p[y:y+mask.shape[0], x:x+mask.shape[1]] for y in range(3) for x in range(3)])
    valid = mask & np.isfinite(world[..., 0])
    count = int(valid.sum())
    evidence = dict(mask_pixels=int(mask.sum()), valid_depth_pixels=count, robot_pixels=int((valid & robot_mask).sum()))
    if count < config.min_mask_pixels:
        return None, evidence, 'insufficient_valid_depth'
    if count/max(1, mask.sum()) < config.min_valid_depth_fraction:
        return None, evidence, 'sparse_valid_depth'
    valid &= ~robot_mask
    nonrobot = int(valid.sum())
    if nonrobot < config.min_mask_pixels:
        return None, evidence, 'robot_surface'
    z = world[..., 2]
    inside = (valid & (world[..., 0] >= discovery.workspace_min[0]) & (world[..., 0] <= discovery.workspace_max[0])
              & (world[..., 1] >= discovery.workspace_min[1]) & (world[..., 1] <= discovery.workspace_max[1])
              & (z > max(discovery.workspace_min[2], discovery.support_plane_z+discovery.min_object_height))
              & (z <= config.max_observation_z))
    pixels = int(inside.sum())
    evidence.update(nonrobot_pixels=nonrobot, foreground_pixels=pixels, foreground_fraction=pixels/max(nonrobot, 1),
                    support_pixels=int((valid & (z <= discovery.support_plane_z+discovery.min_object_height)).sum()))
    if pixels < config.min_mask_pixels:
        return None, evidence, 'outside_workspace_or_support_surface'
    if evidence['foreground_fraction'] < config.min_foreground_fraction:
        return None, evidence, 'insufficient_foreground_fraction'
    parts = DepthObjectDiscoverer._components(inside, world, discovery.component_neighbor_distance)
    ys, xs = max(parts, key=lambda ij: len(ij[0]))
    fraction = len(ys)/pixels
    evidence.update(components=len(parts), dominant_component_fraction=fraction)
    # Do not join separate depth layers via an image mask. Several substantial
    # fragments are retained provisionally; a consistent fit must explain all.
    if fraction >= config.min_component_fraction:
        inside[:] = False
        inside[ys, xs] = True
    return inside, evidence, None


def same_surface(a, b, threshold):
    shared = int(np.count_nonzero(a.mask & b.mask))
    overlap = shared/max(1, min(int(a.mask.sum()), int(b.mask.sum())))
    if overlap >= threshold:
        return True, 'shared_depth_pixels', overlap
    ga, gb = a.metadata['geometry'], b.metadata['geometry']
    if ga.shape == PrimitiveShape.UNKNOWN or ga.shape != gb.shape:
        return False, None, overlap
    # Disjoint fragments may fit the same physical solid. Only strong, mutually
    # consistent full fits qualify; nearby centers alone never trigger suppression.
    if min(ga.confidence, gb.confidence) < .8:
        return False, None, overlap
    scale = min(np.linalg.norm(a.size), np.linalg.norm(b.size))
    if np.linalg.norm(a.position-b.position) > max(.002, .08*scale):
        return False, None, overlap
    if np.linalg.norm(a.size-b.size)/max(scale, 1e-6) > .12:
        return False, None, overlap
    if ga.shape == PrimitiveShape.CYLINDER and abs(np.dot(ga.axis, gb.axis)) < .98:
        return False, None, overlap
    if ga.shape == PrimitiveShape.CUBE and abs((ga.yaw-gb.yaw+np.pi/4)%(np.pi/2)-np.pi/4) > .08:
        return False, None, overlap
    extent = np.maximum(0., np.minimum(a.position+a.size/2, b.position+b.size/2)-
                        np.maximum(a.position-a.size/2, b.position-b.size/2))
    volume = float(np.prod(extent)/max(1e-12, min(np.prod(a.size), np.prod(b.size))))
    return volume >= .85, 'same_fitted_solid' if volume >= .85 else None, overlap


def deduplicate(items, threshold):
    def quality(item):
        c, prediction, detail = item
        ev, g = c.metadata['evidence'], c.metadata['geometry']
        return (prediction.label is not None, g.shape != PrimitiveShape.UNKNOWN,
                ev['foreground_fraction'], ev.get('dominant_component_fraction', 1.), int(c.mask.sum()), c.confidence)
    kept = []
    for item in sorted(items, key=quality, reverse=True):
        for index, previous in enumerate(kept):
            duplicate, reason, overlap = same_surface(item[0], previous[0], threshold)
            if duplicate:
                item[2].update(rejected='duplicate_surface', duplicate_basis=reason, shared_fraction=overlap,
                               duplicate_of=previous[2]['candidate_index'])
                break
        else:
            kept.append(item)
    return kept
