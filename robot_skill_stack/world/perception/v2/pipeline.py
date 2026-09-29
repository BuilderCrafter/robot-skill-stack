"""One fresh RGB-D frame -> model masks -> metric candidates (no ground-truth inputs)."""
from dataclasses import dataclass
import time
import numpy as np
from robot_skill_stack.world.perception.discovery import DepthObjectDiscoverer, ObjectCandidate
from robot_skill_stack.world.perception.localizer import RgbdLocalizer
from robot_skill_stack.world.perception.primitives import PrimitiveEstimator
from robot_skill_stack.world.perception.semantics import SemanticPrediction
from robot_skill_stack.world.perception.diagnostics import describe
from robot_skill_stack.world.perception.v2.geometry import estimate_geometry
from robot_skill_stack.world.perception.v2.types import Instance, Segmentation, validate_frame


@dataclass
class FrameResult:
    frame: object
    segmentation: Segmentation
    candidates: list
    predictions: list
    diagnostics: list
    geometry_s: float = 0.


def erode(mask, count):
    for _ in range(count):
        p = np.pad(mask, 1)
        mask = np.logical_and.reduce([p[y:y+mask.shape[0], x:x+mask.shape[1]] for y in range(3) for x in range(3)])
    return mask


class FrameProcessor:
    def __init__(self, discovery, primitives, config):
        self.discovery, self.config = discovery, config
        self.estimator = PrimitiveEstimator(primitives.classification_threshold, primitives.ambiguity_margin,
                                            primitives.top_band, fit_tolerance=primitives.fit_tolerance)

    def _candidate(self, frame, instance):
        cfg, d = self.config, self.discovery
        mask = erode(instance.mask, cfg.mask_erode_pixels)
        valid = mask & np.isfinite(frame.depth) & (frame.depth > 0)
        ys, xs = np.nonzero(valid)
        if len(xs) < cfg.min_mask_pixels:
            return None, 'insufficient_valid_depth'
        localizer = RgbdLocalizer(frame.intrinsics)
        p = localizer.pixels_to_world(np.column_stack((xs, ys)), frame.depth[ys, xs], frame.world_from_camera)
        inside = (np.isfinite(p).all(1) & (p[:, 0] >= d.workspace_min[0]) & (p[:, 0] <= d.workspace_max[0])
                  & (p[:, 1] >= d.workspace_min[1]) & (p[:, 1] <= d.workspace_max[1])
                  & (p[:, 2] > d.support_plane_z+d.min_object_height)
                  & (p[:, 2] <= cfg.max_observation_z))
        p, ys, xs = p[inside], ys[inside], xs[inside]
        if len(p) < cfg.min_mask_pixels:
            return None, 'outside_workspace_or_support_surface'
        lo, hi = np.percentile(p, [.5, 99.5], axis=0)
        size = np.maximum(hi-lo, 1e-5)
        if max(size) > d.candidate_max_extent:
            return None, 'implausible_extent_or_contaminated_mask'
        clean = np.zeros(frame.depth.shape, bool)
        clean[ys, xs] = True
        candidate = ObjectCandidate((lo+hi)/2, size, clean, float(instance.confidence), True,
                                    {'points': p, 'support_plane_z': d.support_plane_z})
        geometry = estimate_geometry(self.estimator, candidate, instance.label, d.support_contact_tolerance)
        if 'center' in geometry.metadata:
            position, extent = np.asarray(geometry.metadata['center']), np.asarray(geometry.metadata['size_world'])
            if (not np.isfinite(position).all() or not np.isfinite(extent).all()
                    or min(extent) <= 0 or max(extent) > d.candidate_max_extent):
                return None, 'invalid_fitted_geometry'
            candidate.position, candidate.size = position, extent
        candidate.metadata['geometry'] = geometry
        candidate.metadata['visual_label'] = instance.label
        candidate.metadata['position_source'] = 'primitive_fit' if 'center' in geometry.metadata else 'visible_surface_bounds'
        return candidate, None

    def process(self, frame, segmentation):
        validate_frame(frame)
        if segmentation.frame_id != frame.frame_id:
            raise ValueError('Segmentation and depth frame IDs differ')
        start = time.perf_counter()
        selected, diagnostics = [], []
        for inst in segmentation.instances:
            if inst.mask.shape != frame.depth.shape:
                raise ValueError('Mask must use original RGB-D pixel coordinates')
            if inst.confidence >= self.config.confidence and inst.mask.sum() >= self.config.min_mask_pixels:
                selected.append(inst)
        selected.sort(key=lambda i: i.confidence, reverse=True)
        selected = selected[:self.config.max_detections]
        if self.config.unknown_depth_fallback:
            discoverer = DepthObjectDiscoverer(RgbdLocalizer(frame.intrinsics), **vars(self.discovery))
            for c in discoverer.discover(frame):
                if len(selected) >= self.config.max_detections:
                    break
                if not c.spawnable:
                    continue
                # Do not create another track for an already segmented object/fragment.
                covered = any(np.count_nonzero(c.mask & i.mask) / max(1, min(c.mask.sum(), i.mask.sum())) > .2
                              for i in selected)
                if not covered:
                    selected.append(Instance(c.mask, None, c.confidence))
        candidates, predictions = [], []
        for inst in selected:
            c, reason = self._candidate(frame, inst)
            if c is None:
                diagnostics.append(dict(label=inst.label, score=inst.confidence, rejected=reason))
                continue
            candidates.append(c)
            predictions.append(SemanticPrediction(inst.label, inst.confidence))
            detail = describe(c, c.metadata['geometry'])
            detail.update(label=inst.label, score=inst.confidence, rejected=None,
                          position_source=c.metadata['position_source'])
            diagnostics.append(detail)
        return FrameResult(frame, segmentation, candidates, predictions, diagnostics, time.perf_counter()-start)
