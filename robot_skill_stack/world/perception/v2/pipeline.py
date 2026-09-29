"""One RGB-D snapshot -> verified surfaces -> deduplicated candidates; V1 unchanged."""
from dataclasses import dataclass, replace
import time
import numpy as np
from robot_skill_stack.world.model.primitives import PrimitiveShape
from robot_skill_stack.world.perception.discovery import DepthObjectDiscoverer, ObjectCandidate
from robot_skill_stack.world.perception.localizer import RgbdLocalizer
from robot_skill_stack.world.perception.primitives import PrimitiveEstimator
from robot_skill_stack.world.perception.semantics import SemanticPrediction
from robot_skill_stack.world.perception.diagnostics import describe
from robot_skill_stack.world.perception.v2.geometry import estimate_geometry
from robot_skill_stack.world.perception.v2.types import Instance, Segmentation, validate_frame
from robot_skill_stack.world.perception.v2.self_filter import SelfFilterResult, filter_robot
from robot_skill_stack.world.perception.v2.validation import foreground, deduplicate


@dataclass
class FrameResult:
    frame: object
    segmentation: Segmentation
    candidates: list
    predictions: list
    diagnostics: list
    geometry_s: float = 0.
    self_filter: SelfFilterResult | None = None


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

    def _candidate(self, frame, instance, world, robot_mask):
        cfg, d = self.config, self.discovery
        clean, evidence, reason = foreground(frame, instance, cfg, d, world, robot_mask)
        if reason:
            return None, None, evidence, reason
        p = world[clean]
        lo, hi = np.percentile(p, [.5, 99.5], axis=0)
        size = np.maximum(hi-lo, 1e-5)
        if max(size) > d.candidate_max_extent:
            return None, None, evidence, 'implausible_extent_or_contaminated_mask'
        candidate = ObjectCandidate((lo+hi)/2, size, clean, float(instance.confidence), True,
                                    {'points': p, 'support_plane_z': d.support_plane_z, 'evidence': evidence})
        geometry = estimate_geometry(self.estimator, candidate, instance.label, d.support_contact_tolerance)
        known = geometry.shape != PrimitiveShape.UNKNOWN
        if evidence['dominant_component_fraction'] < cfg.min_component_fraction and not known:
            return None, None, evidence, 'fragmented_foreground'
        if 'center' in geometry.metadata:
            position, extent = np.asarray(geometry.metadata['center']), np.asarray(geometry.metadata['size_world'])
            if (not np.isfinite(position).all() or not np.isfinite(extent).all()
                    or min(extent) <= 0 or max(extent) > d.candidate_max_extent):
                return None, None, evidence, 'invalid_fitted_geometry'
            candidate.position, candidate.size = position, extent
        label = instance.label
        reason = geometry.metadata.get('reason')
        if reason in {'visual_geometry_disagreement', 'weak_or_ambiguous_fit'}:
            label = None  # Credible foreground still exists, but the proposed name is not verified.
        validation = 'depth_verified' if known else 'unknown' if label is None else 'visual_only_partial_geometry'
        geometry.metadata['class_validation'] = validation
        candidate.metadata.update(geometry=geometry, visual_label=instance.label, class_validation=validation,
                                  position_source='primitive_fit' if known else 'visible_surface_bounds')
        prediction = SemanticPrediction(label, 1. if label is None else instance.confidence)
        return candidate, prediction, evidence, None

    def process(self, frame, segmentation, robot=None):
        validate_frame(frame)
        if segmentation.frame_id != frame.frame_id:
            raise ValueError('Segmentation and depth frame IDs differ')
        start = time.perf_counter()
        if isinstance(robot, SelfFilterResult):
            exclusion = robot.validate(frame.depth.shape)
            if exclusion.metadata.get('enabled') and exclusion.metadata.get('frame_id') != frame.frame_id:
                raise ValueError('Robot mask and RGB-D frame IDs differ')
        elif self.config.robot_self_filter:
            if robot is None:
                raise ValueError('Self-filter enabled but no time-matched robot snapshot; refusing unfiltered V2 frame')
            exclusion = filter_robot(frame, robot, self.config.robot_depth_tolerance_m, self.config.robot_time_tolerance_s)
        else:
            exclusion = SelfFilterResult(np.zeros(frame.depth.shape, bool), np.full(frame.depth.shape, np.inf),
                                         {'enabled': False, 'source': 'disabled', 'removed_pixels': 0})
        valid = np.isfinite(frame.depth) & (frame.depth > 0)
        y, x = np.nonzero(valid)
        world = np.full((*frame.depth.shape, 3), np.nan)
        world[y, x] = RgbdLocalizer(frame.intrinsics).pixels_to_world(np.column_stack((x, y)), frame.depth[y, x], frame.world_from_camera)
        selected, diagnostics, items = [], [], []
        for inst in segmentation.instances:
            if inst.mask.shape != frame.depth.shape:
                raise ValueError('Mask must use original RGB-D pixel coordinates')
            if inst.confidence >= self.config.confidence and inst.mask.sum() >= self.config.min_mask_pixels:
                selected.append(inst)
        selected.sort(key=lambda i: i.confidence, reverse=True)
        selected = selected[:self.config.max_detections]

        def add(inst, source):
            c, prediction, evidence, reason = self._candidate(frame, inst, world, exclusion.mask)
            detail = dict(candidate_index=len(diagnostics), label=inst.label, score=inst.confidence,
                          source=source, rejected=reason, **evidence)
            diagnostics.append(detail)
            if c is not None:
                detail.update(describe(c, c.metadata['geometry']), final_label=prediction.label,
                              class_validation=c.metadata['class_validation'], position_source=c.metadata['position_source'])
                items.append((c, prediction, detail))
        for inst in selected:
            add(inst, 'yolo')
        # Filter before depth fallback too, and suppress overlaps against accepted,
        # cleaned candidates rather than raw (possibly shadow-dominated) masks.
        if self.config.unknown_depth_fallback:
            depth = frame.depth.copy()
            depth[exclusion.mask] = np.nan
            discoverer = DepthObjectDiscoverer(RgbdLocalizer(frame.intrinsics), **vars(self.discovery))
            for c in discoverer.discover(replace(frame, depth=depth)):
                if c.spawnable:
                    add(Instance(c.mask, None, c.confidence), 'depth_fallback')
        kept = deduplicate(items, self.config.duplicate_surface_overlap)
        for item in kept[self.config.max_detections:]:
            item[2]['rejected'] = 'candidate_limit'
        kept = kept[:self.config.max_detections]
        return FrameResult(frame, segmentation, [i[0] for i in kept], [i[1] for i in kept], diagnostics,
                           time.perf_counter()-start, exclusion)
