"""Keep semantic identity separate from conservative, depth-supported geometry."""
import numpy as np
from robot_skill_stack.world.model.primitives import PrimitiveGeometry, PrimitiveShape


def estimate_geometry(estimator, candidate, label, contact_tolerance):
    p = candidate.metadata['points']
    plane = candidate.metadata['support_plane_z']
    supported = bool(np.percentile(p[:, 2], .5) <= plane+contact_tolerance)
    if supported:
        result = estimator.estimate(candidate)
    else:
        # A lifted object must NOT be stretched down to the table. Free sphere and
        # horizontal-cylinder fits do not need that assumption. Cap-only box/upright
        # cylinder height remains underdetermined here: publish unknown geometry.
        fits = [g for g in (estimator._sphere(p, plane), estimator._horizontal(p, plane)) if g is not None]
        fits.sort(key=lambda g: g.confidence, reverse=True)
        scores = {g.shape.value: g.confidence for g in fits}
        if fits and fits[0].confidence >= estimator.threshold and (
                len(fits) == 1 or fits[0].confidence-fits[1].confidence >= estimator.margin):
            result = fits[0]
            result.scores = scores
        else:
            result = PrimitiveGeometry(PrimitiveShape.UNKNOWN, scores=scores,
                                       metadata={'reason': 'unsupported_or_partial_surface'})
    if label is None:
        result = PrimitiveGeometry(PrimitiveShape.UNKNOWN, metadata={'reason': 'unknown_depth_candidate'})
    elif result.shape != PrimitiveShape.UNKNOWN and result.shape.value != label:
        result = PrimitiveGeometry(PrimitiveShape.UNKNOWN, scores=result.scores,
                                   metadata={'reason': 'visual_geometry_disagreement'})
    result.metadata['support_evidence'] = 'near_support_plane' if supported else 'not_observed'
    return result
