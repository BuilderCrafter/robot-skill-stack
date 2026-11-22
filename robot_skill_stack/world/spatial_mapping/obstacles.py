"""Pack supported occupied cells into bounded cuboids without filling gaps."""
from dataclasses import dataclass

import numpy as np

from .occupancy import OccupancySnapshot, OccupancyState


@dataclass(frozen=True)
class ObstacleExtractionSettings:
    """Proxy budget and optional support height for floor prioritization."""
    max_obstacles: int = 64
    support_plane_z: float | None = None


@dataclass(frozen=True)
class ObstacleBox:
    """Exact occupied-cell cuboid with its oldest evidence deadline."""
    id: str
    center: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    expires_at: float


@dataclass(frozen=True)
class ObstaclePacket:
    """Selected cuboids; the enclosing map publication carries their identity."""
    boxes: tuple[ObstacleBox, ...]
    support_plane_z: float | None


def _supported(mask: np.ndarray) -> np.ndarray:
    """Keep radius-two Chebyshev components containing at least three cells.

    Each qualifying component has a cell with two neighbors. Keep those cells
    and their neighbors, intersecting with the original occupied mask.
    """
    count = mask.astype(np.uint8)
    for axis in range(3):
        previous, count = count, count.copy()
        for offset in (1, 2):
            left, right = [slice(None)]*3, [slice(None)]*3
            left[axis], right[axis] = slice(None, -offset), slice(offset, None)
            left, right = tuple(left), tuple(right)
            count[left] += previous[right]
            count[right] += previous[left]
    # The separable five-cell sums total at most 125, fitting in uint8.
    nearby = mask & (count >= 3)
    for axis in range(3):
        previous, nearby = nearby, nearby.copy()
        for offset in (1, 2):
            left, right = [slice(None)]*3, [slice(None)]*3
            left[axis], right[axis] = slice(None, -offset), slice(offset, None)
            left, right = tuple(left), tuple(right)
            nearby[left] |= previous[right]
            nearby[right] |= previous[left]
    return mask & nearby


def _cuboids(mask: np.ndarray, *, max_runs: int = 100_000) -> np.ndarray:
    """Merge Z runs into Y slabs, then join identical adjacent slabs along X."""
    padded = np.pad(mask, ((0, 0), (0, 0), (1, 1)))
    edges = np.diff(padded.astype(np.int8), axis=2)
    starts, stops = np.argwhere(edges == 1), np.argwhere(edges == -1)
    if not len(starts):
        return np.empty((0, 6), dtype=np.int64)
    if len(starts) > max_runs:
        # Bound sorting work by retaining the largest exact runs, with lexical ties.
        lengths = stops[:, 2]-starts[:, 2]
        chosen = np.lexsort((starts[:, 2], starts[:, 1], starts[:, 0], -lengths))[:max_runs]
        starts, stops = starts[chosen], stops[chosen]
    order = np.lexsort((starts[:, 1], stops[:, 2], starts[:, 2], starts[:, 0]))
    starts, stops = starts[order], stops[order]
    same = ((starts[1:, 0] == starts[:-1, 0]) &
            (starts[1:, 2] == starts[:-1, 2]) &
            (stops[1:, 2] == stops[:-1, 2]) &
            (starts[1:, 1] == starts[:-1, 1]+1))
    first = np.r_[0, np.flatnonzero(~same)+1]
    last = np.r_[first[1:]-1, len(starts)-1]
    boxes = np.column_stack((starts[first, 0], starts[first, 0]+1,
        starts[first, 1], starts[last, 1]+1, starts[first, 2], stops[first, 2]))
    order = np.lexsort((boxes[:, 0], boxes[:, 5], boxes[:, 4], boxes[:, 3], boxes[:, 2]))
    boxes = boxes[order]
    same = (np.all(boxes[1:, 2:] == boxes[:-1, 2:], axis=1) &
            (boxes[1:, 0] == boxes[:-1, 1]))
    first = np.r_[0, np.flatnonzero(~same)+1]
    last = np.r_[first[1:]-1, len(boxes)-1]
    result = boxes[first].copy()
    result[:, 1] = boxes[last, 1]
    return result


class ObstacleExtractor:
    """Extract exact cuboids, optionally retaining confident sparse RGB-D hits."""
    def __init__(self, settings: ObstacleExtractionSettings | None = None,
                 *, temporal_probabilities: bool = False) -> None:
        self.settings = settings or ObstacleExtractionSettings()
        self._temporal_probabilities = temporal_probabilities

    def extract(self, snapshot: OccupancySnapshot, *, now: float,
                evidence_ttl: float) -> ObstaclePacket:
        """Select supported, unexpired occupancy and rank hazards before floor."""
        spec = snapshot.spec
        lower, upper = np.asarray(spec.bounds_min), np.asarray(spec.bounds_max)
        observed_at = snapshot.last_observed
        occupied = ((snapshot.states == OccupancyState.OCCUPIED) & snapshot.observed &
            ~snapshot.expired & np.isfinite(observed_at) & (observed_at <= now) &
            (now-observed_at < evidence_ttl))
        supported = _supported(occupied)
        if self._temporal_probabilities:
            # Only RGB-D hit/miss integration gives saturated probabilities temporal meaning.
            supported |= occupied & (snapshot.probabilities >= .95)
        bounds = _cuboids(supported)
        volumes = np.prod(bounds[:, [1, 3, 5]]-bounds[:, [0, 2, 4]], axis=1, dtype=np.int64)
        floor = np.zeros(len(bounds), bool)
        support_z = self.settings.support_plane_z
        if support_z is not None:
            tops = np.minimum(lower[2]+bounds[:, 5]*spec.voxel_size, upper[2])
            floor = tops <= support_z+spec.voxel_size+1e-12
        # Largest hazards first, then floor; ties use grid-coordinate order.
        selected = np.lexsort((bounds[:, 4], bounds[:, 2], bounds[:, 0], -volumes, floor))[
            :self.settings.max_obstacles]
        boxes = []
        for cells in bounds[selected]:
            start, stop = cells[::2], cells[1::2]
            minimum = lower+start*spec.voxel_size
            maximum = np.minimum(lower+stop*spec.voxel_size, upper)
            oldest = float(observed_at[tuple(slice(a, b) for a, b in zip(start, stop))].min())
            boxes.append(ObstacleBox(':'.join(map(str, cells)), tuple((minimum+maximum)/2),
                tuple(maximum-minimum), oldest+evidence_ttl))
        return ObstaclePacket(tuple(boxes), support_z)
