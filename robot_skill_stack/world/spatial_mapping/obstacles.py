"""Bounded, exact occupied-cell cuboids prepared with the mapping worker.

No FREE or UNKNOWN cell is filled to simplify geometry. A proxy budget can omit
occupied detail; diagnostics report that undercoverage instead of joining gaps.
All public records contain only immutable SDK-independent values.
"""
from dataclasses import dataclass
import math
import time
from types import MappingProxyType

import numpy as np

from .occupancy import OccupancyState


@dataclass(frozen=True)
class ObstacleExtractionSettings:
    max_obstacles: int = 64
    support_plane_z: float | None = None

    def __post_init__(self):
        if type(self.max_obstacles) is not int or not 1 <= self.max_obstacles <= 128:
            raise ValueError('max_obstacles must be an integer between 1 and 128')
        if self.support_plane_z is not None:
            if (type(self.support_plane_z) not in (int, float) or
                    not math.isfinite(self.support_plane_z)):
                raise ValueError('support_plane_z must be finite or omitted')
            object.__setattr__(self, 'support_plane_z', float(self.support_plane_z))


@dataclass(frozen=True)
class ObstacleBox:
    id: str
    center: tuple
    dimensions: tuple
    expires_at: float
    voxel_count: int

    def __post_init__(self):
        center, dimensions = tuple(map(float, self.center)), tuple(map(float, self.dimensions))
        if (not isinstance(self.id, str) or not self.id or len(center) != 3 or len(dimensions) != 3 or
                not np.isfinite(center+dimensions).all() or any(value <= 0 for value in dimensions) or
                not math.isfinite(self.expires_at) or self.expires_at < 0 or
                type(self.voxel_count) is not int or self.voxel_count < 1):
            raise ValueError('Obstacle boxes require an ID, finite XYZ geometry, expiry and support')
        object.__setattr__(self, 'center', center)
        object.__setattr__(self, 'dimensions', dimensions)
        object.__setattr__(self, 'expires_at', float(self.expires_at))

    @property
    def obstacle_id(self):
        return self.id


@dataclass(frozen=True)
class ObstaclePacket:
    generation: int
    revision: int
    timestamp: float | None
    extracted_at: float
    boxes: tuple
    diagnostics: object

    def __post_init__(self):
        object.__setattr__(self, 'boxes', tuple(self.boxes))
        object.__setattr__(self, 'diagnostics', MappingProxyType(dict(self.diagnostics)))


def _supported(mask):
    """Reject components smaller than three in a radius-two Chebyshev graph.

    Stride/budget-limited depth hits need not occupy face-adjacent voxels. A
    component >=3 has a degree >=2 cell; every degree-one leaf in it touches a
    degree >=2 cell. Separable five-cell sums/ORs implement that neighborhood
    without 124 grid shifts or SciPy. Only original occupied cells are retained;
    the support neighborhood never becomes obstacle geometry.
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
    # At most 5**3=125 cells are summed, so uint8 does not overflow.
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


def _cuboids(mask, *, max_runs=100_000):
    """Run-length Z, then merge identical adjacent runs in Y and slabs in X."""
    padded = np.pad(mask, ((0, 0), (0, 0), (1, 1)))
    edges = np.diff(padded.astype(np.int8), axis=2)
    starts, stops = np.argwhere(edges == 1), np.argwhere(edges == -1)
    if not len(starts):
        return np.empty((0, 6), dtype=np.int64), 0
    lengths = stops[:, 2]-starts[:, 2]
    dropped = 0
    if len(starts) > max_runs:
        # Bound subsequent arrays/sorts even for pathological noisy grids. The
        # largest exact runs are kept; tie-breaking is world-grid lexical order.
        chosen = np.lexsort((starts[:, 2], starts[:, 1], starts[:, 0], -lengths))[:max_runs]
        dropped = int(np.sum(lengths))-int(np.sum(lengths[chosen]))
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
    return result, dropped


class ObstacleExtractor:
    """Worker-only dense extraction; callers publish the small packet in O(1)."""
    def __init__(self, settings=None):
        self.settings = settings or ObstacleExtractionSettings()
        self._key, self._packet = None, None

    def extract(self, snapshot, *, now, evidence_ttl):
        now, evidence_ttl = float(now), float(evidence_ttl)
        if not math.isfinite(now) or now < 0 or not math.isfinite(evidence_ttl) or evidence_ttl <= 0:
            raise ValueError('Obstacle extraction requires finite time and positive evidence TTL')
        key = (snapshot.generation, snapshot.revision, snapshot.timestamp, evidence_ttl)
        if key == self._key and self._packet is not None:
            next_expiry = min((box.expires_at for box in self._packet.boxes), default=math.inf)
            if now < next_expiry:
                return self._packet
        started = time.perf_counter()
        observed_at = snapshot.last_observed
        occupied = ((snapshot.states == OccupancyState.OCCUPIED) & snapshot.observed &
            ~snapshot.expired & np.isfinite(observed_at) & (observed_at <= now) &
            (now-observed_at < evidence_ttl))
        occupied_count = int(np.count_nonzero(occupied))
        supported = _supported(occupied)
        supported_count = int(np.count_nonzero(supported))
        bounds, dropped_runs = _cuboids(supported)
        candidate_count = len(bounds)
        floor_candidates = floor_candidate_voxels = represented_floor = 0
        if len(bounds):
            sizes = bounds[:, [1, 3, 5]]-bounds[:, [0, 2, 4]]
            volumes = np.prod(sizes, axis=1, dtype=np.int64)
            floor = np.zeros(len(bounds), bool)
            if self.settings.support_plane_z is not None:
                tops = np.minimum(snapshot.spec.bounds_min[2]+bounds[:, 5]*snapshot.spec.voxel_size,
                                  snapshot.spec.bounds_max[2])
                floor = tops <= self.settings.support_plane_z+snapshot.spec.voxel_size+1e-12
                floor_candidates = int(np.count_nonzero(floor))
                floor_candidate_voxels = int(volumes[floor].sum())
            # Known support-plane-only boxes follow above-plane hazards when a
            # budget binds. They are retained if slots remain, never erased or
            # enlarged. The support plane is derived from existing scene setup.
            order = np.lexsort((bounds[:, 4], bounds[:, 2], bounds[:, 0], -volumes, floor))
            order = order[:self.settings.max_obstacles]
            bounds, volumes = bounds[order], volumes[order]
            represented_floor = int(volumes[floor[order]].sum())
        else:
            volumes = np.empty(0, dtype=np.int64)
        boxes = []
        lower = np.asarray(snapshot.spec.bounds_min)
        upper = np.asarray(snapshot.spec.bounds_max)
        for cells, volume in zip(bounds, volumes):
            x0, x1, y0, y1, z0, z1 = map(int, cells)
            minimum = lower+np.array([x0, y0, z0])*snapshot.spec.voxel_size
            maximum = np.minimum(lower+np.array([x1, y1, z1])*snapshot.spec.voxel_size, upper)
            oldest = float(np.min(observed_at[x0:x1, y0:y1, z0:z1]))
            boxes.append(ObstacleBox(f'{x0}:{x1}:{y0}:{y1}:{z0}:{z1}',
                tuple((minimum+maximum)/2), tuple(maximum-minimum), oldest+evidence_ttl, int(volume)))
        represented = sum(box.voxel_count for box in boxes)
        diagnostics = dict(occupied_voxels=occupied_count, supported_voxels=supported_count,
            rejected_sparse_voxels=occupied_count-supported_count, candidate_boxes=candidate_count,
            proxy_count=len(boxes), represented_voxels=represented,
            omitted_voxels=supported_count-represented, run_budget_omitted_voxels=dropped_runs,
            truncated=candidate_count > self.settings.max_obstacles or dropped_runs > 0,
            max_obstacles=self.settings.max_obstacles, minimum_component_voxels=3,
            support_radius_voxels=2, support_neighborhood='Chebyshev',
            support_plane_z=self.settings.support_plane_z,
            floor_candidate_boxes=floor_candidates,
            hazard_candidate_boxes=candidate_count-floor_candidates,
            represented_floor_voxels=represented_floor,
            omitted_floor_voxels=floor_candidate_voxels-represented_floor,
            padding_m=0., extract_seconds=time.perf_counter()-started)
        packet = ObstaclePacket(snapshot.generation, snapshot.revision, snapshot.timestamp,
                                now, tuple(boxes), diagnostics)
        self._key, self._packet = key, packet
        return packet
