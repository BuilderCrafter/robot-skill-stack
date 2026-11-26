"""Bounded voxel evidence, temporal log-odds and immutable occupancy snapshots."""
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
import math
import time

import numpy as np

from robot_skill_stack.common.arrays import readonly_copy


class OccupancyState(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2
    OUT_OF_BOUNDS = 3


@dataclass(frozen=True)
class GridSpec:
    """Upper-exclusive world bounds and clipped boundary voxels."""
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    voxel_size: float = .01
    max_voxels: int = 2_000_000
    shape: tuple[int, int, int] = field(init=False)

    def __post_init__(self) -> None:
        lower, upper = np.asarray(self.bounds_min, float), np.asarray(self.bounds_max, float)
        counts = [math.ceil(float(length)/self.voxel_size) for length in upper-lower]
        # Rounded quotients such as .07/.01 must not add an empty cell.
        shape = tuple(n-int(n > 1 and a+(n-1)*self.voxel_size >= b)
                      for n, a, b in zip(counts, lower, upper))
        if math.prod(shape) > self.max_voxels:
            raise ValueError('Mapping grid exceeds max_voxels')
        object.__setattr__(self, 'bounds_min', tuple(lower))
        object.__setattr__(self, 'bounds_max', tuple(upper))
        object.__setattr__(self, 'shape', shape)

    def world_to_indices(self, positions: np.ndarray | Sequence) -> np.ndarray:
        """Return XYZ cell indices, or -1 for positions outside the grid."""
        points = np.asarray(positions, float).reshape(-1, 3)
        indices = np.full(points.shape, -1, dtype=np.int64)
        valid = ((points >= self.bounds_min) & (points < self.bounds_max)).all(1)
        indices[valid] = np.minimum(np.floor((points[valid]-self.bounds_min)/self.voxel_size),
                                   np.asarray(self.shape)-1).astype(np.int64)
        return indices

    def indices_to_centers(self, indices: np.ndarray | Sequence) -> np.ndarray:
        """Return centers of clipped cells, including partial boundary cells."""
        indices = np.asarray(indices, np.int64).reshape(-1, 3)
        lower = np.asarray(self.bounds_min)+indices*self.voxel_size
        upper = np.minimum(lower+self.voxel_size, self.bounds_max)
        return np.minimum(lower+(upper-lower)/2, np.nextafter(upper, lower))


@dataclass(frozen=True)
class OccupancyQuery:
    state: OccupancyState
    probability: float
    observed: bool
    last_observed: float
    expired: bool


@dataclass(frozen=True)
class OccupancyBatch:
    states: np.ndarray
    probabilities: np.ndarray
    observed: np.ndarray
    last_observed: np.ndarray
    expired: np.ndarray


def _single(batch: OccupancyBatch) -> OccupancyQuery:
    return OccupancyQuery(OccupancyState(batch.states[0]), float(batch.probabilities[0]),
        bool(batch.observed[0]), float(batch.last_observed[0]), bool(batch.expired[0]))


@dataclass(frozen=True)
class OccupancySnapshot:
    """Independent grid values evaluated at the worker's snapshot time."""
    spec: GridSpec
    states: np.ndarray
    probabilities: np.ndarray
    observed: np.ndarray
    last_observed: np.ndarray
    expired: np.ndarray
    timestamp: float | None
    revision: int
    generation: int
    error: str | None

    def query_many(self, positions: np.ndarray | Sequence) -> OccupancyBatch:
        indices = self.spec.world_to_indices(positions)
        outside = (indices < 0).any(1)
        selected = tuple(np.maximum(indices, 0).T)
        values = [array[selected] for array in
                  (self.states, self.probabilities, self.observed, self.last_observed, self.expired)]
        for array, fill in zip(values, (OccupancyState.OUT_OF_BOUNDS, np.nan, False, np.nan, False)):
            array[outside] = fill
        return OccupancyBatch(*map(readonly_copy, values))

    def query(self, position: np.ndarray | Sequence) -> OccupancyQuery:
        return _single(self.query_many(position))


class OccupancyMap:
    """Accumulate measured misses/hits; expired evidence contributes no prior."""
    def __init__(self, spec: GridSpec, *, clock: Callable[[], float] = time.monotonic,
                 hit_log_odds: float = .85, miss_log_odds: float = -.4,
                 log_odds_limit: float = 4., p_free: float = .35, p_occupied: float = .65,
                 evidence_ttl: float = 5.) -> None:
        self.spec, self.clock = spec, clock
        self.hit_log_odds, self.miss_log_odds = hit_log_odds, miss_log_odds
        self.log_odds_limit, self.p_free, self.p_occupied = log_odds_limit, p_free, p_occupied
        self.evidence_ttl = evidence_ttl
        self._log_odds = np.zeros(spec.shape)
        self._last_observed = np.full(spec.shape, np.nan)
        self.timestamp, self.revision, self.generation = None, 0, 0
        self.error = None

    def note_failure(self, error: Exception | str) -> None:
        self.error = str(error)

    def _timestamp(self, timestamp: float, now: float) -> bool:
        if not np.isfinite(timestamp) or timestamp < 0 or timestamp > now+1e-9:
            raise ValueError('Invalid observation timestamp')
        self.error = None
        return self.timestamp is None or timestamp > self.timestamp

    def _flat_indices(self, indices: np.ndarray | Sequence) -> np.ndarray:
        indices = np.asarray(indices, np.int64).reshape(-1, 3)
        return np.unique(np.ravel_multi_index(indices.T, self.spec.shape))

    def integrate(self, free_indices: np.ndarray, occupied_indices: np.ndarray,
                  *, timestamp: float, now: float | None = None) -> bool:
        """Add one vote per cell, giving hits precedence over misses."""
        now = self.clock() if now is None else now
        if not self._timestamp(timestamp, now):
            return False
        free, occupied = self._flat_indices(free_indices), self._flat_indices(occupied_indices)
        free = np.setdiff1d(free, occupied, assume_unique=True)
        if not len(free) and not len(occupied):
            return False
        log_odds, observed_at = self._log_odds.ravel(), self._last_observed.ravel()
        for indices, evidence in ((free, self.miss_log_odds), (occupied, self.hit_log_odds)):
            previous = np.where(now - observed_at[indices] >= self.evidence_ttl, 0., log_odds[indices])
            updated = np.clip(previous + evidence, -self.log_odds_limit, self.log_odds_limit)
            if evidence > 0:
                # A real hit must immediately withhold any contradictory FREE claim.
                updated[1 / (1 + np.exp(-updated)) <= self.p_free] = 0.
            log_odds[indices], observed_at[indices] = updated, timestamp
        self.timestamp = timestamp
        self.revision += 1
        return True

    def replace(self, occupied_indices: np.ndarray, unknown_indices: np.ndarray | None = None,
                *, timestamp: float, now: float | None = None) -> bool:
        """Replace the GT grid, keeping history for cells now declared UNKNOWN."""
        now = self.clock() if now is None else now
        if not self._timestamp(timestamp, now):
            return False
        occupied = self._flat_indices(occupied_indices)
        unknown = self._flat_indices([] if unknown_indices is None else unknown_indices)
        known = np.ones(self._log_odds.size, bool)
        known[unknown], known[occupied] = False, True
        self._log_odds.fill(-self.log_odds_limit)
        self._log_odds.ravel()[unknown] = 0.
        self._log_odds.ravel()[occupied] = self.log_odds_limit
        self._last_observed.ravel()[known] = timestamp
        self.timestamp = timestamp
        self.revision += 1
        return True

    def _classify(self, log_odds: np.ndarray, observed_at: np.ndarray, now: float) -> tuple:
        observed = np.isfinite(observed_at)
        expired = observed & (now - observed_at >= self.evidence_ttl)
        probabilities = np.where(observed & ~expired, 1 / (1 + np.exp(-log_odds)), .5)
        states = np.full(log_odds.shape, OccupancyState.UNKNOWN, dtype=np.uint8)
        states[observed & ~expired & (probabilities <= self.p_free)] = OccupancyState.FREE
        states[observed & ~expired & (probabilities >= self.p_occupied)] = OccupancyState.OCCUPIED
        return states, probabilities, observed, observed_at, expired

    def snapshot(self, *, now: float | None = None) -> OccupancySnapshot:
        now = self.clock() if now is None else now
        return OccupancySnapshot(self.spec, *map(readonly_copy,
            self._classify(self._log_odds, self._last_observed, now)),
            self.timestamp, self.revision, self.generation, self.error)

    def query_many(self, positions: np.ndarray | Sequence,
                   *, now: float | None = None) -> OccupancyBatch:
        """Age and classify only the requested cells, without a dense scan."""
        now = self.clock() if now is None else now
        indices = self.spec.world_to_indices(positions)
        outside = (indices < 0).any(1)
        selected = tuple(np.maximum(indices, 0).T)
        values = self._classify(self._log_odds[selected], self._last_observed[selected], now)
        for array, fill in zip(values, (OccupancyState.OUT_OF_BOUNDS, np.nan, False, np.nan, False)):
            array[outside] = fill
        return OccupancyBatch(*map(readonly_copy, values))

    def query(self, position: np.ndarray | Sequence, *, now: float | None = None) -> OccupancyQuery:
        return _single(self.query_many(position, now=now))
