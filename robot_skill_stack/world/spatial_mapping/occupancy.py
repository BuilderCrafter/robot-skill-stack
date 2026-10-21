"""Bounded, source-independent occupancy evidence in world-coordinate voxels."""
from dataclasses import dataclass
from enum import IntEnum
import math
from numbers import Integral
import time
from types import MappingProxyType

import numpy as np


class OccupancyState(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2
    OUT_OF_BOUNDS = 3


def _readonly(values):
    """Own immutable bytes, so consumers cannot re-enable array writes."""
    array = np.ascontiguousarray(values)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _points(values):
    points = np.asarray(values, dtype=float)
    if not points.size:
        if points.shape != (0,) and not (points.ndim == 2 and points.shape[1] == 3):
            raise ValueError('Expected XYZ points with shape (N, 3)')
        return np.empty((0, 3), dtype=float)
    if points.shape == (3,):
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Expected XYZ points with shape (N, 3)')
    return points


@dataclass(frozen=True)
class GridSpec:
    bounds_min: tuple
    bounds_max: tuple
    voxel_size: float = .01
    max_voxels: int = 2_000_000

    def __post_init__(self):
        lower, upper = np.asarray(self.bounds_min, float), np.asarray(self.bounds_max, float)
        if (lower.shape != (3,) or upper.shape != (3,) or
                not np.isfinite(lower).all() or not np.isfinite(upper).all() or
                np.any(upper <= lower)):
            raise ValueError('Mapping bounds must be finite increasing XYZ coordinates')
        if (isinstance(self.voxel_size, (bool, np.bool_)) or
                not np.isfinite(self.voxel_size) or self.voxel_size <= 0):
            raise ValueError('voxel_size must be finite and positive')
        if (isinstance(self.max_voxels, (bool, np.bool_)) or
                not isinstance(self.max_voxels, Integral) or self.max_voxels < 1):
            raise ValueError('max_voxels must be a positive integer')
        with np.errstate(over='ignore', divide='ignore'):
            sizes = (upper-lower)/self.voxel_size
        if not np.isfinite(sizes).all():
            raise ValueError('Mapping dimensions exceed the voxel budget')
        dimensions = [max(1, math.ceil(float(value))) for value in sizes]
        for axis, count in enumerate(dimensions):
            # A rounded quotient such as .07/.01 must not add an empty cell.
            if count > 1 and lower[axis]+(count-1)*self.voxel_size >= upper[axis]:
                count -= 1
            dimensions[axis] = count
        shape = tuple(dimensions)
        if math.prod(shape) > self.max_voxels:
            raise ValueError(f'Mapping dimensions {shape} exceed max_voxels={self.max_voxels}')
        object.__setattr__(self, 'bounds_min', tuple(lower.tolist()))
        object.__setattr__(self, 'bounds_max', tuple(upper.tolist()))
        object.__setattr__(self, 'voxel_size', float(self.voxel_size))
        object.__setattr__(self, 'max_voxels', int(self.max_voxels))
        object.__setattr__(self, '_shape', shape)

    @property
    def shape(self):
        return self._shape

    @property
    def dimensions(self):
        return self.shape

    @property
    def voxel_count(self):
        return math.prod(self.shape)

    def world_to_indices(self, positions):
        points = _points(positions)
        indices = np.full(points.shape, -1, dtype=np.int64)
        valid = (np.isfinite(points).all(1) & (points >= self.bounds_min).all(1) &
                 (points < self.bounds_max).all(1))
        # Clip only numerical rounding within physically valid boundary cells.
        indices[valid] = np.minimum(np.floor((points[valid]-self.bounds_min)/self.voxel_size),
                                    np.asarray(self.shape)-1).astype(np.int64)
        return indices

    def validate_indices(self, values):
        raw = np.asarray(values)
        if not raw.size:
            if raw.shape != (0,) and not (raw.ndim == 2 and raw.shape[1] == 3):
                raise ValueError('Voxel indices must have shape (N, 3)')
            return np.empty((0, 3), dtype=np.int64)
        if (raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in 'iuf' or
                not np.isfinite(raw).all() or np.any(raw != np.floor(raw)) or
                np.any(raw < 0) or np.any(raw >= self.shape)):
            raise ValueError('Voxel indices must be integer XYZ coordinates inside the grid')
        return raw.astype(np.int64)

    def indices_to_centers(self, indices):
        indices = self.validate_indices(indices)
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
    index: tuple | None


@dataclass(frozen=True)
class OccupancyBatch:
    states: np.ndarray
    probabilities: np.ndarray
    observed: np.ndarray
    last_observed: np.ndarray
    expired: np.ndarray
    indices: np.ndarray


def _query_many(spec, states, probabilities, observed, last_observed, expired, positions):
    indices = spec.world_to_indices(positions)
    valid = (indices >= 0).all(1)
    count = len(indices)
    result = [np.full(count, OccupancyState.OUT_OF_BOUNDS, dtype=np.uint8),
              np.full(count, np.nan), np.zeros(count, dtype=bool),
              np.full(count, np.nan), np.zeros(count, dtype=bool)]
    selected = tuple(indices[valid].T)
    for output, values in zip(result, (states, probabilities, observed, last_observed, expired)):
        output[valid] = values[selected]
    return OccupancyBatch(*map(_readonly, result), _readonly(indices))


def _single(batch):
    if len(batch.states) != 1:
        raise ValueError('query requires one XYZ position')
    valid = batch.states[0] != OccupancyState.OUT_OF_BOUNDS
    return OccupancyQuery(OccupancyState(batch.states[0]), float(batch.probabilities[0]),
        bool(batch.observed[0]), float(batch.last_observed[0]), bool(batch.expired[0]),
        tuple(batch.indices[0].tolist()) if valid else None)


@dataclass(frozen=True)
class OccupancySnapshot:
    spec: GridSpec
    _states: np.ndarray
    _probabilities: np.ndarray
    _observed: np.ndarray
    _last_observed: np.ndarray
    _expired: np.ndarray
    timestamp: float | None
    revision: int
    generation: int
    health: object

    # Immutable V1 exports; consumers may also use queries/counts/centers.
    states = property(lambda self: self._states)
    probabilities = property(lambda self: self._probabilities)
    observed = property(lambda self: self._observed)
    last_observed = property(lambda self: self._last_observed)
    expired = property(lambda self: self._expired)

    @property
    def metadata(self):
        return MappingProxyType(dict(bounds_min=self.spec.bounds_min, bounds_max=self.spec.bounds_max,
            voxel_size=self.spec.voxel_size, dimensions=self.spec.shape, timestamp=self.timestamp,
            revision=self.revision, generation=self.generation, health=self.health))

    def query_many(self, positions):
        return _query_many(self.spec, self.states, self.probabilities, self.observed,
                           self.last_observed, self.expired, positions)

    def query(self, position):
        return _single(self.query_many(position))

    def counts(self):
        values = np.bincount(self.states.ravel(), minlength=4)
        return {state.name.lower(): int(values[state]) for state in OccupancyState}

    def voxel_centers(self, state=None):
        selected = np.ones(self.spec.shape, dtype=bool) if state is None else self.states == OccupancyState(state)
        return _readonly(self.spec.indices_to_centers(np.argwhere(selected)))


class OccupancyMap:
    def __init__(self, spec, *, clock=time.monotonic, hit_log_odds=.85, miss_log_odds=-.4,
                 log_odds_limit=4., p_free=.35, p_occupied=.65, stale_after=1., evidence_ttl=5.):
        if not isinstance(spec, GridSpec):
            raise TypeError('spec must be a validated GridSpec')
        values = (hit_log_odds, miss_log_odds, log_odds_limit, p_free, p_occupied,
                  stale_after, evidence_ttl)
        if (not callable(clock) or not np.isfinite(values).all() or
                hit_log_odds <= 0 or miss_log_odds >= 0 or log_odds_limit <= 0 or
                not 0 < p_free < .5 < p_occupied < 1 or stale_after <= 0 or evidence_ttl <= 0):
            raise ValueError('Invalid occupancy evidence or aging settings')
        self.spec, self.clock = spec, clock
        self.hit_log_odds, self.miss_log_odds = float(hit_log_odds), float(miss_log_odds)
        self.log_odds_limit, self.p_free, self.p_occupied = float(log_odds_limit), float(p_free), float(p_occupied)
        self.stale_after, self.evidence_ttl = float(stale_after), float(evidence_ttl)
        self._log_odds = np.zeros(spec.shape, dtype=np.float64)
        self._observed = np.zeros(spec.shape, dtype=bool)
        self._last_observed = np.full(spec.shape, np.nan)
        self._expired = np.zeros(spec.shape, dtype=bool)
        self.timestamp, self.revision, self.generation = None, 0, 0
        self._status, self._reason, self._last_error = 'waiting', None, None
        self._last_expiration_time = None

    def world_to_indices(self, positions):
        return self.spec.world_to_indices(positions)

    def indices_to_centers(self, indices):
        return self.spec.indices_to_centers(indices)

    def _now(self, now):
        value = float(self.clock() if now is None else now)
        if not np.isfinite(value) or value < 0:
            raise ValueError('Current monotonic time must be finite and nonnegative')
        return value

    def _health(self, now):
        age = None if self.timestamp is None else max(0., now-self.timestamp)
        stale = age is not None and age > self.stale_after
        status = 'stale' if stale and self._status == 'ok' else self._status
        return MappingProxyType(dict(status=status, fresh=status == 'ok', stale=stale,
            age_seconds=age, timestamp=self.timestamp, reason=self._reason, last_error=self._last_error))

    @property
    def health(self):
        now = self._now(None)
        self.expire(now=now)
        return self._health(now)

    def expire(self, *, now=None):
        now = self._now(now)
        if now == self._last_expiration_time:
            return 0
        self._last_expiration_time = now
        expired = self._observed & ~self._expired & (now-self._last_observed >= self.evidence_ttl)
        count = int(np.count_nonzero(expired))
        if count:
            self._log_odds[expired] = 0.
            self._expired[expired] = True
            self.revision += 1
        return count

    def note_no_frame(self, reason='no_new_frame', *, now=None):
        self.expire(now=now)
        self._status, self._reason = 'no_frame', str(reason)

    def note_failure(self, error, *, now=None):
        self.expire(now=now)
        self._status, self._reason, self._last_error = 'error', 'acquisition_failure', str(error)

    def _timestamp(self, timestamp, now):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or timestamp < 0 or timestamp > now+1e-9:
            raise ValueError('Source timestamp must be finite, nonnegative and no later than now')
        if self.timestamp is not None and timestamp <= self.timestamp:
            self.note_no_frame('repeated_or_regressed_timestamp', now=now)
            return None
        return timestamp

    def _accepted(self, timestamp):
        self.timestamp = timestamp
        self.revision += 1
        self._status, self._reason, self._last_error = 'ok', None, None
        self._last_expiration_time = None

    def _flat_indices(self, indices):
        indices = self.spec.validate_indices(indices)
        return np.unique(np.ravel_multi_index(indices.T, self.spec.shape))

    def integrate(self, free_indices, occupied_indices, *, timestamp, now=None):
        now = self._now(now)
        free, occupied = self._flat_indices(free_indices), self._flat_indices(occupied_indices)
        timestamp = self._timestamp(timestamp, now)
        if timestamp is None:
            return False
        if not len(free) and not len(occupied):
            self.note_no_frame('no_evidence', now=now)
            return False
        self.expire(now=now)
        free = np.setdiff1d(free, occupied, assume_unique=True)
        for indices, evidence in ((free, self.miss_log_odds), (occupied, self.hit_log_odds)):
            self._log_odds.ravel()[indices] = np.clip(self._log_odds.ravel()[indices]+evidence,
                -self.log_odds_limit, self.log_odds_limit)
            self._observed.ravel()[indices] = True
            self._last_observed.ravel()[indices] = timestamp
            self._expired.ravel()[indices] = False
        self._accepted(timestamp)
        return True

    def replace(self, occupied_indices, unknown_indices=None, *, timestamp, now=None):
        now = self._now(now)
        occupied = self._flat_indices(occupied_indices)
        unknown = self._flat_indices([] if unknown_indices is None else unknown_indices)
        timestamp = self._timestamp(timestamp, now)
        if timestamp is None:
            return False
        self.expire(now=now)
        known = np.ones(self.spec.voxel_count, dtype=bool)
        known[unknown], known[occupied] = False, True
        self._log_odds.ravel()[known] = -self.log_odds_limit
        self._log_odds.ravel()[unknown] = 0.
        self._log_odds.ravel()[occupied] = self.log_odds_limit
        self._observed.ravel()[known] = True
        self._last_observed.ravel()[known] = timestamp
        self._expired.ravel()[known] = False
        self._accepted(timestamp)
        return True

    def clear(self):
        self._log_odds.fill(0.)
        self._observed.fill(False)
        self._last_observed.fill(np.nan)
        self._expired.fill(False)
        self.timestamp = None
        self.revision += 1
        self.generation += 1
        self._status, self._reason, self._last_error = 'waiting', None, None
        self._last_expiration_time = None

    def _classify(self, log_odds, observed, expired):
        probabilities = 1/(1+np.exp(-log_odds))
        states = np.full(log_odds.shape, OccupancyState.UNKNOWN, dtype=np.uint8)
        usable = observed & ~expired
        states[usable & (probabilities <= self.p_free)] = OccupancyState.FREE
        states[usable & (probabilities >= self.p_occupied)] = OccupancyState.OCCUPIED
        return states, probabilities

    def _arrays(self):
        states, probabilities = self._classify(self._log_odds, self._observed, self._expired)
        return states, probabilities, self._observed, self._last_observed, self._expired

    def snapshot(self, *, now=None):
        now = self._now(now)
        self.expire(now=now)
        return OccupancySnapshot(self.spec, *map(_readonly, self._arrays()), self.timestamp,
                                 self.revision, self.generation, self._health(now))

    def query_many(self, positions, *, now=None):
        now = self._now(now)
        self.expire(now=now)
        indices = self.world_to_indices(positions)
        outside = (indices < 0).any(1)
        selected = tuple(np.maximum(indices, 0).T)
        observed = self._observed[selected]
        expired = self._expired[selected]
        last_observed = self._last_observed[selected]
        states, probabilities = self._classify(self._log_odds[selected], observed, expired)
        states[outside], probabilities[outside] = OccupancyState.OUT_OF_BOUNDS, np.nan
        observed[outside], expired[outside], last_observed[outside] = False, False, np.nan
        return OccupancyBatch(*map(_readonly, (states, probabilities, observed, last_observed, expired, indices)))

    def query(self, position, *, now=None):
        return _single(self.query_many(position, now=now))

    def counts(self, *, now=None):
        return self.snapshot(now=now).counts()

    def voxel_centers(self, state=None, *, now=None):
        return self.snapshot(now=now).voxel_centers(state)
