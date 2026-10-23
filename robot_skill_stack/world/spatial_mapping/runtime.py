"""SDK-free worker-owned mapping and constant-time immutable publications.

Only explicit dense exports scan the grid. Snapshot construction on the owner
thread fixes the read time; sparse queries age just the requested cells.
"""
from dataclasses import dataclass, replace
import time
from types import MappingProxyType

import numpy as np

from robot_skill_stack.runtime.worker import BoundedWorker
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from .geometry import freeze, rasterize_ground_truth
from .occupancy import (OccupancyMap, OccupancyState, OccupancyBatch, _readonly, _single)
from .rgbd import RgbdMapper
from .robot_filter import RobotSelfFilter


def _owned(array):
    """Reuse genuinely immutable buffers from the camera pool, otherwise copy."""
    value = np.asarray(array)
    base = value
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    return value if isinstance(base, bytes) and not value.flags.writeable else _readonly(value)


@dataclass(frozen=True)
class MappingInput:
    generation: int = 0
    frames: tuple = ()
    robot_snapshots: object = None
    exclusion_masks: object = None
    ground_truth: object = None
    reset: bool = False

    def __post_init__(self):
        frames = tuple(DepthFrame(frame.camera_id, tuple(frame.token), _owned(frame.depth),
            _owned(frame.intrinsics), _owned(frame.world_from_camera), frame.acquired_at,
            None if frame.rgb is None else _owned(frame.rgb)) for frame in self.frames)
        if self.ground_truth is not None and (frames or self.robot_snapshots is not None):
            raise ValueError('GT input cannot contain RGB-D frames or robot poses')
        object.__setattr__(self, 'frames', frames)
        if self.robot_snapshots is not None:
            object.__setattr__(self, 'robot_snapshots', MappingProxyType(dict(self.robot_snapshots)))
        if self.exclusion_masks is not None:
            object.__setattr__(self, 'exclusion_masks', MappingProxyType(
                {key: _owned(mask) for key, mask in self.exclusion_masks.items()}))


def _input_timestamp(payload):
    if payload.ground_truth is not None:
        return payload.ground_truth.timestamp
    try:
        stamp = min((float(frame.acquired_at) for frame in payload.frames), default=None)
        return stamp if stamp is not None and np.isfinite(stamp) else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class MappingCompleted:
    snapshot: object
    diagnostics: object
    accepted: bool
    generation: int
    input_timestamp: float | None = None


class MappingProcessor:
    """One mutable map, touched only by the single computation worker."""
    def __init__(self, settings, *, clock=time.monotonic):
        self.settings, self.clock = settings, clock
        self.map = self.mapper = self.filter = None
        self.generation = None

    def _reset(self, generation):
        cfg = self.settings
        self.map = OccupancyMap(cfg.grid_spec(), clock=self.clock,
            stale_after=cfg.stale_after, evidence_ttl=cfg.evidence_ttl)
        self.map.generation = generation
        self.mapper = RgbdMapper(self.map, stride=cfg.stride, max_rays=cfg.max_rays,
            max_voxel_visits=cfg.max_voxel_visits, max_depth=cfg.max_depth,
            max_skew=cfg.max_skew, max_frame_age=cfg.max_frame_age,
            expected_camera_ids=cfg.camera_prim_paths or None, clock=self.clock)
        self.filter = RobotSelfFilter(stride=cfg.stride,
            depth_tolerance=cfg.robot_depth_tolerance, pose_tolerance=cfg.robot_pose_tolerance)
        self.generation = generation

    def compute(self, payload):
        if self.map is None or payload.generation != self.generation or payload.reset:
            self._reset(payload.generation)
        started = time.perf_counter()
        if self.settings.source == 'ground_truth':
            if payload.ground_truth is None:
                raise ValueError('GT mapping requires prepared geometry')
            indices, diagnostics = rasterize_ground_truth(self.map.spec, payload.ground_truth,
                                                         self.settings.max_candidate_voxels)
            accepted = self.map.replace(indices, timestamp=payload.ground_truth.timestamp,
                                        now=self.clock())
        else:
            if payload.ground_truth is not None:
                raise ValueError('RGB-D mapping cannot consume GT geometry')
            provider = None
            if self.settings.automatic_robot_filter:
                if payload.robot_snapshots is None:
                    raise ValueError('Automatic filtering requires exposure-matched robot snapshots')
                def provider(frames, pixels):
                    return self.filter.masks(frames, payload.robot_snapshots, pixels_by_camera=pixels)
            accepted = self.mapper.update(payload.frames, exclusion_masks=payload.exclusion_masks,
                                          mask_provider=provider)
            diagnostics = dict(self.mapper.diagnostics)
            if provider is not None:
                diagnostics['robot_filter'] = dict(self.filter.diagnostics)
        snapshot = self.map.snapshot(now=self.clock())
        diagnostics.update(accepted=accepted, compute_seconds=time.perf_counter()-started)
        return MappingCompleted(snapshot, freeze(diagnostics), accepted, payload.generation,
                                _input_timestamp(payload))


@dataclass(frozen=True, init=False, eq=False)
class MappingReadView:
    """Immutable fixed-time facade over an already immutable worker snapshot."""
    def __init__(self, snapshot, *, now, evidence_ttl, stale_after, health=None):
        now = float(now)
        if not np.isfinite(now) or now < 0:
            raise ValueError('Read time must be finite and nonnegative')
        for key, value in dict(_snapshot=snapshot, evaluated_at=now,
                evidence_ttl=evidence_ttl, stale_after=stale_after, spec=snapshot.spec,
                timestamp=snapshot.timestamp, revision=snapshot.revision,
                generation=snapshot.generation).items():
            object.__setattr__(self, key, value)
        values = dict(snapshot.health if health is None else health)
        age = None if self.timestamp is None else max(0., now-self.timestamp)
        stale = age is not None and age > stale_after
        status = values.get('status', 'waiting')
        if stale and status == 'ok':
            status = 'stale'
        values.update(status=status, age_seconds=age, stale=stale,
                      fresh=status == 'ok' and not stale, timestamp=self.timestamp)
        object.__setattr__(self, 'health', MappingProxyType(values))

    def _age(self, states, probabilities, observed, last_observed, expired):
        expired = expired | (observed & (self.evaluated_at-last_observed >= self.evidence_ttl))
        states, probabilities = states.copy(), probabilities.copy()
        states[expired], probabilities[expired] = OccupancyState.UNKNOWN, .5
        return states, probabilities, expired

    def query_many(self, positions):
        base = self._snapshot.query_many(positions)
        states, probabilities, expired = self._age(base.states, base.probabilities,
            base.observed, base.last_observed, base.expired)
        return OccupancyBatch(_readonly(states), _readonly(probabilities), base.observed,
                              base.last_observed, _readonly(expired), base.indices)

    def query(self, position):
        return _single(self.query_many(position))

    @property
    def states(self):
        return _readonly(self._age(self._snapshot.states, self._snapshot.probabilities,
            self.observed, self.last_observed, self._snapshot.expired)[0])

    @property
    def probabilities(self):
        return _readonly(self._age(self._snapshot.states, self._snapshot.probabilities,
            self.observed, self.last_observed, self._snapshot.expired)[1])

    @property
    def expired(self):
        return _readonly(self._snapshot.expired | (self.observed &
            (self.evaluated_at-self.last_observed >= self.evidence_ttl)))

    observed = property(lambda self: self._snapshot.observed)
    last_observed = property(lambda self: self._snapshot.last_observed)

    @property
    def metadata(self):
        return MappingProxyType(dict(self._snapshot.metadata, generation=self.generation,
                                     health=self.health, evaluated_at=self.evaluated_at))

    def counts(self):
        values = np.bincount(self.states.ravel(), minlength=4)
        return {state.name.lower(): int(values[state]) for state in OccupancyState}

    def voxel_centers(self, state=None):
        selected = np.ones(self.spec.shape, bool) if state is None else self.states == OccupancyState(state)
        return _readonly(self.spec.indices_to_centers(np.argwhere(selected)))


class MappingService:
    """Owner-thread admission/publication; no SDK calls and no map access on poll."""
    def __init__(self, settings, *, worker_factory=BoundedWorker, clock=time.monotonic):
        self.settings, self.clock = settings, clock
        self._processor = MappingProcessor(settings, clock=clock)
        self._worker = worker_factory(self._processor.compute, name='spatial-mapping')
        # Allocate the empty publication once at construction, never on reads.
        self._empty = OccupancyMap(settings.grid_spec(), clock=clock,
            stale_after=settings.stale_after, evidence_ttl=settings.evidence_ttl).snapshot(now=clock())
        self._published = None
        self.generation, self._barrier, self._closed = 0, 0, False
        self._status, self._reason, self._last_error = 'waiting', None, None
        self.last_result, self.diagnostics = None, MappingProxyType({})
        self._counts = dict(submitted=0, published=0, rejected=0, failures=0, source_failures=0, worker_failures=0, processing_failures=0, resets=0)
        self._completion = {}

    def submit(self, payload):
        if self._closed or payload.generation != self.generation:
            self._counts['rejected'] += 1
            return False
        accepted = self._worker.submit(payload, tag=(self.generation, self._barrier, _input_timestamp(payload)))
        self._counts['submitted'] += int(accepted)
        return accepted

    def poll(self, now=None):
        if self._closed:
            return None
        result = self._worker.poll()
        if result is None:
            return None
        now = self.clock() if now is None else float(now)
        if not np.isfinite(now) or now < 0:
            raise ValueError('Publication time must be finite and nonnegative')
        if result.tag[:2] != (self.generation, self._barrier):
            self._counts['rejected'] += 1
            return None
        timing = dict(worker_seconds=result.duration,
            queue_seconds=result.started_at-result.queued_at, completed_at=result.completed_at,
            worker_to_publication_seconds=max(0., now-result.completed_at))
        if result.error is not None:
            self._barrier += 1
            self._worker.invalidate()
            self._status, self._reason, self._last_error = 'error', 'worker_failure', result.error
            self._counts['worker_failures'] += 1
            self._counts['failures'] += 1
            self._completion = dict(timing, timestamp=None, input_timestamp=result.tag[2],
                                    status='worker_failure', filter_seconds=None, compute_seconds=None, transport_seconds=None,
                                    publish_age_seconds=None if result.tag[2] is None else max(0., now-result.tag[2]))
            return None
        completed = result.value
        if completed.generation != self.generation:
            self._counts['rejected'] += 1
            return None
        timestamp = completed.snapshot.timestamp
        age = None if timestamp is None else max(0., now-timestamp)
        limit = self.settings.stale_after
        if self.settings.source == 'rgbd':
            limit = min(limit, self.settings.max_frame_age)
        fresh = (completed.accepted and timestamp is not None and timestamp <= now+1e-9
                 and age < limit and completed.snapshot.health['status'] == 'ok')
        processing_error = not completed.accepted and completed.snapshot.health['status'] == 'error'
        status = ('published' if fresh else 'processing_failure' if processing_error else
                  'stale_completion' if completed.accepted else completed.snapshot.health['status'])
        filter_stats = completed.diagnostics.get('robot_filter', {})
        compute_seconds = completed.diagnostics.get('compute_seconds')
        self._completion = dict(timing, timestamp=timestamp,
            input_timestamp=completed.input_timestamp, status=status,
            publish_age_seconds=(None if completed.input_timestamp is None else
                                 max(0., now-completed.input_timestamp)),
            filter_seconds=filter_stats.get('elapsed_seconds'), compute_seconds=compute_seconds,
            transport_seconds=(None if compute_seconds is None else
                               max(0., result.duration-compute_seconds)))
        self.diagnostics = completed.diagnostics
        if fresh:
            self._published = completed.snapshot
            self._status, self._reason, self._last_error = 'ok', None, None
            self._counts['published'] += 1
        else:
            self._counts['rejected'] += 1
            self._status = 'error' if processing_error else 'no_frame'
            self._reason = status
            if processing_error:
                self._last_error = (completed.snapshot.health.get('last_error') or
                                    completed.diagnostics.get('reason') or 'Mapping processing failed')
                self._counts['processing_failures'] += 1
                self._counts['worker_failures'] += 1
                self._counts['failures'] += 1
            completed = replace(completed, accepted=False)
        self.last_result = completed
        return completed

    def note_source_failure(self, error):
        if self._closed:
            return
        self._barrier += 1
        self._worker.invalidate()
        self._status, self._reason, self._last_error = 'error', 'acquisition_failure', str(error)
        self._counts['failures'] += 1
        self._counts['source_failures'] += 1

    def note_no_frame(self, reason='no_new_frame'):
        if not self._closed:
            self._status, self._reason = 'no_frame', str(reason)

    def reset(self, generation=None):
        if self._closed:
            return
        next_generation = self.generation+1 if generation is None else generation
        if (type(next_generation) is not int or next_generation <= self.generation):
            raise ValueError('Reset requires a strictly newer integer generation')
        self.generation = next_generation
        self._barrier += 1
        self._worker.invalidate()
        self._published, self.last_result = None, None
        self.diagnostics = MappingProxyType({})
        self._completion = {}
        self._status, self._reason, self._last_error = 'waiting', 'reset', None
        self._counts['resets'] += 1

    def snapshot(self, now=None):
        base = self._published if self._published is not None else self._empty
        if base.generation != self.generation:
            base = replace(base, generation=self.generation)
        return MappingReadView(base, now=self.clock() if now is None else now,
            evidence_ttl=self.settings.evidence_ttl, stale_after=self.settings.stale_after,
            health=dict(status=self._status, reason=self._reason, last_error=self._last_error))

    @property
    def health(self):
        values = dict(self.snapshot().health)
        values.update(self._counts, generation=self.generation,
                      closed=self._closed, worker=self._worker.health)
        # Completion metadata must never replace the published source timestamp
        # or authoritative error/closed/stale status.
        values.update({key: value for key, value in self._completion.items()
                       if key not in ('timestamp', 'status')})
        values['completion_status'] = self._completion.get('status')
        values['last'] = dict(self._completion)
        return values

    def close(self):
        if not self._closed:
            self._closed = True
            self._barrier += 1
            self._worker.close()
            self._status, self._reason = 'closed', 'closed'
