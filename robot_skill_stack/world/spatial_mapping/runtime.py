"""SDK-free worker-owned mapping and constant-time immutable publications.

Dense processing and optional obstacle extraction run in the worker. Snapshot
construction on the owner thread fixes the read time without a grid scan;
sparse queries age just the requested cells.
"""
from dataclasses import dataclass, replace
import time
from types import MappingProxyType
from types import SimpleNamespace

import numpy as np

from robot_skill_stack.runtime.worker import BoundedWorker
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from .geometry import freeze, rasterize_ground_truth
from .occupancy import (OccupancyMap, OccupancyState, OccupancyBatch, _readonly, _single)
from .rgbd import RgbdMapper
from .robot_filter import RobotSelfFilter
from .obstacles import ObstacleExtractor


def _owned(array):
    """Reuse genuinely immutable buffers from the camera pool, otherwise copy."""
    value = np.asarray(array)
    base = value
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    return value if isinstance(base, bytes) and not value.flags.writeable else _readonly(value)


@dataclass(frozen=True)
class MappingInput:
    """Immutable input containing RGB-D frames or prepared ground-truth geometry."""
    generation: int = 0                 # In case of a restart, reject old mapping generation
    frames: tuple[DepthFrame, ...] = ()
    robot_snapshots: object = None
    exclusion_masks: object = None
    ground_truth: object = None         # Used when we are using Isaac ground truth
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


def _input_timestamp(payload: MappingInput) -> float | None:
    if payload.ground_truth is not None:
        return payload.ground_truth.timestamp
    try:
        stamp = min((float(frame.acquired_at) for frame in payload.frames), default=None)
        return stamp if stamp is not None and np.isfinite(stamp) else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class MappingCompleted:
    """Worker result containing an occupancy snapshot and optional map products."""
    snapshot: object
    diagnostics: object
    accepted: bool
    generation: int
    input_timestamp: float | None = None
    obstacles: object = None
    inferred: object = None


class MappingProcessor:
    """One mutable map, touched only by the single computation worker."""
    def __init__(self, settings, *, clock=time.monotonic, obstacle_settings=None, reconstruction_scene=None):
        self.settings, self.clock = settings, clock
        self.map = self.mapper = self.filter = None
        self.generation = None
        self._obstacle_settings, self._extractor = obstacle_settings, None
        self._reconstruction_scene = (None if reconstruction_scene is None else SimpleNamespace(
            **{k:getattr(reconstruction_scene,k) for k in ('workspace_min','workspace_max','support_plane_z')}))

    def _reset(self, generation):
        """Create fresh mapping components for a new simulation generation."""
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
        self._extractor = (None if self._obstacle_settings is None else
                           ObstacleExtractor(self._obstacle_settings,
                               temporal_probabilities=cfg.source == "rgbd"))

    def compute(self, payload: MappingInput) -> MappingCompleted:
        """Update the occupancy grid from camera rays or ground-truth geometry.

        Prepare optional obstacle and reconstruction products in the worker.
        """
        mapping_input = payload
        if self.map is None or mapping_input.generation != self.generation or mapping_input.reset:
            self._reset(mapping_input.generation)
        started = time.perf_counter()
        if self.settings.source == 'ground_truth':
            if mapping_input.ground_truth is None:
                raise ValueError('GT mapping requires prepared geometry')
            indices, diagnostics = rasterize_ground_truth(self.map.spec, mapping_input.ground_truth,
                                                         self.settings.max_candidate_voxels)
            accepted = self.map.replace(indices, timestamp=mapping_input.ground_truth.timestamp,
                                        now=self.clock())
        else:
            if mapping_input.ground_truth is not None:
                raise ValueError('RGB-D mapping cannot consume GT geometry')
            provider = None
            if self.settings.automatic_robot_filter:
                if mapping_input.robot_snapshots is None:
                    raise ValueError('Automatic filtering requires exposure-matched robot snapshots')
                def provider(frames, pixels):
                    return self.filter.masks(frames, mapping_input.robot_snapshots, pixels_by_camera=pixels)
            accepted = self.mapper.update(mapping_input.frames, exclusion_masks=mapping_input.exclusion_masks,
                                          mask_provider=provider)
            diagnostics = dict(self.mapper.diagnostics)
            if provider is not None:
                diagnostics['robot_filter'] = dict(self.filter.diagnostics)
            if accepted:
                # Preserve the original accepted exposure identity alongside
                # the worker publication. These small calibration records do
                # not copy image buffers or add another acquisition pipeline.
                diagnostics.update(source_type='rgbd', accepted_frame_count=len(mapping_input.frames),
                    source_frames=tuple(dict(camera_id=frame.camera_id, token=tuple(frame.token),
                        acquired_at=frame.acquired_at, depth_shape=tuple(frame.depth.shape),
                        intrinsics=frame.intrinsics, world_from_camera=frame.world_from_camera)
                        for frame in mapping_input.frames))
        occupancy_snapshot = self.map.snapshot(now=self.clock())
        obstacles = (None if self._extractor is None else self._extractor.extract(
            occupancy_snapshot, now=self.clock(), evidence_ttl=self.settings.evidence_ttl))
        if obstacles is not None:
            diagnostics['obstacle_extraction'] = dict(obstacles.diagnostics)
        diagnostics.update(accepted=accepted, compute_seconds=time.perf_counter()-started)
        inferred = None
        if accepted and self._reconstruction_scene is not None:
            from .reconstruction import infer
            inferred = infer(mapping_input, self._reconstruction_scene, occupancy_snapshot)
            diagnostics['reconstruction'] = dict(status=inferred.status, reason=inferred.reason,
                elapsed_seconds=inferred.diagnostics['elapsed_seconds'])
            diagnostics['mapping_compute_seconds'] = diagnostics['compute_seconds']
            diagnostics['compute_seconds'] = time.perf_counter()-started
        return MappingCompleted(occupancy_snapshot, freeze(diagnostics), accepted, mapping_input.generation,
                                _input_timestamp(mapping_input), obstacles, inferred)


@dataclass(frozen=True, init=False, eq=False)
class MappingReadView:
    """Immutable fixed-time facade over an already immutable worker snapshot."""
    def __init__(self, snapshot, *, now: float, evidence_ttl: float,
                 stale_after: float, health=None, obstacles=None, inferred=None):
        now = float(now)
        if not np.isfinite(now) or now < 0:
            raise ValueError('Read time must be finite and nonnegative')
        occupancy_snapshot = snapshot
        for key, value in dict(_snapshot=occupancy_snapshot, evaluated_at=now,
                evidence_ttl=evidence_ttl, stale_after=stale_after, spec=occupancy_snapshot.spec,
                timestamp=occupancy_snapshot.timestamp, revision=occupancy_snapshot.revision,
                generation=occupancy_snapshot.generation, obstacles=obstacles, inferred=inferred).items():
            object.__setattr__(self, key, value)
        values = dict(occupancy_snapshot.health if health is None else health)
        age = None if self.timestamp is None else max(0., now-self.timestamp)
        stale = age is not None and age > stale_after
        status = values.get('status', 'waiting')
        if stale and status == 'ok':
            status = 'stale'
        values.update(status=status, age_seconds=age, stale=stale,
                      fresh=status == 'ok' and not stale, timestamp=self.timestamp)
        object.__setattr__(self, 'health', MappingProxyType(values))

    def _age(self, states, probabilities, observed, last_observed, expired):
        """Mark expired evidence unknown at the view's fixed evaluation time."""
        expired = expired | (observed & (self.evaluated_at-last_observed >= self.evidence_ttl))
        states, probabilities = states.copy(), probabilities.copy()
        states[expired], probabilities[expired] = OccupancyState.UNKNOWN, .5
        return states, probabilities, expired

    def query_many(self, positions) -> OccupancyBatch:
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
    def __init__(self, settings, *, worker_factory=BoundedWorker, clock=time.monotonic,
                 obstacle_settings=None, reconstruction_scene=None):
        self.settings, self.clock = settings, clock
        self._processor = MappingProcessor(settings, clock=clock, obstacle_settings=obstacle_settings,
                                          reconstruction_scene=reconstruction_scene)
        self._worker = worker_factory(self._processor.compute, name='spatial-mapping')
        # Allocate the empty publication once at construction, never on reads.
        self._empty = OccupancyMap(settings.grid_spec(), clock=clock,
            stale_after=settings.stale_after, evidence_ttl=settings.evidence_ttl).snapshot(now=clock())
        self._published = None
        self._published_obstacles = None
        self._published_inferred = None
        self.generation, self._barrier, self._closed = 0, 0, False
        self._status, self._reason, self._last_error = 'waiting', None, None
        self.last_result, self.diagnostics = None, MappingProxyType({})
        self._counts = dict(submitted=0, published=0, rejected=0, failures=0, source_failures=0, worker_failures=0, processing_failures=0, resets=0)
        self._completion = {}

    def submit(self, payload: MappingInput) -> bool:
        """Queue mapping work only for the current simulation generation."""
        if self._closed or payload.generation != self.generation:
            self._counts['rejected'] += 1
            return False
        accepted = self._worker.submit(payload, tag=(self.generation, self._barrier, _input_timestamp(payload)))
        self._counts['submitted'] += int(accepted)
        return accepted

    def poll(self, now: float | None = None) -> MappingCompleted | None:
        """Accept a fresh worker result and publish its immutable map products."""
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
        mapping_result = result.value
        if mapping_result.generation != self.generation:
            self._counts['rejected'] += 1
            return None
        snapshot_timestamp = mapping_result.snapshot.timestamp
        age = None if snapshot_timestamp is None else max(0., now-snapshot_timestamp)
        limit = self.settings.stale_after
        if self.settings.source == 'rgbd':
            limit = min(limit, self.settings.max_frame_age)
        fresh = (mapping_result.accepted and snapshot_timestamp is not None and snapshot_timestamp <= now+1e-9
                 and age < limit and mapping_result.snapshot.health['status'] == 'ok')
        processing_error = not mapping_result.accepted and mapping_result.snapshot.health['status'] == 'error'
        status = ('published' if fresh else 'processing_failure' if processing_error else
                  'stale_completion' if mapping_result.accepted else mapping_result.snapshot.health['status'])
        filter_stats = mapping_result.diagnostics.get('robot_filter', {})
        compute_seconds = mapping_result.diagnostics.get('compute_seconds')
        self._completion = dict(timing, timestamp=snapshot_timestamp,
            input_timestamp=mapping_result.input_timestamp, status=status,
            publish_age_seconds=(None if mapping_result.input_timestamp is None else
                                 max(0., now-mapping_result.input_timestamp)),
            filter_seconds=filter_stats.get('elapsed_seconds'), compute_seconds=compute_seconds,
            transport_seconds=(None if compute_seconds is None else
                               max(0., result.duration-compute_seconds)))
        self.diagnostics = mapping_result.diagnostics
        if fresh:
            self._published = mapping_result.snapshot
            self._published_obstacles = mapping_result.obstacles
            self._published_inferred = mapping_result.inferred
            self._status, self._reason, self._last_error = 'ok', None, None
            self._counts['published'] += 1
        else:
            self._counts['rejected'] += 1
            self._status = 'error' if processing_error else 'no_frame'
            self._reason = status
            if processing_error:
                self._last_error = (mapping_result.snapshot.health.get('last_error') or
                                    mapping_result.diagnostics.get('reason') or 'Mapping processing failed')
                self._counts['processing_failures'] += 1
                self._counts['worker_failures'] += 1
                self._counts['failures'] += 1
            mapping_result = replace(mapping_result, accepted=False)
        self.last_result = mapping_result
        return mapping_result

    def note_source_failure(self, error):
        """Invalidate pending work after an acquisition or source failure."""
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
        """Start a new generation and discard previously published map evidence."""
        if self._closed:
            return
        next_generation = self.generation+1 if generation is None else generation
        if (type(next_generation) is not int or next_generation <= self.generation):
            raise ValueError('Reset requires a strictly newer integer generation')
        self.generation = next_generation
        self._barrier += 1
        self._worker.invalidate()
        self._published, self.last_result = None, None
        self._published_obstacles = None
        self._published_inferred = None
        self.diagnostics = MappingProxyType({})
        self._completion = {}
        self._status, self._reason, self._last_error = 'waiting', 'reset', None
        self._counts['resets'] += 1

    def snapshot(self, now: float | None = None) -> MappingReadView:
        """Return a fixed-time read view of the latest accepted map publication."""
        occupancy_snapshot = self._published if self._published is not None else self._empty
        if occupancy_snapshot.generation != self.generation:
            occupancy_snapshot = replace(occupancy_snapshot, generation=self.generation)
        return MappingReadView(occupancy_snapshot, now=self.clock() if now is None else now,
            evidence_ttl=self.settings.evidence_ttl, stale_after=self.settings.stale_after,
            health=dict(status=self._status, reason=self._reason, last_error=self._last_error),
            obstacles=self._published_obstacles, inferred=self._published_inferred)

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
