"""SDK-free mapping computation and fixed-time worker publications."""
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import time
from types import MappingProxyType, SimpleNamespace

import numpy as np

from robot_skill_stack.runtime.worker import BoundedWorker
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from .geometry import rasterize_ground_truth
from .occupancy import OccupancyMap, OccupancySnapshot, OccupancyState, OccupancyBatch, _readonly, _single
from .rgbd import RgbdMapper
from .robot_filter import RobotSelfFilter
from .obstacles import ObstacleExtractor


def _owned(array: np.ndarray) -> np.ndarray:
    """Copy writable inputs; reuse read-only buffers supplied by acquisition."""
    return _readonly(array) if array.flags.writeable else array


@dataclass(frozen=True)
class MappingInput:
    """Stable camera/robot observations or prepared ground-truth geometry."""
    generation: int = 0
    frames: tuple[DepthFrame, ...] = ()
    robot_snapshots: Mapping[str, object] | None = None
    ground_truth: object = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'frames', tuple(replace(frame, depth=_owned(frame.depth),
            intrinsics=_owned(frame.intrinsics), world_from_camera=_owned(frame.world_from_camera),
            rgb=None) for frame in self.frames))
        if self.robot_snapshots is not None:
            object.__setattr__(self, 'robot_snapshots', MappingProxyType(dict(self.robot_snapshots)))


@dataclass(frozen=True)
class MappingCompleted:
    """Occupancy and optional products computed from the same input."""
    snapshot: OccupancySnapshot
    accepted: bool
    source_frames: tuple[Mapping[str, object], ...] = ()
    obstacles: object = None
    inferred: object = None


class MappingProcessor:
    """Keep mutable numerical components exclusively in the mapping worker."""
    def __init__(self, settings, *, clock: Callable[[], float] = time.monotonic,
                 obstacle_settings=None, reconstruction_scene=None) -> None:
        self.settings, self.clock = settings, clock
        self.map = self.mapper = self.filter = None
        self._obstacle_settings, self._extractor = obstacle_settings, None
        self._reconstruction_scene = (None if reconstruction_scene is None else SimpleNamespace(
            workspace_min=reconstruction_scene.workspace_min, workspace_max=reconstruction_scene.workspace_max,
            support_plane_z=reconstruction_scene.support_plane_z))

    def _reset(self, generation: int) -> None:
        """Initialize only the components used by the configured source."""
        cfg = self.settings
        self.map = OccupancyMap(cfg.grid_spec(), clock=self.clock, evidence_ttl=cfg.evidence_ttl)
        self.map.generation = generation
        if cfg.source == 'rgbd':
            self.mapper = RgbdMapper(self.map, stride=cfg.stride, max_rays=cfg.max_rays,
                max_voxel_visits=cfg.max_voxel_visits, max_depth=cfg.max_depth,
                max_skew=cfg.max_skew, max_frame_age=cfg.max_frame_age, clock=self.clock)
            if cfg.automatic_robot_filter:
                self.filter = RobotSelfFilter(depth_tolerance=cfg.robot_depth_tolerance,
                                              pose_tolerance=cfg.robot_pose_tolerance)
        self._extractor = (None if self._obstacle_settings is None else
                           ObstacleExtractor(self._obstacle_settings, temporal_probabilities=cfg.source == 'rgbd'))

    def compute(self, payload: MappingInput) -> MappingCompleted:
        """Process observations and prepare optional geometry before publication."""
        if self.map is None or payload.generation != self.map.generation:
            self._reset(payload.generation)
        if self.settings.source == 'ground_truth':
            indices, _ = rasterize_ground_truth(self.map.spec, payload.ground_truth,
                                               self.settings.max_candidate_voxels)
            accepted = self.map.replace(indices, timestamp=payload.ground_truth.timestamp, now=self.clock())
        else:
            if payload.ground_truth is not None:
                raise ValueError('RGB-D mapping cannot consume GT geometry')
            provider = None
            if self.settings.automatic_robot_filter:
                def provider(frames, pixels):
                    return self.filter.masks(frames, payload.robot_snapshots, pixels_by_camera=pixels)
            accepted = self.mapper.update(payload.frames, mask_provider=provider)
        snapshot = self.map.snapshot(now=self.clock())
        obstacles = inferred = None
        if accepted and self._extractor is not None:
            obstacles = self._extractor.extract(snapshot, now=self.clock(), evidence_ttl=self.settings.evidence_ttl)
        if accepted and self._reconstruction_scene is not None:
            from .reconstruction import infer
            inferred = infer(payload, self._reconstruction_scene)
        frames = tuple(MappingProxyType(dict(camera_id=f.camera_id, token=f.token, acquired_at=f.acquired_at,
            depth_shape=f.depth.shape, intrinsics=f.intrinsics, world_from_camera=f.world_from_camera))
            for f in payload.frames) if accepted else ()
        return MappingCompleted(snapshot, accepted, frames, obstacles, inferred)


@dataclass(frozen=True, eq=False)
class MappingReadView:
    """Sparse occupancy queries evaluated at one fixed time."""
    _snapshot: OccupancySnapshot
    evaluated_at: float
    evidence_ttl: float
    stale_after: float
    status: str = 'ok'
    reason: str | None = None
    source_frames: tuple[Mapping[str, object], ...] = ()
    obstacles: object = None
    inferred: object = None

    spec = property(lambda self: self._snapshot.spec)
    timestamp = property(lambda self: self._snapshot.timestamp)
    revision = property(lambda self: self._snapshot.revision)
    generation = property(lambda self: self._snapshot.generation)

    @property
    def health(self) -> dict[str, object]:
        age = None if self.timestamp is None else self.evaluated_at-self.timestamp
        fresh = self.status == 'ok' and age is not None and 0 <= age < self.stale_after
        status = 'stale' if self.status == 'ok' and not fresh else self.status
        return dict(status=status, fresh=fresh, reason=self.reason)

    def query_many(self, positions: np.ndarray | Sequence) -> OccupancyBatch:
        base = self._snapshot.query_many(positions)
        expired = base.expired | (base.observed & (self.evaluated_at-base.last_observed >= self.evidence_ttl))
        states = np.where(expired, OccupancyState.UNKNOWN, base.states)
        probabilities = np.where(expired, .5, base.probabilities)
        return OccupancyBatch(_readonly(states), _readonly(probabilities), base.observed,
                              base.last_observed, _readonly(expired))

    def query(self, position: np.ndarray | Sequence):
        return _single(self.query_many(position))


class MappingService:
    """Submit observations and publish fresh results from the current generation."""
    def __init__(self, settings, *, worker_factory: Callable = BoundedWorker,
                 clock: Callable[[], float] = time.monotonic,
                 obstacle_settings=None, reconstruction_scene=None) -> None:
        self.settings, self.clock = settings, clock
        self._processor = MappingProcessor(settings, clock=clock, obstacle_settings=obstacle_settings,
                                          reconstruction_scene=reconstruction_scene)
        self._worker = worker_factory(self._processor.compute, name='spatial-mapping')
        self._empty = OccupancyMap(settings.grid_spec(), clock=clock,
            evidence_ttl=settings.evidence_ttl).snapshot(now=clock())
        self._published = None
        self.generation, self._barrier = 0, 0
        self._status, self._reason = 'waiting', None
        self._published_count, self._failures = 0, 0

    def submit(self, payload: MappingInput) -> bool:
        if self._status == 'closed' or payload.generation != self.generation:
            return False
        return self._worker.submit(payload, tag=(self.generation, self._barrier))

    def poll(self, now: float | None = None) -> MappingCompleted | None:
        """Ignore invalidated work; publish only a fresh successful computation."""
        if self._status == 'closed':
            return None
        work = self._worker.poll()
        if work is None or work.tag != (self.generation, self._barrier):
            return None
        if work.error is not None:
            self.note_source_failure(work.error)
            return None
        result = work.value
        if not result.accepted:
            if result.snapshot.error is not None:
                self.note_source_failure(result.snapshot.error)
            else:
                self.note_no_frame()
            return result
        now = self.clock() if now is None else now
        limit = self.settings.stale_after
        if self.settings.source == 'rgbd':
            limit = min(limit, self.settings.max_frame_age)
        if not 0 <= now-result.snapshot.timestamp < limit:
            self._status, self._reason = 'stale', 'stale_completion'
            return replace(result, accepted=False)
        self._published = result
        self._published_count += 1
        self._status, self._reason = 'ok', None
        return result

    def note_source_failure(self, error: Exception | str) -> None:
        if self._status != 'closed':
            self._barrier += 1
            self._worker.invalidate()
            self._failures += 1
            self._status, self._reason = 'error', str(error)

    def note_no_frame(self, reason: str = 'no_new_frame') -> None:
        if self._status != 'closed':
            self._status, self._reason = 'no_frame', reason

    def reset(self) -> None:
        if self._status != 'closed':
            self.generation += 1
            self._barrier += 1
            self._worker.invalidate()
            self._published = None
            self._empty = replace(self._empty, generation=self.generation)
            self._status, self._reason = 'waiting', None

    def snapshot(self, now: float | None = None) -> MappingReadView:
        result = self._published
        return MappingReadView(result.snapshot if result is not None else self._empty,
            self.clock() if now is None else now, self.settings.evidence_ttl, self.settings.stale_after,
            self._status, self._reason, () if result is None else result.source_frames,
            None if result is None else result.obstacles, None if result is None else result.inferred)

    @property
    def health(self) -> dict[str, object]:
        return dict(self.snapshot().health, published=self._published_count, failures=self._failures,
                    closed=self._status == 'closed')

    def close(self) -> None:
        if self._status != 'closed':
            self._worker.close()
            self._status, self._reason = 'closed', None
