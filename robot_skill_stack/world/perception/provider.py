from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.provider import ObservationBatch
from robot_skill_stack.world.perception.diagnostics import describe
from robot_skill_stack.world.perception.frame import PerceptionFrame
from robot_skill_stack.world.perception.geometry import ObjectGeometryService


def _immutable_array(value):
    array = np.ascontiguousarray(value)
    base = array
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    if isinstance(base, bytes) and not array.flags.writeable:
        return array
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class PerceptionInput:
    frame: PerceptionFrame
    acquisition_ms: float

    @property
    def timestamp(self):
        return self.frame.timestamp


@dataclass(frozen=True)
class PerceptionResult:
    frame: PerceptionFrame
    candidates: tuple
    diagnostics: tuple
    discovery_diagnostics: tuple
    processing_ms: float
    acquisition_ms: float

    @property
    def timestamp(self):
        return self.frame.timestamp


class PerceptionProcessor:
    """SDK-free discovery and geometry; no camera, tracker or WorldModel."""
    def __init__(self, discoverer, primitive_estimator):
        self.discoverer, self.primitive_estimator = discoverer, primitive_estimator

    def compute(self, prepared):
        frame = prepared.frame
        started = time.perf_counter()
        candidates = self.discoverer.discover(frame)
        diagnostics = []
        for candidate in candidates:
            geometry = None
            if self.primitive_estimator is not None:
                geometry = self.primitive_estimator.estimate(candidate)
                candidate.metadata['geometry'] = geometry
                if 'center' in geometry.metadata:
                    candidate.position = np.asarray(geometry.metadata['center'], float)
                    candidate.size = np.asarray(geometry.metadata['size_world'], float)
                    candidate.spawnable = self.discoverer.is_spawnable(candidate.size)
            diagnostics.append(describe(candidate, geometry))
        return PerceptionResult(frame, tuple(candidates), tuple(diagnostics),
            tuple(deepcopy(self.discoverer.diagnostics)), (time.perf_counter()-started)*1000,
            prepared.acquisition_ms)


class PerceptionStateProvider:
    def __init__(self, camera, localizer, discoverer, tracker,
                 primitive_estimator=None, *, source='rgbd_geometry_perception'):
        self.camera = camera
        self.localizer = localizer
        self.discoverer = discoverer
        self.tracker = tracker
        self.primitive_estimator = primitive_estimator
        self.source = source
        self._closed = False
        self._frame_id = 0
        self._last_token = None
        self.geometry = ObjectGeometryService(localizer, tracker)
        self.processor = PerceptionProcessor(discoverer, primitive_estimator)
        self.diagnostics = []
        self.published_discovery_diagnostics = []
        self.stage_diagnostics = {}

    def _current_observations(self):
        observations = []
        for track in self.tracker.tracks():
            if not track.confirmed or not track.visible:
                continue
            geometry = track.geometry
            shape = None if geometry is None or geometry.shape.value == 'unknown' else geometry.shape.value
            yaw = None if geometry is None else geometry.yaw
            orientation = None if yaw is None else np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
            observations.append(ObjectObservation(
                object_id=track.object_id, class_name=shape,
                pose=Pose(track.position.copy(), orientation, frame='world'),
                size=track.size.copy(), geometry=geometry, graspable=None,
                confidence=track.confidence, source=self.source, timestamp=track.last_seen,
                metadata={
                    'class_confidence': 0. if geometry is None else geometry.confidence,
                    'mask_pixels': 0 if track.mask is None else int(track.mask.sum()),
                    'association_hint_match': track.matched_by_hint,
                    'association_hint_reason': track.association_hint_reason,
                },
            ))
        return observations

    def acquire(self, frames=None):
        """Own camera reads; a shared input uses exactly its original timestamp."""
        if self._closed:
            return None
        started = time.perf_counter()
        shared = None
        if frames is not None:
            if hasattr(frames, 'camera_id'):
                shared = frames
            else:
                frames = tuple(frames)
                if len(frames) != 1:
                    raise ValueError('V1 perception requires exactly one shared camera frame')
                shared = frames[0]
        timestamp = time.monotonic() if shared is None else shared.acquired_at
        token = self.camera.get_frame_token() if shared is None else tuple(shared.token)
        if token is None or token == self._last_token:
            return None
        self._last_token = token
        depth = self.camera.get_depth() if shared is None else shared.depth
        if depth is None:
            return None
        depth = np.squeeze(np.asarray(depth))
        if depth.ndim != 2 or not depth.size:
            return None
        rgb = self.camera.get_rgb() if shared is None else shared.rgb
        if rgb is None:
            rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
        self._frame_id += 1
        transform = self.camera.get_world_from_camera_transform() if shared is None else shared.world_from_camera
        frame = PerceptionFrame(self._frame_id, _immutable_array(rgb), _immutable_array(depth),
                                _immutable_array(self.localizer.K), _immutable_array(transform), timestamp)
        return PerceptionInput(frame, (time.perf_counter()-started)*1000)

    def associate(self, result, context=None):
        if self._closed or result is None:
            return ObservationBatch()
        started = time.perf_counter()
        frame = result.frame
        self.tracker.update(result.candidates, frame_id=frame.frame_id, timestamp=frame.timestamp, context=context)
        self.geometry.update_frame(frame)
        self.diagnostics = deepcopy(list(result.diagnostics))
        self.published_discovery_diagnostics = deepcopy(list(result.discovery_diagnostics))
        self.stage_diagnostics = dict(acquisition_ms=result.acquisition_ms,
            processing_ms=result.processing_ms, tracking_ms=(time.perf_counter()-started)*1000)
        return ObservationBatch(self._current_observations(), frame.timestamp)

    def observe(self, context=None) -> ObservationBatch:
        prepared = self.acquire()
        if prepared is None:
            return ObservationBatch()
        return self.associate(self.processor.compute(prepared), context)

    def forget(self, object_id: str):
        return self.tracker.forget(object_id)

    def get_mask(self, object_id: str):
        return self.geometry.get_mask(object_id)

    def get_point_cloud(self, object_id: str):
        return self.geometry.get_point_cloud(object_id)

    def close(self):
        if not self._closed:
            self._closed = True
            self.camera.close()
