from __future__ import annotations

import time
import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.provider import ObservationBatch
from robot_skill_stack.world.perception.diagnostics import describe
from robot_skill_stack.world.perception.frame import PerceptionFrame
from robot_skill_stack.world.perception.geometry import ObjectGeometryService


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
        self.diagnostics = []

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

    def observe(self, context=None) -> ObservationBatch:
        timestamp = time.monotonic()
        token = self.camera.get_frame_token()
        if token is None or token == self._last_token:
            return ObservationBatch()
        self._last_token = token
        depth = self.camera.get_depth()
        if depth is None:
            return ObservationBatch()
        depth = np.squeeze(np.asarray(depth))
        if depth.ndim != 2 or not depth.size:
            return ObservationBatch()
        rgb = self.camera.get_rgb()
        if rgb is None:
            rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
        self._frame_id += 1
        frame = PerceptionFrame(self._frame_id, np.asarray(rgb), depth.copy(), self.localizer.K,
                                self.camera.get_world_from_camera_transform(), timestamp)
        candidates = self.discoverer.discover(frame)
        self.diagnostics = []
        for candidate in candidates:
            geometry = None
            if self.primitive_estimator is not None:
                geometry = self.primitive_estimator.estimate(candidate)
                candidate.metadata['geometry'] = geometry
                if 'center' in geometry.metadata:
                    candidate.position = np.asarray(geometry.metadata['center'], float)
                    candidate.size = np.asarray(geometry.metadata['size_world'], float)
                    candidate.spawnable = self.discoverer.is_spawnable(candidate.size)
            self.diagnostics.append(describe(candidate, geometry))
        self.tracker.update(candidates, frame_id=frame.frame_id, timestamp=timestamp, context=context)
        self.geometry.update_frame(frame)
        return ObservationBatch(self._current_observations(), timestamp)

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
