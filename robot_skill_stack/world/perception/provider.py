from __future__ import annotations

import time
import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.perception.diagnostics import describe
from robot_skill_stack.world.perception.frame import PerceptionFrame
from robot_skill_stack.world.perception.geometry import ObjectGeometryService
from robot_skill_stack.world.perception.semantics import SemanticPrediction


class PerceptionStateProvider:
    def __init__(self, camera, localizer, discoverer, tracker, semantic_classifier=None,
                 primitive_estimator=None, *, source='rgbd_geometry_perception', stale_frame_timeout=1.0):
        self.camera = camera
        self.localizer = localizer
        self.discoverer = discoverer
        self.tracker = tracker
        self.semantic_classifier = semantic_classifier
        self.primitive_estimator = primitive_estimator
        self.source = source
        self.stale_frame_timeout = float(stale_frame_timeout)
        self._frame_id = 0
        self._last_token = None
        self.geometry = ObjectGeometryService(localizer, tracker)
        self.diagnostics = []

    def _current_observations(self):
        observations = []
        for track in self.tracker.tracks():
            if not track.confirmed:
                continue
            geometry = track.geometry
            shape = None if geometry is None or geometry.shape.value == 'unknown' else geometry.shape.value
            label = track.class_name if self.semantic_classifier is not None else shape
            yaw = None if geometry is None else geometry.yaw
            orientation = None if yaw is None else np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
            observations.append(ObjectObservation(
                object_id=track.object_id, class_name=label,
                pose=Pose(track.position.copy(), orientation, frame='world'),
                size=track.size.copy(), geometry=geometry, graspable=None,
                visible=track.visible, confidence=track.confidence, source=self.source, timestamp=track.last_seen,
                metadata={
                    'class_confidence': track.class_confidence if self.semantic_classifier is not None else
                                        (0. if geometry is None else geometry.confidence),
                    'class_belief_authoritative': True, 'semantic_samples': track.belief.samples,
                    'track_hits': track.hits, 'track_misses': track.misses,
                    'mask_pixels': 0 if track.mask is None else int(track.mask.sum()),
                    'association_hint_match': track.matched_by_hint,
                    'association_hint_reason': track.association_hint_reason,
                },
            ))
        return observations

    def age(self, now=None, context=None):
        now = time.monotonic() if now is None else float(now)
        return self.tracker.age(now, self.stale_frame_timeout, context)

    def observe(self, context=None) -> list[ObjectObservation]:
        timestamp = time.monotonic()
        get_token = getattr(self.camera, 'get_frame_token', None)
        token = get_token() if get_token is not None else None
        depth = self.camera.get_depth()
        if depth is None or (token is not None and token == self._last_token):
            self.age(timestamp, context)
            return self._current_observations()
        depth = np.squeeze(np.asarray(depth))
        if depth.ndim != 2 or not depth.size:
            self.age(timestamp, context)
            return self._current_observations()
        rgb = self.camera.get_rgb()
        if rgb is None:
            rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
        self._frame_id += 1
        frame = PerceptionFrame(self._frame_id, np.asarray(rgb), depth.copy(), self.localizer.K,
                                self.camera.get_world_from_camera_transform(), timestamp)
        candidates = self.discoverer.discover(frame)
        self.diagnostics = []
        predictions = []
        for candidate in candidates:
            geometry = None
            if self.primitive_estimator is not None:
                geometry = self.primitive_estimator.estimate(candidate)
                candidate.metadata['geometry'] = geometry
                if 'center' in geometry.metadata:
                    candidate.position = np.asarray(geometry.metadata['center'], float)
                    candidate.size = np.asarray(geometry.metadata['size_world'], float)
                    candidate.spawnable = self.discoverer.is_spawnable(candidate.size)
            if self.semantic_classifier is not None:
                prediction = self.semantic_classifier.classify(candidate, frame.rgb)
            else:
                label = None if geometry is None or geometry.shape.value == 'unknown' else geometry.shape.value
                prediction = SemanticPrediction(label, 1. if label is None else geometry.confidence)
            predictions.append(prediction)
            self.diagnostics.append(describe(candidate, geometry))
        self.tracker.update(candidates, predictions, frame_id=frame.frame_id, timestamp=timestamp, context=context)
        self.geometry.update_frame(frame)
        self._last_token = token
        return self._current_observations()

    def forget(self, object_id: str):
        return self.tracker.forget(object_id)

    def get_mask(self, object_id: str):
        return self.geometry.get_mask(object_id)

    def get_point_cloud(self, object_id: str):
        return self.geometry.get_point_cloud(object_id)
