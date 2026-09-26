from __future__ import annotations

import time
import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.perception.frame import PerceptionFrame
from robot_skill_stack.world.perception.geometry import ObjectGeometryService
from robot_skill_stack.world.model.observations import ObjectObservation


class PerceptionStateProvider:
    def __init__(
        self,
        camera,
        localizer,
        discoverer,
        tracker,
        semantic_classifier,
        *,
        source="rgbd_geometry_perception",
    ):
        self.camera = camera
        self.localizer = localizer
        self.discoverer = discoverer
        self.tracker = tracker
        self.semantic_classifier = semantic_classifier
        self.source = source
        self._frame_id = 0
        self.geometry = ObjectGeometryService(localizer, tracker)

    def _current_observations(self):
        observations = []
        for track in self.tracker.tracks():
            observations.append(
                ObjectObservation(
                    object_id=track.object_id,
                    class_name=track.class_name,
                    pose=Pose(
                        track.position.copy(),
                        orientation=None,
                        frame="world",
                    ),
                    size=track.size.copy(),
                    graspable=None,
                    visible=track.visible,
                    confidence=track.confidence,
                    source=self.source,
                    timestamp=track.last_seen,
                    metadata={
                        "class_confidence": track.class_confidence,
                        "class_belief_authoritative": True,
                        "semantic_samples": track.belief.samples,
                        "track_hits": track.hits,
                        "track_misses": track.misses,
                        "mask_pixels": (
                            0
                            if track.mask is None
                            else int(track.mask.sum())
                        ),
                        "association_hint_match": track.matched_by_hint,
                        "association_hint_reason": (
                            track.association_hint_reason
                        ),
                    },
                )
            )
        return observations

    def observe(self, context=None) -> list[ObjectObservation]:
        rgb = self.camera.get_rgb()
        depth = self.camera.get_depth()
        if rgb is None or depth is None:
            return self._current_observations()

        self._frame_id += 1
        timestamp = time.monotonic()
        frame = PerceptionFrame(
            frame_id=self._frame_id,
            rgb=np.asarray(rgb),
            depth=np.squeeze(np.asarray(depth)),
            intrinsics=self.localizer.K,
            world_from_camera=self.camera.get_world_from_camera_transform(),
            timestamp=timestamp,
        )

        candidates = self.discoverer.discover(frame)
        predictions = [
            self.semantic_classifier.classify(candidate, frame.rgb)
            if self.semantic_classifier is not None
            else None
            for candidate in candidates
        ]
        self.tracker.update(
            candidates,
            predictions,
            frame_id=frame.frame_id,
            timestamp=timestamp,
            context=context,
        )
        self.geometry.update_frame(frame)
        return self._current_observations()

    def get_mask(self, object_id: str):
        return self.geometry.get_mask(object_id)

    def get_point_cloud(self, object_id: str):
        return self.geometry.get_point_cloud(object_id)
