"""Independent multi-view identities; expiration remains owned by the updater."""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from robot_skill_stack.world.model.context import ObservationContext
from .fitting import PrimitiveFit


@dataclass
class Detection:
    """One fitted object cloud and the cameras that contributed to it."""
    fit: PrimitiveFit
    points: np.ndarray
    views: tuple[str, ...]


@dataclass
class Track:
    """Persistent identity and confirmation history for a multi-view detection."""
    object_id: str
    detection: Detection
    hits: int = 1
    visible: bool = True
    matched_by_hint: bool = False


class MultiViewTracker:
    """Greedily match geometry, giving eligible placement hints first priority."""
    def __init__(self, *, confirm_hits: int = 3, max_distance: float = .12):
        self.confirm_hits, self.max_distance = confirm_hits, max_distance
        self.tracks: dict[str, Track] = {}
        self._next_id = 1

    def update(self, detections: Sequence[Detection], timestamp: float,
               context: ObservationContext | None = None) -> tuple[Track, ...]:
        """Associate detections one-to-one and return confirmed visible tracks."""
        hints = {} if context is None else {hint.object_id: hint for hint in context.association_hints
                                            if hint.expires_at > timestamp}
        held = None if context is None else context.held_object_id
        tracks = tuple(self.tracks.values())
        candidates, suppressed = [], set()
        for track_index, track in enumerate(tracks):
            previous_fit = track.detection.fit
            size_norm = np.linalg.norm(previous_fit.size)
            hint = hints.get(track.object_id)
            for detection_index, detection in enumerate(detections):
                fit = detection.fit
                distance = np.linalg.norm(fit.center - previous_fit.center)
                if track.object_id == held and hint is None:
                    # Do not steal an attached/uncertain identity for tabletop clutter.
                    if distance < max(.025, size_norm / 2):
                        suppressed.add(detection_index)
                    continue
                size_change = np.linalg.norm(fit.size - previous_fit.size) / max(size_norm, .01)
                if size_change > 1.5:
                    continue
                matched_by_hint, limit = False, self.max_distance
                if hint is not None:
                    expected_distance = np.linalg.norm(fit.center - hint.expected_position)
                    if expected_distance <= hint.max_distance:
                        distance, limit, matched_by_hint = expected_distance, hint.max_distance, True
                if distance <= limit:
                    score = distance / limit + .2 * size_change
                    candidates.append((not matched_by_hint, score, track_index, detection_index))
        used_tracks, used_detections = set(), set()
        # Ordinary matches follow hint matches; indices break geometric ties.
        for ordinary_match, _, track_index, detection_index in sorted(candidates):
            if track_index in used_tracks or detection_index in used_detections or detection_index in suppressed:
                continue
            track = tracks[track_index]
            track.detection, track.matched_by_hint = detections[detection_index], not ordinary_match
            track.hits += 1
            track.visible = True
            used_tracks.add(track_index)
            used_detections.add(detection_index)
        for track_index, track in enumerate(tracks):
            if track_index not in used_tracks:
                track.visible, track.matched_by_hint = False, False
                if track.hits < self.confirm_hits and track.object_id != held:
                    self.forget(track.object_id)
        for detection_index, detection in enumerate(detections):
            if detection_index not in used_detections and detection_index not in suppressed:
                key = f'multiview_{self._next_id}'
                self._next_id += 1
                self.tracks[key] = Track(key, detection)
        return tuple(track for track in self.tracks.values() if track.visible and track.hits >= self.confirm_hits)

    def forget(self, object_id: str) -> None:
        self.tracks.pop(object_id, None)
