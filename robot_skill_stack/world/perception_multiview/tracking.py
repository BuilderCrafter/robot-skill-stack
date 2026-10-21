"""Independent multi-view identities; expiration remains owned by the updater."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    fit: object
    points: np.ndarray
    views: tuple[str, ...]


@dataclass
class Track:
    object_id: str
    detection: Detection
    last_seen: float
    hits: int = 1
    visible: bool = True
    hint_reason: str | None = None


class MultiViewTracker:
    def __init__(self, *, confirm_hits=3, max_distance=.12):
        self.confirm_hits, self.max_distance = confirm_hits, max_distance
        self.tracks = {}
        self._next_id = 1

    def update(self, detections, timestamp, context=None):
        hints = {} if context is None else {h.object_id: h for h in context.association_hints
                                            if h.expires_at > timestamp}
        held = None if context is None else context.held_object_id
        tracks = tuple(self.tracks.values())
        edges = []
        suppressed = set()
        for ti, track in enumerate(tracks):
            old = track.detection.fit
            hint = hints.get(track.object_id)
            for di, detection in enumerate(detections):
                fit = detection.fit
                distance = float(np.linalg.norm(fit.center-old.center))
                if track.object_id == held and hint is None:
                    # Do not steal an attached/uncertain identity for tabletop clutter.
                    if distance < max(.025, np.linalg.norm(old.size)/2):
                        suppressed.add(di)
                    continue
                change = np.linalg.norm(fit.size-old.size)/max(np.linalg.norm(old.size), .01)
                if change > 1.5:
                    continue
                reason, limit = None, self.max_distance
                if hint is not None:
                    expected = float(np.linalg.norm(fit.center-hint.expected_position))
                    if expected <= hint.max_distance:
                        distance, limit, reason = expected, hint.max_distance, hint.reason
                if distance <= limit:
                    edges.append((reason is None, distance/limit+.2*change, ti, di, reason))
        used_tracks, used_detections = set(), set()
        for _, _, ti, di, reason in sorted(edges):
            if ti in used_tracks or di in used_detections or di in suppressed:
                continue
            track = tracks[ti]
            track.detection, track.last_seen, track.hint_reason = detections[di], timestamp, reason
            track.hits += 1
            track.visible = True
            used_tracks.add(ti)
            used_detections.add(di)
        for ti, track in enumerate(tracks):
            if ti not in used_tracks:
                track.visible, track.hint_reason = False, None
                if track.hits < self.confirm_hits and track.object_id != held:
                    self.forget(track.object_id)
        for di, detection in enumerate(detections):
            if di not in used_detections and di not in suppressed:
                key = f'multiview_{self._next_id}'
                self._next_id += 1
                self.tracks[key] = Track(key, detection, timestamp)
        return tuple(t for t in self.tracks.values() if t.visible and t.hits >= self.confirm_hits)

    def forget(self, object_id):
        self.tracks.pop(object_id, None)
