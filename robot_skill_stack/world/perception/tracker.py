from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ObjectTrack:
    object_id: str
    position: np.ndarray
    size: np.ndarray
    confidence: float
    last_seen: float
    frame_id: int | None
    mask: np.ndarray | None
    geometry: object | None = None
    hits: int = 1
    confirmed: bool = False
    association_hint_reason: str | None = None

    @property
    def visible(self):
        return self.frame_id is not None

    @property
    def matched_by_hint(self):
        return self.association_hint_reason is not None


class ObjectTracker:
    """One greedy distance/size assignment; the updater owns confirmed-ID expiry.

    Only complete (spawnable) detections refresh tracks. Pending detections must
    match consecutive fresh frames; missing pending tracks are discarded.
    """
    def __init__(self, *, max_distance=0.18, max_size_ratio=1.5, min_confirm_hits=3):
        self.max_distance = float(max_distance)
        self.max_size_ratio = float(max_size_ratio)
        self.min_confirm_hits = int(min_confirm_hits)
        if self.max_distance <= 0 or self.max_size_ratio < 0 or self.min_confirm_hits < 1:
            raise ValueError('Invalid track association/confirmation settings')
        self._tracks: dict[str, ObjectTrack] = {}
        self._next_id = 1

    def update(self, candidates, *, frame_id, timestamp, context=None):
        candidates = [c for c in candidates if c.spawnable]
        tracks = self.tracks()
        hints = {} if context is None else {
            h.object_id: h for h in context.association_hints if h.expires_at > timestamp
        }
        pairs = []
        for ti, track in enumerate(tracks):
            hint = hints.get(track.object_id)
            for ci, candidate in enumerate(candidates):
                size_ratio = float(np.linalg.norm(candidate.size-track.size) /
                                   max(np.linalg.norm(track.size), 1e-6))
                if size_ratio > self.max_size_ratio:
                    continue
                distance = float(np.linalg.norm(candidate.position-track.position))
                limit, reason = self.max_distance, None
                if hint is not None:
                    expected_distance = float(np.linalg.norm(candidate.position-hint.expected_position))
                    if expected_distance <= hint.max_distance:
                        distance, limit, reason = expected_distance, hint.max_distance, hint.reason
                if distance <= limit:
                    score = distance/limit + .25*min(size_ratio, 2.)
                    pairs.append((reason is None, score, ti, ci, reason))

        matched_tracks, matched_candidates = set(), set()
        for _, _, ti, ci, reason in sorted(pairs):
            if ti in matched_tracks or ci in matched_candidates:
                continue
            track, candidate = tracks[ti], candidates[ci]
            track.position, track.size = candidate.position.copy(), candidate.size.copy()
            track.confidence, track.last_seen = candidate.confidence, timestamp
            track.frame_id, track.mask = frame_id, candidate.mask.copy()
            track.geometry = candidate.metadata.get('geometry')
            track.hits += 1
            track.confirmed = track.hits >= self.min_confirm_hits
            track.association_hint_reason = reason
            matched_tracks.add(ti)
            matched_candidates.add(ci)

        for ti, track in enumerate(tracks):
            if ti not in matched_tracks:
                if not track.confirmed:
                    self.forget(track.object_id)
                else:
                    track.frame_id = track.mask = None
                    track.association_hint_reason = None

        for ci, candidate in enumerate(candidates):
            if ci not in matched_candidates:
                object_id = f'object_{self._next_id}'
                self._next_id += 1
                self._tracks[object_id] = ObjectTrack(
                    object_id, candidate.position.copy(), candidate.size.copy(), candidate.confidence,
                    timestamp, frame_id, candidate.mask.copy(), candidate.metadata.get('geometry'),
                    confirmed=self.min_confirm_hits == 1,
                )
        return self.tracks()

    def tracks(self):
        return tuple(self._tracks.values())

    def forget(self, object_id):
        self._tracks.pop(object_id, None)

    def get(self, object_id):
        return self._tracks.get(object_id)
