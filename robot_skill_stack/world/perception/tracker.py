from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_skill_stack.world.perception.semantics import SemanticBelief


@dataclass
class ObjectTrack:
    object_id: str
    position: np.ndarray
    size: np.ndarray
    confidence: float
    last_seen: float
    frame_id: int | None
    mask: np.ndarray | None
    belief: SemanticBelief
    hits: int = 1
    misses: int = 0
    matched_by_hint: bool = False
    association_hint_reason: str | None = None

    @property
    def visible(self):
        return self.misses == 0

    @property
    def class_name(self):
        return self.belief.resolve()[0]

    @property
    def class_confidence(self):
        return self.belief.resolve()[1]


class ObjectTracker:
    def __init__(
        self,
        *,
        max_distance=0.18,
        max_size_ratio=1.5,
        max_misses=10,
        occlusion_distance=0.12,
        semantic_window=7,
        semantic_min_samples=3,
        semantic_threshold=0.70,
    ):
        self.max_distance = float(max_distance)
        self.max_size_ratio = float(max_size_ratio)
        self.max_misses = int(max_misses)
        self.occlusion_distance = float(occlusion_distance)
        self.semantic_window = int(semantic_window)
        self.semantic_min_samples = int(semantic_min_samples)
        self.semantic_threshold = float(semantic_threshold)
        self._tracks: dict[str, ObjectTrack] = {}
        self._next_id = 1

    def _new_belief(self):
        return SemanticBelief(
            window_size=self.semantic_window,
            min_samples=self.semantic_min_samples,
            assignment_threshold=self.semantic_threshold,
        )

    def _new_track(self, candidate, prediction, frame_id, timestamp):
        object_id = f"object_{self._next_id}"
        self._next_id += 1
        belief = self._new_belief()
        belief.add(prediction)
        track = ObjectTrack(
            object_id=object_id,
            position=candidate.position.copy(),
            size=candidate.size.copy(),
            confidence=candidate.confidence,
            last_seen=timestamp,
            frame_id=frame_id,
            mask=candidate.mask.copy(),
            belief=belief,
        )
        self._tracks[object_id] = track
        return track

    @staticmethod
    def _size_ratio(track, candidate):
        return float(
            np.linalg.norm(candidate.size - track.size)
            / max(np.linalg.norm(track.size), 1e-6)
        )

    def _apply_match(
        self,
        track,
        candidate,
        prediction,
        *,
        size_ratio,
        frame_id,
        timestamp,
        hint_reason=None,
    ):
        track.position = candidate.position.copy()
        if candidate.spawnable or size_ratio <= 0.5:
            track.size = candidate.size.copy()
        track.confidence = candidate.confidence
        track.last_seen = timestamp
        track.frame_id = frame_id
        track.mask = candidate.mask.copy()
        track.hits += 1
        track.misses = 0
        track.matched_by_hint = hint_reason is not None
        track.association_hint_reason = hint_reason
        if candidate.spawnable:
            track.belief.add(prediction)

    def update(
        self,
        candidates,
        predictions,
        *,
        frame_id,
        timestamp,
        context=None,
    ):
        tracks = list(self._tracks.values())
        matched_tracks, matched_candidates = set(), set()

        hints = ()
        held_object_id = None
        if context is not None:
            hints = context.association_hints
            held_object_id = context.held_object_id

        track_index = {
            track.object_id: i
            for i, track in enumerate(tracks)
        }

        hint_pairs = []
        for hint in hints:
            ti = track_index.get(hint.object_id)
            if ti is None:
                continue
            track = tracks[ti]

            for ci, candidate in enumerate(candidates):
                if not candidate.spawnable:
                    continue

                distance = float(
                    np.linalg.norm(
                        candidate.position - hint.expected_position
                    )
                )
                if distance > hint.max_distance:
                    continue

                size_ratio = self._size_ratio(track, candidate)
                if size_ratio > self.max_size_ratio:
                    continue

                score = (
                    distance / max(hint.max_distance, 1e-6)
                    + 0.25 * min(size_ratio, 2.0)
                )
                hint_pairs.append(
                    (score, ti, ci, size_ratio, hint.reason)
                )

        for _, ti, ci, size_ratio, reason in sorted(hint_pairs):
            if ti in matched_tracks or ci in matched_candidates:
                continue
            self._apply_match(
                tracks[ti],
                candidates[ci],
                predictions[ci],
                size_ratio=size_ratio,
                frame_id=frame_id,
                timestamp=timestamp,
                hint_reason=reason,
            )
            matched_tracks.add(ti)
            matched_candidates.add(ci)

        normal_pairs = []
        for ti, track in enumerate(tracks):
            if ti in matched_tracks:
                continue

            for ci, candidate in enumerate(candidates):
                if ci in matched_candidates:
                    continue

                distance = float(
                    np.linalg.norm(track.position - candidate.position)
                )
                if distance > self.max_distance:
                    continue

                size_ratio = self._size_ratio(track, candidate)
                if candidate.spawnable:
                    if size_ratio > self.max_size_ratio:
                        continue
                elif distance > self.occlusion_distance:
                    continue

                score = (
                    distance / max(self.max_distance, 1e-6)
                    + 0.25 * min(size_ratio, 2.0)
                )
                normal_pairs.append((score, ti, ci, size_ratio))

        for _, ti, ci, size_ratio in sorted(normal_pairs):
            if ti in matched_tracks or ci in matched_candidates:
                continue
            self._apply_match(
                tracks[ti],
                candidates[ci],
                predictions[ci],
                size_ratio=size_ratio,
                frame_id=frame_id,
                timestamp=timestamp,
            )
            matched_tracks.add(ti)
            matched_candidates.add(ci)

        for ti, track in enumerate(tracks):
            if ti in matched_tracks:
                continue
            track.misses += 1
            track.frame_id = None
            track.mask = None
            track.matched_by_hint = False
            track.association_hint_reason = None

        for ci, candidate in enumerate(candidates):
            if ci in matched_candidates or not candidate.spawnable:
                continue
            self._new_track(
                candidate,
                predictions[ci],
                frame_id,
                timestamp,
            )

        protected = {hint.object_id for hint in hints}
        if held_object_id is not None:
            protected.add(held_object_id)

        expired = [
            object_id
            for object_id, track in self._tracks.items()
            if track.misses > self.max_misses
            and object_id not in protected
        ]
        for object_id in expired:
            del self._tracks[object_id]

        return self.tracks()

    def tracks(self):
        return tuple(self._tracks.values())

    def get(self, object_id):
        return self._tracks.get(object_id)
