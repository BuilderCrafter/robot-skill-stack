from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SemanticPrediction:
    label: str | None
    confidence: float


class SemanticBelief:
    def __init__(self, window_size=7, min_samples=3, assignment_threshold=0.70):
        self.history = deque(maxlen=int(window_size))
        self.min_samples = int(min_samples)
        self.assignment_threshold = float(assignment_threshold)

    def add(self, prediction: SemanticPrediction | None):
        if prediction is None:
            prediction = SemanticPrediction(None, 1.0)
        confidence = float(np.clip(prediction.confidence, 0.0, 1.0))
        self.history.append(SemanticPrediction(prediction.label, confidence))

    def resolve(self):
        if len(self.history) < self.min_samples:
            return None, 0.0

        scores = {}
        total = 0.0
        for item in self.history:
            weight = max(item.confidence, 1e-6)
            scores[item.label] = scores.get(item.label, 0.0) + weight
            total += weight

        label, score = max(scores.items(), key=lambda kv: kv[1])
        share = score / max(total, 1e-9)
        if label is None or share < self.assignment_threshold:
            return None, share
        return label, share

    @property
    def samples(self):
        return len(self.history)


class CubeGeometryClassifier:
    def __init__(self, *, ratio_max=1.35, min_size=0.025, max_size=0.10):
        self.ratio_max = float(ratio_max)
        self.min_size = float(min_size)
        self.max_size = float(max_size)

    def classify(self, candidate, rgb=None):
        size = np.asarray(candidate.size, dtype=float)
        if np.any(size < self.min_size) or np.any(size > self.max_size):
            return SemanticPrediction(None, 1.0)

        ratio = float(np.max(size) / max(np.min(size), 1e-6))
        if ratio > self.ratio_max:
            return SemanticPrediction(None, min(1.0, ratio / self.ratio_max - 1.0 + 0.5))

        span = max(self.ratio_max - 1.0, 1e-6)
        confidence = float(np.clip(1.0 - (ratio - 1.0) / span, 0.5, 1.0))
        return SemanticPrediction("cube", confidence)
