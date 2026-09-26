from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.perception.discovery import DepthObjectDiscoverer, ObjectCandidate
from robot_skill_stack.world.perception.frame import PerceptionFrame
from robot_skill_stack.world.perception.geometry import ObjectGeometryService
from robot_skill_stack.world.perception.localizer import RgbdLocalizer
from robot_skill_stack.world.perception.semantics import (
    CubeGeometryClassifier,
    SemanticBelief,
    SemanticPrediction,
)
from robot_skill_stack.world.perception.tracker import ObjectTracker
from tests.support.results import fail_result, write_result
from robot_skill_stack.world.model.context import AssociationHint, ObservationContext
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.world_model import WorldModel


def candidate(
    x=0.4,
    y=0.0,
    z=0.025,
    size=(0.05, 0.05, 0.05),
    spawnable=True,
):
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 6:11] = True
    return ObjectCandidate(
        np.array([x, y, z]),
        np.array(size),
        mask,
        confidence=1.0,
        spawnable=spawnable,
    )


def prediction():
    return SemanticPrediction("cube", 1.0)


def hint(object_id, position, radius=0.08):
    return AssociationHint(
        object_id=object_id,
        expected_position=np.asarray(position, dtype=float),
        max_distance=radius,
        expires_at=time.monotonic() + 60.0,
        reason="place",
    )


def main(result_path=None):
    metrics = {}

    belief = SemanticBelief(
        window_size=5,
        min_samples=3,
        assignment_threshold=0.6,
    )
    belief.add(SemanticPrediction("box", 0.9))
    belief.add(SemanticPrediction("cube", 0.9))
    belief.add(SemanticPrediction("cube", 0.9))
    assert belief.resolve()[0] == "cube"
    belief.add(SemanticPrediction("box", 1.0))
    belief.add(SemanticPrediction("box", 1.0))
    assert belief.resolve()[0] == "box"
    metrics["semantic_reclassification"] = True

    classifier = CubeGeometryClassifier(
        ratio_max=1.35,
        min_size=0.025,
        max_size=0.10,
    )
    assert classifier.classify(candidate(), None).label == "cube"
    assert (
        classifier.classify(
            candidate(size=(0.05, 0.05, 0.09)),
            None,
        ).label
        is None
    )
    metrics["cube_geometry_classifier"] = True

    tracker = ObjectTracker(
        max_distance=0.15,
        max_misses=2,
        semantic_min_samples=1,
    )
    tracks = tracker.update(
        [candidate()],
        [prediction()],
        frame_id=1,
        timestamp=1.0,
    )
    first_id = tracks[0].object_id
    tracks = tracker.update(
        [candidate(x=0.43)],
        [prediction()],
        frame_id=2,
        timestamp=2.0,
    )
    assert len(tracks) == 1 and tracks[0].object_id == first_id
    tracks = tracker.update(
        [candidate(x=0.43), candidate(x=0.62)],
        [prediction(), prediction()],
        frame_id=3,
        timestamp=3.0,
    )
    assert len(tracks) == 2
    tracker.update([], [], frame_id=4, timestamp=4.0)
    assert not tracker.get(first_id).visible
    metrics["tracker_identity"] = True
    metrics["tracker_new_object"] = True
    metrics["tracker_missing_object"] = True

    held_tracker = ObjectTracker(
        max_distance=0.10,
        max_misses=1,
        semantic_min_samples=1,
    )
    held_id = held_tracker.update(
        [candidate()],
        [prediction()],
        frame_id=1,
        timestamp=1.0,
    )[0].object_id
    held_context = ObservationContext(held_object_id=held_id)
    for frame_id in range(2, 7):
        held_tracker.update(
            [],
            [],
            frame_id=frame_id,
            timestamp=float(frame_id),
            context=held_context,
        )
    assert held_tracker.get(held_id) is not None
    metrics["held_track_retention"] = True

    moved = candidate(x=0.4, y=0.25)
    moved_context = ObservationContext(
        association_hints=(
            hint(held_id, moved.position),
        )
    )
    tracks = held_tracker.update(
        [moved],
        [prediction()],
        frame_id=7,
        timestamp=7.0,
        context=moved_context,
    )
    assert len(tracks) == 1
    assert tracks[0].object_id == held_id
    assert tracks[0].matched_by_hint
    assert np.linalg.norm(tracks[0].position - moved.position) < 1e-9
    metrics["long_distance_hint_reacquisition"] = True

    multi = ObjectTracker(
        max_distance=0.10,
        max_misses=5,
        semantic_min_samples=1,
    )
    first = multi.update(
        [candidate(x=0.35, y=0.0), candidate(x=0.60, y=0.0)],
        [prediction(), prediction()],
        frame_id=1,
        timestamp=1.0,
    )
    id_a, id_b = first[0].object_id, first[1].object_id
    context = ObservationContext(
        association_hints=(
            hint(id_a, [0.35, 0.25, 0.025]),
        )
    )
    multi.update(
        [
            candidate(x=0.60, y=0.01),
            candidate(x=0.35, y=0.25),
        ],
        [prediction(), prediction()],
        frame_id=2,
        timestamp=2.0,
        context=context,
    )
    assert np.linalg.norm(
        multi.get(id_a).position - np.array([0.35, 0.25, 0.025])
    ) < 1e-9
    assert np.linalg.norm(
        multi.get(id_b).position - np.array([0.60, 0.01, 0.025])
    ) < 1e-9
    metrics["hint_priority_with_two_objects"] = True

    model = WorldModel()
    model.register(
        WorldObject(
            object_id="object_1",
            pose=Pose([0.4, 0.0, 0.025]),
            size=np.array([0.05, 0.05, 0.05]),
        )
    )
    model.expect_object_at(
        "object_1",
        [0.4, 0.25, 0.025],
        radius=0.08,
        ttl=3.0,
        reason="place",
    )
    assert len(model.observation_context().association_hints) == 1
    model.apply_observations(
        [
            ObjectObservation(
                object_id="object_1",
                pose=Pose([0.4, 0.25, 0.025]),
                size=np.array([0.05, 0.05, 0.05]),
                visible=True,
                source="test",
                metadata={
                    "association_hint_match": True,
                    "association_hint_reason": "place",
                },
            )
        ]
    )
    assert len(model.observation_context().association_hints) == 0
    metrics["world_model_hint_consumption"] = True

    mask = np.zeros((10, 12), dtype=bool)
    mask[1:3, 1:3] = True
    mask[6:9, 8:11] = True
    components = DepthObjectDiscoverer._components(mask)
    assert sorted(len(xs) for _, xs in components) == [4, 9]
    metrics["connected_components"] = True

    K = np.array(
        [[100.0, 0, 10.0], [0, 100.0, 10.0], [0, 0, 1.0]]
    )
    localizer = RgbdLocalizer(K)
    tracker2 = ObjectTracker(semantic_min_samples=1)
    track = tracker2.update(
        [candidate()],
        [prediction()],
        frame_id=1,
        timestamp=1.0,
    )[0]
    frame = PerceptionFrame(
        1,
        np.zeros((20, 20, 3), dtype=np.uint8),
        np.ones((20, 20), dtype=np.float32),
        K,
        np.eye(4),
        1.0,
    )
    geometry = ObjectGeometryService(localizer, tracker2)
    geometry.update_frame(frame)
    cloud = geometry.get_point_cloud(track.object_id)
    assert cloud is not None and cloud.shape == (25, 3)
    metrics["lazy_point_cloud_points"] = int(len(cloud))

    print("=== PERCEPTION V1 UNIT ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("PASS")
    print("==========================")
    write_result(result_path, status="PASS", metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result")
    args = parser.parse_args()
    try:
        main(args.result)
    except Exception as exc:
        fail_result(args.result, exc)
        sys.exit(1)
