from __future__ import annotations

import argparse
import sys

import numpy as np

from perception.discovery import DepthObjectDiscoverer, ObjectCandidate
from perception.frame import PerceptionFrame
from perception.geometry_service import ObjectGeometryService
from perception.rgbd_localizer import RgbdLocalizer
from perception.semantics import CubeGeometryClassifier, SemanticBelief, SemanticPrediction
from perception.tracker import ObjectTracker
from scripts.perception_v1_common import fail_result, write_result


def candidate(x=0.4, y=0.0, z=0.025, size=(0.05, 0.05, 0.05), spawnable=True):
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 6:11] = True
    return ObjectCandidate(
        np.array([x, y, z]),
        np.array(size),
        mask,
        confidence=1.0,
        spawnable=spawnable,
    )


def main(result_path=None):
    metrics = {}

    belief = SemanticBelief(window_size=5, min_samples=3, assignment_threshold=0.6)
    belief.add(SemanticPrediction("box", 0.9))
    belief.add(SemanticPrediction("cube", 0.9))
    belief.add(SemanticPrediction("cube", 0.9))
    assert belief.resolve()[0] == "cube"
    belief.add(SemanticPrediction("box", 1.0))
    belief.add(SemanticPrediction("box", 1.0))
    assert belief.resolve()[0] == "box"
    metrics["semantic_reclassification"] = True

    classifier = CubeGeometryClassifier(ratio_max=1.35, min_size=0.025, max_size=0.10)
    assert classifier.classify(candidate(), None).label == "cube"
    assert classifier.classify(candidate(size=(0.05, 0.05, 0.09)), None).label is None
    metrics["cube_geometry_classifier"] = True

    tracker = ObjectTracker(max_distance=0.15, max_misses=2, semantic_min_samples=1)
    tracks = tracker.update(
        [candidate()],
        [SemanticPrediction("cube", 1.0)],
        frame_id=1,
        timestamp=1.0,
    )
    first_id = tracks[0].object_id
    tracks = tracker.update(
        [candidate(x=0.43)],
        [SemanticPrediction("cube", 1.0)],
        frame_id=2,
        timestamp=2.0,
    )
    assert len(tracks) == 1 and tracks[0].object_id == first_id
    tracks = tracker.update(
        [candidate(x=0.43), candidate(x=0.62)],
        [SemanticPrediction("cube", 1.0), SemanticPrediction("cube", 1.0)],
        frame_id=3,
        timestamp=3.0,
    )
    assert len(tracks) == 2
    tracker.update([], [], frame_id=4, timestamp=4.0)
    assert not tracker.get(first_id).visible
    metrics["tracker_identity"] = True
    metrics["tracker_new_object"] = True
    metrics["tracker_missing_object"] = True

    mask = np.zeros((10, 12), dtype=bool)
    mask[1:3, 1:3] = True
    mask[6:9, 8:11] = True
    components = DepthObjectDiscoverer._components(mask)
    assert sorted(len(xs) for _, xs in components) == [4, 9]
    metrics["connected_components"] = True

    K = np.array([[100.0, 0, 10.0], [0, 100.0, 10.0], [0, 0, 1.0]])
    localizer = RgbdLocalizer(K)
    tracker2 = ObjectTracker(semantic_min_samples=1)
    track = tracker2.update(
        [candidate()],
        [SemanticPrediction("cube", 1.0)],
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
