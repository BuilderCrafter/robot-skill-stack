"""Research-harness correctness; these tests do not prescribe detector answers."""
from types import SimpleNamespace
import unittest

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from tests.support.multiview_iteration2 import (
    EYES, SEEDS, SnapshotRig, footprint_half_extent, object_from_payload, object_payload,
    perturb_frames, robustness_scenes, scene_objects, score_observations, sphere,
    surface_memberships,
)


class IterationTwoResearchTests(unittest.TestCase):
    def test_manifest_is_reproducible_with_distinct_holdout_seeds_and_viewpoints(self):
        first, second = robustness_scenes(), robustness_scenes()
        self.assertEqual(len(first), 54)
        self.assertEqual([(scene.key, scene.seed) for scene in first], [(scene.key, scene.seed) for scene in second])
        self.assertNotEqual(SEEDS['selection'], SEEDS['holdout'])
        self.assertNotEqual(EYES['selection'], EYES['holdout'])
        for scene_a, scene_b in zip(first, second):
            for obj_a, obj_b in zip(scene_a.objects, scene_b.objects):
                np.testing.assert_array_equal(obj_a.center, obj_b.center)
                restored = object_from_payload(object_payload(obj_a))
                np.testing.assert_array_equal(restored.center, obj_a.center)

    def test_declared_close_gaps_are_geometrically_exact(self):
        prim_names = []
        for scene in robustness_scenes():
            if scene.gap_mm is None:
                continue
            first, second = scene.objects
            delta = second.center[:2]-first.center[:2]
            direction = delta/np.linalg.norm(delta)
            gap = (np.linalg.norm(delta)-footprint_half_extent(first, direction)-
                   footprint_half_extent(second, direction))
            self.assertAlmostEqual(gap*1000, scene.gap_mm, places=8, msg=scene.key)
            prim_names.extend(obj.name for obj in scene.objects)
        self.assertEqual(len(prim_names), len(set(prim_names)), 'USD gap fixtures need unique schema paths')

    def test_invisible_gap_fixture_fails_a_precondition_instead_of_scoring_a_detector_miss(self):
        from scripts.benchmark_multiview_robustness import validate_gap_fixture
        record = dict(key='selection_gap', gap_mm=2., frames=[dict(index=2,
            objects=[dict(name='A')], visible_surface_points={'0': {'A': 0}, '1': {'A': 0}})])
        errors = validate_gap_fixture(record)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['kind'], 'native_fixture_target_not_observed')
        record['frames'][0]['visible_surface_points']['1']['A'] = 8
        self.assertEqual(validate_gap_fixture(record), [])

    def test_exclusive_surface_scoring_does_not_invent_a_merge_at_two_mm_gap(self):
        objects = (sphere('A', (.4, 0.), .025), sphere('B', (.452, 0.), .025))
        points = np.array([[.425, 0., .025], [.427, 0., .025]])
        np.testing.assert_array_equal(surface_memberships(points, objects), [0, 1])

    def test_scoring_retains_miss_false_split_mixed_and_identity_errors(self):
        obj_a, obj_b = sphere('A', (.4, 0.), .025), sphere('B', (.5, 0.), .025)
        def observation(key, obj):
            return ObjectObservation(key, pose=Pose(obj.center.copy()), size=obj.size,
                geometry=PrimitiveGeometry('sphere', .8, radius=obj.radius, scores={'sphere': .8, 'cube': .2}))
        theta = np.linspace(0., 2*np.pi, 40, endpoint=False)
        def cloud(obj):
            return obj.center + np.column_stack((obj.radius*np.cos(theta), obj.radius*np.sin(theta), theta*0))
        observations = [observation('new_A', obj_a), observation('duplicate_A', obj_a)]
        rows, counts, _ = score_observations((obj_a, obj_b), observations, [cloud(obj_a), cloud(obj_a)],
            method='test', trial='failure', frame_index=2, previous_ids={'A': 'old_A'})
        self.assertEqual([row['predicted'] for row in rows], ['sphere', 'missing'])
        self.assertEqual(counts['missing'], 1)
        self.assertEqual(counts['false_splits'], 1)
        self.assertEqual(counts['identity_switches'], 1)
        self.assertEqual(counts['false_positives'], 1)
        _, counts, _ = score_observations((obj_a, obj_b), [observations[0]], [np.r_[cloud(obj_a), cloud(obj_b)]],
            method='test', trial='mixed', frame_index=2)
        self.assertEqual(counts['mixed_accepted_clouds'], 1)

    def test_snapshot_rig_never_republishes_or_retimestamps(self):
        frame = SimpleNamespace(token=(3., .3), acquired_at=123.)
        rig = SnapshotRig()
        rig.push((frame, frame))
        captured = rig.capture()
        self.assertIs(captured[0], frame)
        self.assertEqual(captured[0].acquired_at, 123.)
        self.assertIsNone(rig.capture())

    def test_perturbations_are_deterministic_and_preserve_snapshot_identity(self):
        frame = DepthFrame('A', (3., .3), np.full((12, 16), 1.), np.eye(3), np.eye(4), 123.,
                           np.zeros((12, 16, 3), np.uint8))
        for variant in ('clean', 'depth_noise', 'calibration', 'sparse'):
            first = perturb_frames((frame, frame), variant, seed=42)
            second = perturb_frames((frame, frame), variant, seed=42)
            np.testing.assert_array_equal(first[0].depth, second[0].depth)
            self.assertEqual(first[0].token, frame.token)
            self.assertEqual(first[0].acquired_at, frame.acquired_at)
        np.testing.assert_array_equal(frame.depth, np.ones((12, 16)))

    def test_temporal_fixture_moves_hides_and_reintroduces_same_truth_name(self):
        scene = next(scene for scene in robustness_scenes('holdout') if scene.temporal)
        self.assertEqual(len(scene_objects(scene, 4)), 1)
        self.assertEqual(len(scene_objects(scene, 5)), 2)
        self.assertEqual(scene_objects(scene, 5)[0].name, scene.objects[0].name)
        self.assertGreater(np.linalg.norm(scene_objects(scene, 5)[0].center-scene.objects[0].center), .01)


if __name__ == '__main__':
    unittest.main()
