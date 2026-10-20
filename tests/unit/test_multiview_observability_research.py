"""Independent calibrated-ray preflights, not perception acceptance tests."""
from dataclasses import replace
import unittest

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from tests.support.multiview_iteration2 import object_payload
from tests.support.multiview_readiness_research import EYES
from tests.support.multiview_observability_research import (
    CASES, SEEDS, ObservabilityScene, _look_at, audit_observability_scene, observability_scenes,
    scene_frame_content, scene_frame_count,
)
from tests.support.primitive_scene import Solid, intersect


def ideal_fixture_frames(scene, index=3, resolution=(320, 240)):
    """Independent analytic intersections with the native scene's pinhole lens."""
    objects, clutter, _ = scene_frame_content(scene, index)
    solids = [Solid(obj.shape, tuple(obj.center), tuple(obj.size), obj.yaw,
                    obj.axis, obj.radius, obj.length) for obj in objects+clutter]
    width, height = resolution
    k = np.array([[554.25622559*width/640, 0., width/2],
                  [0., 554.25628662*height/480, height/2], [0., 0., 1.]])
    y, x = np.indices((height, width))
    rays = np.c_[(x.ravel()+.5-k[0, 2])/k[0, 0],
                 (y.ravel()+.5-k[1, 2])/k[1, 1], np.ones(height*width)]
    frames = []
    for ci, (eye, target) in enumerate(zip(scene.eyes, scene.targets)):
        rotation = _look_at(eye, target)
        world_rays = rays @ rotation.T
        origin = np.asarray(eye)
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = -origin[2]/world_rays[:, 2]
        depth = np.where(depth > 0, depth, np.inf)
        for solid in solids:
            depth = np.minimum(depth, intersect(solid, origin, world_rays))
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, origin
        frames.append(DepthFrame(str(ci), (index+1, (index+1)*.1),
            depth.reshape(height, width).astype(np.float32), k.copy(), transform, .1*(index+1)))
    return tuple(frames)


class ObservabilityResearchTests(unittest.TestCase):
    def test_selection_coverage_identity_and_mandatory_geometry(self):
        from scripts.benchmark_multiview_partial_grasp import partial_scenes
        scenes = observability_scenes()
        self.assertEqual(len(scenes), 45)
        self.assertEqual(len({scene.key for scene in scenes}), 45)
        self.assertEqual({scene.requirements['case'] for scene in scenes[:40]}, set(CASES))
        self.assertEqual(len(set(SEEDS.values())), 2)
        # Do not generate or inspect the independent holdout before source freeze.
        legacy = next(scene for scene in scenes if scene.name == 'partial_box_both')
        old = partial_scenes()[-1]
        self.assertEqual([object_payload(obj) for obj in legacy.objects+legacy.clutter],
                         [object_payload(obj) for obj in old.objects+old.clutter])
        self.assertEqual(legacy.eyes, EYES['selection'])
        temporal = next(scene for scene in scenes if scene.temporal)
        self.assertEqual(scene_frame_count(temporal), 9)
        contents = [scene_frame_content(temporal, i) for i in range(9)]
        self.assertEqual([row[2] for row in contents], ['complete']*4+['incomplete']*2+['restored']*3)
        self.assertTrue(all(row[0] is temporal.objects for row in contents))
        self.assertFalse(contents[3][1])
        self.assertTrue(contents[4][1])
        self.assertFalse(contents[6][1])
        from robot_skill_stack.world.model.world_model import WorldModel
        from tests.support.multiview_readiness_research import make_planner, truth_world_object
        self.assertTrue(make_planner(WorldModel()).plan(truth_world_object(temporal.objects[0])).ok)

    def test_corrected_legacy_b_control_at_actual_native_resolution(self):
        from scripts.benchmark_multiview_partial_grasp import partial_scenes
        old = next(scene for scene in partial_scenes() if scene.name == 'partial_box_B')
        scene = ObservabilityScene(old.name, old.split, old.seed, old.objects, old.clutter,
            EYES['selection'], ((.45, 0., .04),)*2,
            dict(full_views=[0], clipped=[[1, 0]], fused_full=True))
        audit = audit_observability_scene(scene, ideal_fixture_frames(scene, resolution=(640, 480)))
        self.assertTrue(audit['valid'], audit['errors'])

    def test_selection_physical_visibility_preflight(self):
        failures = []
        for scene in observability_scenes():
            indexes = (3, 4, 6) if scene.temporal else (3,)
            for index in indexes:
                audit = audit_observability_scene(scene, ideal_fixture_frames(scene, index), frame_index=index)
                if not audit['valid']:
                    failures.append((scene.key, index, audit['errors']))
        self.assertEqual(failures, [])

    def test_missing_depth_is_invalid_fixture_without_silent_success(self):
        scene = observability_scenes()[0]
        frames = ideal_fixture_frames(scene)
        invalid = tuple(replace(frame, depth=np.full_like(frame.depth, np.nan)) for frame in frames)
        audit = audit_observability_scene(scene, invalid)
        self.assertFalse(audit['valid'])
        self.assertIn('target_missing_or_insufficient_raw_surface_samples', audit['errors'])
        self.assertIsNone(audit['measurements'][0]['raw_cameras'][0]['span_mm'])


if __name__ == '__main__':
    unittest.main()
