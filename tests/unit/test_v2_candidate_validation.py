from dataclasses import replace
from pathlib import Path
import unittest
import numpy as np
from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception.factory import build_tracker
from robot_skill_stack.world.perception.v2.pipeline import FrameProcessor
from robot_skill_stack.world.perception.v2.settings import V2Config
from robot_skill_stack.world.perception.v2.types import Instance, Segmentation
from robot_skill_stack.world.perception.v2.self_filter import SelfFilterResult
from tests.support.v2_helpers import scene, Solid, INFO

ROOT = Path(__file__).resolve().parents[2]


class CandidateValidationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_scene_config(ROOT/'config/scenes/playground_v2.toml')
        self.config = replace(self.cfg.perception.v2, unknown_depth_fallback=False, robot_self_filter=False)
        self.processor = FrameProcessor(self.cfg.perception.discovery, self.cfg.perception.primitives, self.config)
        self.frame, self.instances = scene()

    def process(self, instances=None, frame=None, robot=None):
        frame = self.frame if frame is None else frame
        return self.processor.process(frame, Segmentation(frame.frame_id, self.instances if instances is None else instances, INFO), robot)

    def ghost_mask(self):
        m = self.instances[0].mask
        y, x = np.nonzero(m)
        ghost = np.zeros(m.shape, bool)
        cut = int(np.quantile(x, .7))
        ghost[max(0, y.min()-22):min(m.shape[0], y.max()+22), cut:min(m.shape[1], x.max()+50)] = True
        return Instance(ghost, 'sphere', .97)

    def test_pure_table_mask_rejected(self):
        mask = np.zeros(self.frame.depth.shape, bool); mask[175:210, 100:170] = True
        self.assertFalse(self.process([Instance(mask, 'sphere', .99)]).candidates)

    def test_shadow_with_real_object_fringe_is_rejected(self):
        result = self.process([self.ghost_mask()])
        self.assertFalse(result.candidates)
        self.assertEqual(result.diagnostics[0]['rejected'], 'insufficient_foreground_fraction')
        self.assertLess(result.diagnostics[0]['foreground_fraction'], .2)

    def test_true_object_plus_shadow_never_creates_second_track(self):
        tracker = build_tracker(self.cfg)
        for frame_id in range(3):
            result = self.process([*self.instances, self.ghost_mask()])
            tracker.update(result.candidates, result.predictions, frame_id=frame_id, timestamp=100+frame_id)
        tracks = [t for t in tracker.tracks() if t.confirmed]
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].class_name, 'sphere')

    def test_identical_masks_are_deduplicated(self):
        result = self.process(self.instances*2)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(sum(d['rejected'] == 'duplicate_surface' for d in result.diagnostics), 1)

    def test_nested_surface_fragment_is_deduplicated(self):
        m = self.instances[0].mask.copy()
        y, x = np.nonzero(m); m[:, :int(np.quantile(x, .4))] = False
        result = self.process([*self.instances, Instance(m, 'sphere', .99)])
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(int(result.candidates[0].mask.sum()), int(self.instances[0].mask.sum()))

    def test_disjoint_sphere_fragments_with_same_fit_are_deduplicated(self):
        m = self.instances[0].mask
        _, x = np.nonzero(m); cut = int(np.median(x))
        a, b = m.copy(), m.copy(); a[:, cut:] = False; b[:, :cut] = False
        result = self.process([Instance(a, 'sphere', .95), Instance(b, 'sphere', .94)])
        self.assertEqual(len(result.candidates), 1)
        self.assertTrue(any(d.get('duplicate_basis') == 'same_fitted_solid' for d in result.diagnostics))

    def test_two_nearby_spheres_not_merged(self):
        f, i = scene([Solid('sphere', center=(.42, 0, .025)), Solid('sphere', center=(.475, 0, .025))])
        self.assertEqual(len(self.process(i, f).candidates), 2)

    def test_two_touching_cubes_not_merged(self):
        f, i = scene([Solid('cube', center=(.42, 0, .025)), Solid('cube', center=(.47, 0, .025))])
        self.assertEqual(len(self.process(i, f).candidates), 2)

    def test_contradictory_class_is_unknown_not_named_cube(self):
        result = self.process([Instance(self.instances[0].mask, 'cube', .99)])
        self.assertEqual(len(result.candidates), 1)
        self.assertIsNone(result.predictions[0].label)
        self.assertEqual(result.candidates[0].metadata['geometry'].shape.value, 'unknown')
        self.assertEqual(result.diagnostics[0]['label'], 'cube')
        self.assertEqual(result.diagnostics[0]['class_validation'], 'unknown')

    def test_top_only_real_cylinder_not_rejected_as_flat(self):
        f, i = scene([Solid('cylinder', center=(.45, 0, .05), radius=.02)], eye=(.45, 0, .9))
        result = self.process(i, f)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.predictions[0].label, 'cylinder')
        self.assertEqual(result.candidates[0].metadata['geometry'].shape.value, 'unknown')

    def test_elevated_cube_no_table_stretch(self):
        f, i = scene([Solid('cube', center=(.45, 0, .16))])
        c = self.process(i, f).candidates[0]
        self.assertLess(c.size[2], .06)
        self.assertGreater(c.position[2], .13)

    def test_robot_only_detection_rejected(self):
        m = self.instances[0].mask
        robot = SelfFilterResult(m, np.where(m, self.frame.depth, np.inf), {'enabled': True, 'frame_id': self.frame.frame_id})
        result = self.process(robot=robot)
        self.assertFalse(result.candidates)
        self.assertEqual(result.diagnostics[0]['rejected'], 'robot_surface')

    def test_fallback_cannot_recreate_removed_robot(self):
        self.processor.config = replace(self.config, unknown_depth_fallback=True)
        m = self.instances[0].mask
        robot = SelfFilterResult(m, np.where(m, self.frame.depth, np.inf), {'enabled': True, 'frame_id': self.frame.frame_id})
        self.assertFalse(self.process([], robot=robot).candidates)

    def test_fallback_can_recover_object_after_bad_yolo_mask(self):
        self.processor.config = replace(self.config, unknown_depth_fallback=True)
        result = self.process([self.ghost_mask()])
        self.assertEqual(len(result.candidates), 1)
        self.assertIsNone(result.predictions[0].label)
        self.assertTrue(any(d['rejected'] == 'insufficient_foreground_fraction' for d in result.diagnostics))

    def test_fallback_deduplicates_after_cleaning(self):
        self.processor.config = replace(self.config, unknown_depth_fallback=True)
        result = self.process()
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.predictions[0].label, 'sphere')
        self.assertTrue(any(d['source'] == 'depth_fallback' and d['rejected'] == 'duplicate_surface' for d in result.diagnostics))

    def test_sparse_depth_has_explicit_rejection(self):
        self.frame, self.instances = scene(resolution=(640, 480))
        mask = self.instances[0].mask
        d = np.full_like(self.frame.depth, np.nan)
        y, x = np.nonzero(mask); d[y[:31], x[:31]] = self.frame.depth[y[:31], x[:31]]
        result = self.process(frame=replace(self.frame, depth=d))
        self.assertFalse(result.candidates)
        self.assertEqual(result.diagnostics[0]['rejected'], 'sparse_valid_depth')

    def test_robot_mask_frame_mismatch_rejected(self):
        robot = SelfFilterResult(np.zeros(self.frame.depth.shape, bool), self.frame.depth, {'enabled': True, 'frame_id': -1})
        with self.assertRaisesRegex(ValueError, 'frame IDs'):
            self.process(robot=robot)

    def test_enabled_filter_without_snapshot_fails_closed(self):
        self.processor.config = replace(self.config, robot_self_filter=True)
        with self.assertRaisesRegex(ValueError, 'no time-matched'):
            self.process()

    def test_config_rejects_unsafe_tolerances_and_fractions(self):
        for values in ({'robot_depth_tolerance_m': .1}, {'robot_time_tolerance_s': 1}, {'robot_self_filter': 'yes'},
                       {'min_foreground_fraction': 0}, {'duplicate_surface_overlap': 1.5}, {'min_component_fraction': float('nan')}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                V2Config(**values)


if __name__ == '__main__':
    unittest.main(verbosity=2)
