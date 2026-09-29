from __future__ import annotations
import contextlib
from dataclasses import replace
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception.factory import build_perception_provider
from robot_skill_stack.world.perception.v2.types import CLASSES, Segmentation
from tools.perception_v2.dataset import list_samples, load_sample, read_png, sha256
from tools.perception_v2.metrics import Metrics, assign, iou_matrix
from tools.perception_v2.benchmark import evaluate, run_v1, candidates_output
from tools.perception_v2_data.common import write_png
from tests.support.v2_helpers import INFO, scene, FrameCamera, Solid

ROOT = Path(__file__).resolve().parents[2]


def obj(label='cube', mask=None, position=None):
    return dict(label=label, mask=np.eye(8, dtype=bool) if mask is None else mask,
                position=np.zeros(3) if position is None else np.asarray(position), size=np.ones(3), elevated=False)


def make_dataset(root):
    root = Path(root)
    scenes = [[Solid('cube', yaw=.4)], [Solid('sphere')],
              [Solid('cylinder', center=(.45, 0, .05), radius=.02)], []]
    segmentations = {}
    for index, solids in enumerate(scenes):
        frame, instances = scene(solids)
        masks, entries = np.zeros(frame.depth.shape, np.uint16), []
        for instance_id, (solid, instance) in enumerate(zip(solids, instances), 1):
            masks[instance.mask] = instance_id
            size = list(solid.size) if solid.shape == 'cube' else ([solid.radius*2]*3 if solid.shape == 'sphere' else [solid.radius*2]*2+[solid.length])
            entries.append(dict(instance_id=instance_id, shape=solid.shape, class_id=CLASSES.index(solid.shape),
                                visible_pixels=int(instance.mask.sum()), position=list(solid.center),
                                size_world_aabb=size, elevated=False))
        name = f'{index:06}'
        rgbp, maskp = root/f'images/test/{name}.png', root/f'masks/test/{name}.png'
        write_png(rgbp, frame.rgb)
        write_png(maskp, masks)
        (root/'depth/test').mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root/f'depth/test/{name}.npz', depth=frame.depth)
        meta = dict(frame_id=index, split='test', objects=entries, K=frame.intrinsics.tolist(),
                    T_world_camera_opencv=frame.world_from_camera.tolist(), resolution=[frame.rgb.shape[1], frame.rgb.shape[0]],
                    depth_type='distance_to_image_plane_m', support_z=0., rgb_sha256=sha256(rgbp), mask_sha256=sha256(maskp))
        (root/'meta/test').mkdir(parents=True, exist_ok=True)
        (root/f'meta/test/{name}.json').write_text(json.dumps(meta))
        segmentations[index] = Segmentation(index, instances, INFO)
    (root/'manifest.json').write_text(json.dumps(dict(format='primitive-sdg-v1', classes=list(CLASSES), settings=dict(support_z=0., save_depth=True))))
    (root/'progress.json').write_text(json.dumps(dict(complete=True, splits=dict(test=len(scenes)))))
    return segmentations


class MatchingTests(unittest.TestCase):
    def test_perfect_detection(self):
        m = Metrics()
        m.add([obj()], [obj()], .01)
        r = m.report()
        self.assertEqual(r['named_target_f1'], 1.)
        self.assertEqual(r['localization_recall'], 1.)
        self.assertEqual(r['center_error_mm']['mean'], 0.)

    def test_wrong_class_is_both_fp_and_fn(self):
        m = Metrics()
        m.add([obj('cube')], [obj('sphere')], .01)
        r = m.report()
        self.assertEqual(r['localization_recall'], 1.)
        self.assertEqual(r['named_target_recall'], 0.)
        self.assertEqual(r['per_class']['cube']['fn'], 1)
        self.assertEqual(r['per_class']['sphere']['fp'], 1)
        self.assertEqual(r['confusion']['cube']['sphere'], 1)

    def test_unknown_localizes_but_is_not_correct_classification(self):
        m = Metrics()
        m.add([obj()], [obj(None)], .01)
        r = m.report()
        self.assertEqual(r['classification_accuracy_on_localized'], 0.)
        self.assertEqual(r['named_target_recall'], 0.)
        self.assertIsNone(r['named_target_precision'])
        self.assertEqual(r['confusion']['cube']['unknown'], 1)

    def test_duplicate_predictions_do_not_double_count(self):
        m = Metrics()
        m.add([obj()], [obj(), obj()], .01)
        r = m.report()
        self.assertEqual(r['named_target_precision'], .5)
        self.assertEqual(r['named_target_recall'], 1.)
        self.assertEqual(r['per_class']['cube']['fp'], 1)

    def test_empty_negatives_and_false_positives(self):
        m = Metrics()
        m.add([], [], .01)
        m.add([], [obj()], .01)
        r = m.report()
        self.assertEqual(r['counts']['negative_false_positive_frames'], 1)
        self.assertEqual(r['per_class']['cube']['fp'], 1)
        self.assertIsNone(r['named_target_recall'])

    def test_misses_remain_in_denominator(self):
        m = Metrics()
        m.add([obj()], [], .01)
        r = m.report()
        self.assertEqual(r['named_target_recall'], 0.)
        self.assertEqual(r['confusion']['cube']['missed'], 1)
        self.assertIsNone(r['classification_accuracy_on_localized'])

    def test_iou_threshold_not_just_bbox(self):
        a = np.zeros((8, 8), bool)
        a[:, :2] = True
        b = np.zeros_like(a)
        b[:, 1:3] = True
        self.assertAlmostEqual(iou_matrix([obj(mask=a)], [obj(mask=b)])[0, 0], 1/3)
        m = Metrics(.5)
        m.add([obj(mask=a)], [obj(mask=b)], .01)
        self.assertEqual(m.report()['named_target_recall'], 0.)

    def test_maximum_cardinality_not_greedy(self):
        ious = np.array([[.9, .6], [.6, .1]])
        self.assertEqual(set(assign(ious, ious >= .5)), {(0, 1), (1, 0)})

    def test_assignment_matches_bruteforce_small_matrices(self):
        rng = np.random.default_rng(14)
        for _ in range(30):
            a = rng.random((3, 3))
            allowed = a > .4
            pairs = assign(a, allowed)
            score = (len(pairs), sum(a[i, j] for i, j in pairs))
            scores = []
            for perm in itertools.permutations(range(3)):
                valid = [(i, j) for i, j in enumerate(perm) if allowed[i, j]]
                scores.append((len(valid), sum(a[i, j] for i, j in valid)))
            self.assertEqual(score[0], max(scores)[0])
            self.assertAlmostEqual(score[1], max(scores)[1])

    def test_mm_errors_and_elevated_strata(self):
        m = Metrics()
        a, b = obj(), obj(position=[.003, .004, 0])
        a['elevated'] = True
        m.add([a], [b], .02)
        r = m.report()
        self.assertAlmostEqual(r['center_error_mm']['mean'], 5.)
        self.assertEqual(r['elevated_center_error_mm']['n'], 1)
        self.assertEqual(r['supported_center_error_mm']['n'], 0)
        self.assertEqual(r['latency_ms']['mean'], 20.)

    def test_class_matching_not_biased_by_wrong_high_iou(self):
        m = Metrics()
        m.add([obj('cube')], [obj('sphere'), obj('cube')], .01)
        self.assertEqual(m.report()['per_class']['cube']['tp'], 1)
        self.assertEqual(m.report()['per_class']['sphere']['fp'], 1)


class DatasetAndBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root/'raw'
        self.results = make_dataset(self.raw)
        self.meta = self.raw/'meta/test/000000.json'
        self.profile = ROOT/'config/scenes/playground_v2.toml'

    def test_raw_png_ids_not_class_ids(self):
        frame, truth, _ = load_sample(self.raw, self.raw/'meta/test/000001.json')
        self.assertEqual(truth[0]['label'], 'sphere')
        self.assertEqual(truth[0]['mask'].shape, frame.depth.shape)
        self.assertTrue(truth[0]['mask'].any())

    def test_original_frames_rgb_and_depth_roundtrip(self):
        frame, _, _ = load_sample(self.raw, self.meta)
        expected, _ = scene([Solid('cube', yaw=.4)])
        np.testing.assert_array_equal(frame.depth, expected.depth)
        np.testing.assert_array_equal(frame.rgb, expected.rgb)
        np.testing.assert_allclose(frame.world_from_camera, expected.world_from_camera)

    def test_png_crc_rejected(self):
        path = self.raw/'images/test/000000.png'
        data = bytearray(path.read_bytes())
        data[35] ^= 1
        path.write_bytes(data)
        with self.assertRaises(ValueError):
            read_png(path)

    def test_missing_depth_is_not_silently_skipped(self):
        (self.raw/'depth/test/000000.npz').unlink()
        with self.assertRaisesRegex(ValueError, 'save-depth'):
            load_sample(self.raw, self.meta)

    def test_corrupt_rgb_hash_rejected(self):
        path = self.raw/'images/test/000000.png'
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            load_sample(self.raw, self.meta)

    def test_incomplete_dataset_rejected(self):
        (self.raw/'progress.json').write_text(json.dumps(dict(complete=False)))
        with self.assertRaisesRegex(ValueError, 'not complete'):
            list_samples(self.raw, 'test')

    def test_visible_pixel_mismatch_rejected(self):
        d = json.loads(self.meta.read_text())
        d['objects'][0]['visible_pixels'] += 1
        self.meta.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, 'pixel count'):
            load_sample(self.raw, self.meta)

    def test_only_selected_split_is_read(self):
        (self.raw/'images/train').mkdir(parents=True)
        (self.raw/'images/train/bad.png').write_text('invalid')
        samples, _ = list_samples(self.raw, 'test')
        self.assertEqual(len(samples), 4)

    def test_v1_offline_matches_native_provider_without_confirmation(self):
        frame, _, _ = load_sample(self.raw, self.meta)
        cfg = load_scene_config(ROOT/'config/scenes/playground.toml')
        cfg = replace(cfg, perception=replace(cfg.perception, tracking=replace(cfg.perception.tracking, min_confirm_hits=1)))
        provider = build_perception_provider(FrameCamera(frame), cfg)
        observations = provider.observe()
        offline = run_v1(frame, cfg)
        self.assertEqual(len(offline), len(observations))
        self.assertEqual(offline[0]['label'], observations[0].class_name)
        np.testing.assert_allclose(offline[0]['position'], observations[0].pose.position)
        np.testing.assert_array_equal(offline[0]['mask'], provider.get_mask(observations[0].object_id))

    def test_v1_only_runs_without_any_vision_client(self):
        with patch('tools.perception_v2.benchmark.VisionClient', side_effect=AssertionError('must not instantiate')):
            with contextlib.redirect_stdout(io.StringIO()):
                r = evaluate(self.raw, self.profile, out=self.root/'v1', v1_only=True, previews=0)
        self.assertEqual(set(r['metrics']), {'v1'})
        self.assertEqual(r['frames'], 4)
        self.assertTrue((self.root/'v1/summary.md').is_file())

    def test_full_benchmark_protocol_and_reports_with_fixture_masks(self):
        outer = self
        calls = []
        class Client:
            def health(self): return INFO
            def predict(self, frame_id, rgb):
                calls.append((frame_id, rgb.shape))
                return outer.results[frame_id]
        with contextlib.redirect_stdout(io.StringIO()):
            r = evaluate(self.raw, self.profile, out=self.root/'comparison', client=Client())
        self.assertEqual(len(calls), 4)
        self.assertEqual(r['metrics']['v2_rgb']['named_target_f1'], 1.)
        for name in ('metrics.json', 'summary.md', 'per_class.csv', 'frames.csv', 'review.html'):
            self.assertTrue((self.root/'comparison'/name).is_file())
        self.assertEqual(r['metrics']['v2_rgb']['counts']['gt'], 3)
        self.assertTrue(r['worker']['test_double'])
        self.assertFalse(r['robot_filter_applied'])
        self.assertFalse(r['v2_settings']['robot_self_filter'])
        self.assertTrue(r['v2_settings_requested']['robot_self_filter'])
        self.assertTrue(any('NOT applied' in n for n in r['notes']))

    def test_worker_failure_does_not_turn_into_successful_partial_comparison(self):
        class Client:
            def health(self): raise RuntimeError('worker absent')
        with self.assertRaisesRegex(RuntimeError, 'worker absent'):
            evaluate(self.raw, self.profile, out=self.root/'comparison', client=Client())
        self.assertFalse((self.root/'comparison/metrics.json').exists())

    def test_gt_pose_changes_never_change_provider_predictions(self):
        frame, _, _ = load_sample(self.raw, self.meta)
        config = load_scene_config(self.profile)
        before = run_v1(frame, config)
        d = json.loads(self.meta.read_text())
        d['objects'][0]['position'] = [123., 456., 789.]
        self.meta.write_text(json.dumps(d))
        frame2, truth2, _ = load_sample(self.raw, self.meta)
        after = run_v1(frame2, config)
        np.testing.assert_allclose(before[0]['position'], after[0]['position'])
        np.testing.assert_array_equal(truth2[0]['position'], d['objects'][0]['position'])

    def test_existing_report_not_overwritten(self):
        out = self.root/'existing'
        out.mkdir()
        (out/'keep.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'new benchmark'):
            evaluate(self.raw, self.profile, out=out, v1_only=True)
        self.assertEqual((out/'keep.txt').read_text(), 'keep')

    def test_split_size_mismatch_detected(self):
        self.meta.unlink()
        with self.assertRaisesRegex(ValueError, 'incomplete split'):
            list_samples(self.raw, 'test')


if __name__ == '__main__':
    unittest.main(verbosity=2)
