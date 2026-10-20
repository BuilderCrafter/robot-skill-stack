"""Final research metrics: axis correspondence, retained failures and source clocks."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from robot_skill_stack.world.model.provider import ObservationBatch
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from scripts.benchmark_multiview_final_perception import balanced_pairs, completion_gate, layout_hash, verify_freeze
from scripts.benchmark_multiview_final_perception import annotate_visibility, parser, run
from scripts.benchmark_multiview_robustness import save_frames
from tests.support.multiview_final_evaluation import (
    PerceptionSession, box_orientation, geometry_row, latest_scene_summary, score_frame, settle_physics,
    summarize, temporal_recovery,
)
from tests.support.multiview_iteration2 import box, object_payload, sphere


def observation(target, geometry=None, object_id='id_1'):
    geometry = geometry or PrimitiveGeometry(target.shape, box_size=target.size.copy(), yaw=target.yaw,
        radius=target.radius, metadata={'parameter_observable': True})
    return ObjectObservation(object_id=object_id, class_name=target.shape, pose=Pose(target.center.copy()),
        size=target.size.copy(), geometry=geometry, graspable=True, timestamp=123.)


class ControlledSource:
    def __init__(self, rig, scene):
        self.rig, self.calls, self.diagnostics = rig, 0, {}

    def observe(self):
        frames = self.rig.capture()
        if frames is None:
            return ObservationBatch()
        self.calls += 1
        self.last_frames = frames
        target = box('target', (.45, 0.), (.06, .03, .05))
        obs = observation(target)
        obs.timestamp = frames[0].acquired_at
        return ObservationBatch([obs] if self.calls >= 3 else [], frames[0].acquired_at)

    def get_point_cloud(self, object_id):
        return None

    def close(self):
        pass


class FinalPerceptionEvaluationTests(unittest.TestCase):
    def test_rectangular_yaw_aligns_side_correspondence_and_retains_180_symmetry(self):
        target = box('rectangle', (.45, 0.), (.09, .04, .05), .3)
        equivalent = PrimitiveGeometry('cube', box_size=np.array([.04, .09, .05]), yaw=.3+np.pi/2)
        wrong = PrimitiveGeometry('cube', box_size=target.size, yaw=.3+np.pi/2)
        self.assertAlmostEqual(box_orientation(target, equivalent)['yaw_error_deg'], 0.)
        self.assertAlmostEqual(box_orientation(target, wrong)['yaw_error_deg'], 90.)
        self.assertEqual(box_orientation(target, wrong)['yaw_symmetry_degrees'], 180)

    def test_square_and_near_square_have_distinct_truth_symmetry_and_ambiguity(self):
        target = box('square', (.45, 0.), (.05, .05, .04), .2)
        fitted = PrimitiveGeometry('cube', box_size=target.size, yaw=.2+np.pi/2)
        self.assertAlmostEqual(box_orientation(target, fitted)['yaw_error_deg'], 0.)
        self.assertEqual(box_orientation(target, fitted)['yaw_symmetry_degrees'], 90)
        near = box('near', (.45, 0.), (.05, .0505, .04), .2)
        result = box_orientation(near, fitted)
        self.assertEqual(result['yaw_symmetry_degrees'], 180)
        self.assertFalse(result['yaw_precision_eligible'])
        self.assertIsNotNone(result['yaw_raw_major_axis_error_deg'])
        self.assertEqual(result['yaw_ambiguity'], 'near_square_truth')

    def test_box_height_is_not_hidden_by_sorting_all_three_dimensions(self):
        target = box('rectangle', (.45, 0.), (.09, .04, .05))
        fitted = PrimitiveGeometry('cube', box_size=np.array([.09, .05, .04]), yaw=0.)
        row = geometry_row(target, observation(target, fitted))
        self.assertAlmostEqual(row['historical_sorted_xyz_error_mm'], 0.)
        self.assertAlmostEqual(row['height_error_mm'], 10.)
        self.assertAlmostEqual(row['dimension_error_mm'], 10.)

    def test_sphere_radius_has_signed_bias_and_no_orientation_metric(self):
        target = sphere('sphere', (.45, 0.), .03)
        fitted = PrimitiveGeometry('sphere', radius=.029)
        row = geometry_row(target, observation(target, fitted))
        self.assertAlmostEqual(row['radius_bias_mm'], -1.)
        self.assertAlmostEqual(row['diameter_error_mm'], 2.)
        self.assertFalse(row['yaw_precision_eligible'])
        self.assertIsNone(row['yaw_error_deg'])

    def test_real_source_adapter_keeps_original_tokens_stamps_and_persistent_identity(self):
        target = box('target', (.45, 0.), (.06, .03, .05))
        config = SimpleNamespace(scene=None)
        session = PerceptionSession(ControlledSource, config, method='controlled')
        for index in range(4):
            frame = DepthFrame('A', (index+1, .1*(index+1)), np.ones((1, 1)), np.eye(3), np.eye(4),
                               100.+index*.2, None)
            session.consume((frame,), (target,), index=index, trial='scene', temporal=False)
        self.assertTrue(all(item['passed'] for item in session.contracts))
        self.assertEqual(session.source.calls, 4)
        self.assertEqual(session.source.last_frames[0].token, (4, .4))
        self.assertEqual(session.rows[-1]['observation_timestamp'], 100.6)
        self.assertEqual(session.rows[-1]['object_id'], 'id_1')
        summary = summarize(session.rows, session.samples, session.detections, session.contracts)
        self.assertEqual(summary['expected'], 2)
        self.assertEqual(summary['correct'], 2)

    def test_failure_rows_remain_and_engineering_targets_do_not_control_completion(self):
        target = box('target', (.45, 0.), (.06, .03, .05))
        session = PerceptionSession(ControlledSource, SimpleNamespace(scene=None), method='controlled')
        session.retain_failure((target,), index=2, error='deliberate', trial='scene')
        summary = summarize(session.rows, session.samples, session.detections, session.contracts)
        self.assertEqual(summary['expected'], 1)
        self.assertEqual(summary['missing'], 1)
        self.assertEqual(summary['runtime_error_rows'], 1)
        self.assertFalse(summary['engineering_targets']['correct_at_least_95pct'])
        passing_contracts = {'one': {'source_contracts_passed': True}}
        self.assertTrue(completion_gate(errors=[], hashes_stable=True, audits=[{'valid': True}],
                                        summaries=passing_contracts, capture_only=False))
        self.assertFalse(completion_gate(errors=[], hashes_stable=True, audits=[{'valid': False}],
                                         summaries=passing_contracts, capture_only=False))

    def test_unmatched_mixed_clouds_keep_class_and_graspability(self):
        targets = (sphere('one', (.4, 0.), .02), sphere('two', (.5, 0.), .02))
        obs = observation(sphere('merged', (.45, 0.), .04), object_id='merged')
        cloud = np.repeat(np.array([[.4, 0., .04], [.5, 0., .04]]), 10, axis=0)
        rows, counts, _, raw = score_frame(targets, [obs], [cloud])
        self.assertEqual(counts['missing'], 2)
        self.assertEqual(counts['known_mixed'], 1)
        self.assertEqual(counts['graspable_mixed'], 1)
        self.assertIsNone(raw[0]['matched_truth_index'])
        self.assertEqual(raw[0]['shape'], 'sphere')
        self.assertTrue(raw[0]['graspable'])

    def test_freeze_covers_every_package_file_and_exact_selected_layout(self):
        @dataclass
        class Layout:
            name: str
            eyes: tuple
        layouts = [Layout('selected', ((1., 0., 1.),))]
        hashes = {'robot_skill_stack/world/perception_multiview/a.py': 'abc', 'harness': 'def'}
        with tempfile.TemporaryDirectory() as directory, patch(
                'scripts.benchmark_multiview_final_perception.file_hashes', return_value=hashes):
            path = Path(directory)/'freeze.json'
            path.write_text(json.dumps(dict(sha256=hashes, layout_sha256=layout_hash(layouts))))
            self.assertEqual(verify_freeze(path, layouts)['layout_sha256'], layout_hash(layouts))
            with self.assertRaisesRegex(ValueError, 'layout'):
                verify_freeze(path, [Layout('different', layouts[0].eyes)])
            path.write_text(json.dumps(dict(sha256={}, layout_sha256=layout_hash(layouts))))
            with self.assertRaisesRegex(ValueError, 'omits package files'):
                verify_freeze(path, layouts)

    def test_layout_order_balanced_and_latest_scene_count_does_not_repeat_static(self):
        layouts = ['A', 'B', 'C']
        pairs = balanced_pairs(list(range(3)), layouts)
        self.assertEqual(pairs, [(0, 'A'), (0, 'B'), (0, 'C'), (1, 'B'), (1, 'C'), (1, 'A'),
                                 (2, 'C'), (2, 'A'), (2, 'B')])
        target = box('target', (.45, 0.), (.06, .03, .05))
        session = PerceptionSession(ControlledSource, SimpleNamespace(scene=None), method='controlled')
        for index in range(3):
            session.retain_failure((target,), index=index+2, error='retained', trial='one_scene', temporal=False)
        latest = latest_scene_summary(session.rows, session.samples, session.detections, session.contracts)
        self.assertEqual(latest['physical_static_scenes'], 1)
        self.assertEqual(latest['expected'], 1)

    def test_saved_capture_entry_exports_all_trials_without_simulator_import(self):
        # Controlled arrays verify the replay protocol, not physical accuracy.
        target = box('target', (.45, 0.), (.06, .03, .05))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, output = root/'capture', root/'replay'
            capture.mkdir()
            frames = []
            for index in range(4):
                pair = tuple(DepthFrame(camera, (index+1, .1*(index+1)), np.ones((1, 1)),
                    np.eye(3), np.eye(4), 100.+index*.2, None) for camera in ('A', 'B'))
                frames.append(dict(index=index, files=save_frames(capture, 'fixture', index, pair),
                                   objects=[object_payload(target)], phase='static'))
            record = dict(key='coarse_protocol__test', scene_key='coarse_protocol', name='protocol',
                split='coarse', seed=1, temporal=False, note='Controlled unit fixture', requirements={},
                objects=[object_payload(target)], layout=dict(name='test'), frames=frames)
            (capture/'capture_manifest.json').write_text(json.dumps(dict(scenes=[record])))
            args = parser().parse_args(['--replay', str(capture), '--result', str(output/'result.json'),
                                         '--artifacts-dir', str(output)])
            with patch('scripts.benchmark_multiview_final_perception.method_specs',
                       return_value={'controlled': (ControlledSource, 'fused')}):
                self.assertEqual(run(args), 0)
            result = json.loads((output/'result.json').read_text())
            rows = json.loads((output/'rows.json').read_text())
            raw = json.loads((output/'observations.json').read_text())
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['metrics']['summary']['test']['controlled']['expected'], 2)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(raw), 2)
        self.assertEqual(raw[-1]['timestamp'], 100.6)

    def test_strict_one_view_requires_zero_hidden_points_and_recovery_uses_saved_clock(self):
        target = box('target', (.45, 0.), (.06, .03, .05))
        rows, samples = [], []
        for index, correct, counts in [(2, True, [30, 30]), (3, False, [0, 0]),
                                       (4, False, [0, 12]), (5, True, [1, 24]), (6, False, [2, 3])]:
            row = geometry_row(target, observation(target) if correct else None)
            row.update(method='current', layout='test', trial='temporal', frame_index=index,
                       phase='covered' if index < 5 else 'revealed', scored=True, temporal=True)
            annotate_visibility([row], dict(objects=[dict(object='target',
                cameras=[dict(raw_points=count) for count in counts], sufficiently_visible=min(counts) >= 8)]))
            rows.append(row)
            samples.append(dict(trial='temporal', frame_index=index, original_acquired_at=100.+index*.1))
        self.assertTrue(rows[2]['strict_true_one_view'])
        self.assertFalse(rows[3]['strict_true_one_view'])
        self.assertTrue(rows[3]['effective_one_view'])
        events = temporal_recovery(rows, samples)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['fully_hidden_frames'], [3])
        self.assertEqual(events[0]['first_revisible_frame'], 4)
        self.assertAlmostEqual(events[0]['saved_capture_reacquisition_ms'], 100.)
        self.assertTrue(events[0]['identity_preserved'])
        self.assertTrue(events[1]['censored'])
        self.assertEqual(events[1]['low_density_frames'], [6])

    def test_physics_settling_rejects_cumulative_drift_then_requires_full_stable_window(self):
        counter = [0]
        def step():
            counter[0] += 1
        def transforms():
            pose = np.eye(4)
            pose[0, 3] = min(counter[0], 80)*.00002
            return {'/Franka/link': pose.tolist()}
        def bounds():
            offset = min(counter[0], 80)*.00002
            return [[offset, 0., 0.], [offset+1., 1., 1.]]
        result = settle_physics(step, transforms, bounds)
        self.assertTrue(result['stable'])
        self.assertGreaterEqual(result['steps'], 135)
        self.assertEqual(result['consecutive_stable_steps'], 60)
        self.assertGreater(result['window_resets'], 0)
        self.assertLessEqual(result['final_window_max_translation_m'], .0001)
        self.assertNotEqual(result['initial_transforms'], result['final_transforms'])

    def test_physics_settling_checks_aabb_and_times_out_without_faking_stability(self):
        counter = [0]
        def step():
            counter[0] += 1
        transform = lambda: {'/Franka/link': np.eye(4).tolist()}
        def bounds():
            offset = min(counter[0], 80)*.00001
            return [[offset, 0., 0.], [offset+1., 1., 1.]]
        result = settle_physics(step, transform, bounds)
        self.assertTrue(result['stable'])
        self.assertEqual(result['steps'], 180)
        self.assertEqual(result['window_resets'], 2)
        counter[0] = 0
        def moving():
            pose = np.eye(4)
            pose[0, 3] = counter[0]*.001
            return {'/Franka/link': pose.tolist()}
        timeout = settle_physics(step, moving, bounds, max_steps=20, stable_steps=6)
        self.assertFalse(timeout['stable'])
        self.assertEqual(timeout['steps'], 20)


if __name__ == '__main__':
    unittest.main()
