"""Saved-evidence scoring distinguishes hidden objects from failed reacquisition."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_multiview_robot import analyze


class RobotAnalysisTests(unittest.TestCase):
    def fixture(self, path):
        scenes, samples, rows, detections = [], [], [], []
        # Two genuine tracker-confirmation frames, an established ID, full
        # occlusion, sparse reappearance, and a measured correct recovery.
        phases = [('clear', [12, 12, 12]), ('hidden', [0, 0]), ('reveal', [12, 2, 12])]
        index = 0
        for phase, counts in phases:
            frames = []
            for frame_index, count in enumerate(counts):
                stamp = float(index+1)
                confirmation = phase == 'clear' and frame_index < 2
                detected = (phase == 'clear' and not confirmation) or (phase == 'reveal' and frame_index == 2)
                frames.append(dict(index=frame_index, objects=[dict(name='target')],
                    files=[], visible_surface_points={'0': {'target': count}, '1': {'target': count}}))
                samples.append(dict(phase=phase, frame_index=frame_index, fresh_capture=True,
                    batch_timestamp=stamp, captured_at=stamp, delivered_at=stamp+.2,
                    source_health='fresh', source_error=None, delivery_ms=201., acquisition_to_delivery_ms=200.,
                    physics_step=index*8, simulation_time_s=index*.1,
                    diagnostics=dict(total_ms=190., fitting_ms=140., numerical_fits=2,
                        fit_budget=24, fit_budget_exhaustions=0,
                        synchronization=dict(state='synchronized', skew_seconds=0., tokens={'A': [index], 'B': [index]}))))
                rows.append(dict(trial=phase, frame_index=frame_index, object='target',
                    detected=detected, correct=detected, predicted='sphere' if detected else 'missing',
                    object_id='stable_7' if detected else None))
                detections.append(dict(trial=phase, frame_index=frame_index, confirmation_frame=confirmation,
                    false_positives=0, false_splits=0, mixed_accepted_clouds=0, duplicate_ids=0, identity_switches=0))
                index += 1
            scenes.append(dict(key=phase, needs_confirmation=phase == 'clear', frames=frames))
        values = dict(capture_manifest=dict(scenes=scenes), samples=samples, rows=rows,
                      detections=detections, result=dict(status='PASS'))
        for name, value in values.items():
            (path/f'{name}.json').write_text(json.dumps(value))
        return values

    def test_visibility_confirmation_and_reacquisition_use_measured_reveal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.fixture(path)
            result, frames, objects, visibility = analyze(path)
        self.assertEqual(result['samples'], 8)
        self.assertEqual(result['expected_after_confirmation'], 6)
        self.assertEqual(result['correct_after_confirmation'], 2)
        self.assertEqual(result['missing_fully_hidden'], 2)
        self.assertEqual(result['missing_visible'], 1)
        self.assertEqual(result['missing_low_density'], 1)
        self.assertTrue(result['fully_hidden_both_cameras_established'])
        self.assertEqual(result['successful_correct_reacquisitions'], 1)
        event = result['occlusion_events'][0]
        self.assertTrue(event['identity_preserved'])
        self.assertAlmostEqual(event['correct_reacquisition_from_first_revisible_capture_ms'], 2200.)
        self.assertFalse(result['timestamp_issues'])
        self.assertEqual(result['physics_step_intervals']['samples'], 7)
        self.assertEqual(len(visibility), 16)

    def test_hidden_detection_cannot_finish_visible_reacquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            evidence = self.fixture(path)
            # A scorer claiming an accepted hidden object must be retained and
            # flagged, without moving the measured visible recovery earlier.
            evidence['capture_manifest']['scenes'][2]['frames'][1]['visible_surface_points'] = {
                '0': {'target': 0}, '1': {'target': 0}}
            evidence['rows'][6].update(detected=True, correct=True, predicted='sphere', object_id='stable_7')
            for name in ('capture_manifest', 'rows'):
                (path/f'{name}.json').write_text(json.dumps(evidence[name]))
            result, *_ = analyze(path)
        self.assertEqual(result['detected_while_fully_hidden'], 1)
        self.assertAlmostEqual(result['occlusion_events'][0]['correct_reacquisition_from_first_revisible_capture_ms'], 2200.)

    def test_reused_source_tokens_and_missing_scoring_evidence_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            evidence = self.fixture(path)
            evidence['samples'][3]['diagnostics']['synchronization']['tokens'] = {'A': [2], 'B': [2]}
            evidence['rows'].pop()
            for name in ('samples', 'rows'):
                (path/f'{name}.json').write_text(json.dumps(evidence[name]))
            result, *_ = analyze(path)
        self.assertEqual(len(result['timestamp_issues']), 2)
        self.assertEqual(result['artifact_integrity_issues'][0]['kind'], 'capture_truth_missing_scoring_row')
        self.assertEqual(result['successful_correct_reacquisitions'], 0)
        self.assertTrue(result['occlusion_events'][0]['censored_without_correct_reacquisition'])


if __name__ == '__main__':
    unittest.main()
