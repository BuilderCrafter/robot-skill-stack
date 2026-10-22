"""Exact saved-evidence replay, baseline cohorts and native-checkpoint guards.

The fixture uses synthetic calibrated frames and an explicit offline clock. It
does not measure live freshness or native performance. Actual mapper-selected
pixels and integrate arguments are saved, rather than reconstructing vote math.
"""
from dataclasses import replace
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.world.spatial_mapping.rgbd import RgbdMapper
from tests.unit.test_spatial_mapping_rgbd import ray


SCRIPT = Path(__file__).resolve().parents[2]/'scripts/evaluate_spatial_mapping_reliability.py'
SPEC = importlib.util.spec_from_file_location('spatial_reliability_replay', SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class ReliabilityReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline, cls.baseline_sha = replay.baseline_module()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.manifest = self.capture()

    def write_manifest(self):
        (self.directory/'capture_manifest.json').write_text(json.dumps(self.manifest))

    def capture(self):
        settings = dict(bounds_min=[0., 0., 0.], bounds_max=[.04, .02, .02],
                        voxel_size=.01, stale_after=1., evidence_ttl=5.)
        clock = [100.]
        occupancy = self.baseline.OccupancyMap(self.baseline.GridSpec(
            settings['bounds_min'], settings['bounds_max']), clock=lambda: clock[0])
        mapper = RgbdMapper(occupancy, stride=1,
            expected_camera_ids=('A', 'B'), clock=lambda: clock[0])
        records = []
        for index in range(14):
            clear = index in (0, 13)
            if clear:
                occupancy.clear()
            entering = 10 <= index < 13
            acquired = 100.+index*.25
            integration_now, snapshot_now = acquired+.125, acquired+.15
            clock[0] = integration_now
            endpoint = (.015 if entering else .035, .005, .005)
            first = ray((.005, .005, .005), endpoint,
                camera='A', number=index+1, acquired=acquired)
            first = replace(first, depth=np.repeat(first.depth, 2, axis=1))
            second = ray((.005, .015, .005), (.035, .015, .005),
                camera='B', number=index+1, acquired=acquired+.001)
            frames = (first, second)
            masks = {'A': np.array([[False, True]]), 'B': np.zeros((1, 1), bool)}
            pixels, evidence = {}, {}

            def record_masks(current, selected):
                pixels.update(selected)
                return masks

            original = occupancy.integrate

            def record_evidence(free_indices, occupied_indices, **kwargs):
                evidence.update(free_indices=np.array(free_indices, copy=True),
                    occupied_indices=np.array(occupied_indices, copy=True), **kwargs)
                return original(free_indices, occupied_indices, **kwargs)

            with patch.object(occupancy, 'integrate', side_effect=record_evidence):
                self.assertTrue(mapper.update(frames, now=integration_now,
                    mask_provider=record_masks), mapper.diagnostics)
            evidence.update(snapshot_now=snapshot_now,
                evaluated_indices=np.array([[1, 0, 0], [2, 0, 0], [3, 1, 0]]),
                excluded_indices=np.array([[3, 1, 0]]))
            evidence_file = f'evidence_{index:03d}.npz'
            np.savez(self.directory/evidence_file, **evidence)
            transform = np.eye(4)
            transform[:3, 3] = (.015, .005, .005)
            shapes = [dict(name='/Fixture/Target', kind='box', dimensions=[.01]*3,
                           world_from_local=transform.tolist())] if entering else []
            record = dict(index=index, episode='entry', phase='hit' if entering else 'clear',
                clear_map_before=clear, evidence_file=evidence_file, cameras=[],
                exposure_shapes_by_camera={'A': shapes, 'B': shapes})
            for frame in frames:
                filename = f'frame_{index:03d}_{frame.camera_id}.npz'
                y, x = pixels[frame.camera_id]
                labels = np.array([[11, 99]]) if frame.camera_id == 'A' else np.array([[22]])
                paths = {'11': '/Fixture/Target' if entering else '/Fixture/BackdropA',
                         '22': '/Fixture/BackdropB', '99': '/Robot/Masked'}
                np.savez(self.directory/filename, depth=frame.depth,
                    intrinsics=frame.intrinsics, world_from_camera=frame.world_from_camera,
                    token=frame.token, acquired_at=frame.acquired_at,
                    selected_y=y, selected_x=x, automatic_mask=masks[frame.camera_id],
                    instance_ids=labels, instance_paths_json=np.asarray(json.dumps(paths)))
                record['cameras'].append(dict(camera_id=frame.camera_id, file=filename,
                    token=frame.token, acquired_at=frame.acquired_at))
            if index in (12, 13):
                snapshot = occupancy.snapshot(now=snapshot_now)
                filename = f'snapshot_{index:03d}.npz'
                np.savez(self.directory/filename,
                    **{key: getattr(snapshot, key) for key in
                       ('states', 'probabilities', 'observed', 'last_observed', 'expired')},
                    timestamp=snapshot.timestamp, bounds_min=snapshot.spec.bounds_min,
                    bounds_max=snapshot.spec.bounds_max, voxel_size=snapshot.spec.voxel_size)
                record['snapshot_file'] = filename
            records.append(record)
        manifest = dict(status='PASS', settings=settings, records=records)
        (self.directory/'capture_manifest.json').write_text(json.dumps(manifest))
        return manifest

    def evaluate_with(self, factory=None):
        # Inject an explicit comparator to keep these replay tests independent of
        # any subsequent production policy change. The real frozen baseline is
        # still loaded through git by evaluate(), including its checkpoint guard.
        comparator = types.SimpleNamespace(GridSpec=self.baseline.GridSpec,
            OccupancyState=self.baseline.OccupancyState,
            OccupancyMap=factory or self.baseline.OccupancyMap,
            __file__=replay.current.__file__)
        with patch.object(replay, 'current', comparator):
            return replay.evaluate(self.directory)

    def test_unchanged_replay_matches_native_arrays_and_preserves_original_timeline(self):
        current_maps = []

        def record_current(*args, **kwargs):
            occupancy = self.baseline.OccupancyMap(*args, **kwargs)
            current_maps.append(occupancy)
            return occupancy

        result = self.evaluate_with(record_current)
        self.assertFalse(result['timing_claims'])
        self.assertFalse(result['timestamps_rewritten'])
        self.assertEqual(result['baseline_occupancy_sha256'], self.baseline_sha)
        for index, row in enumerate(result['measurements']):
            self.assertEqual(row['original_timestamp'], 100.+index*.25)
            self.assertEqual(row['integration_now'], 100.+index*.25+.125)
            self.assertEqual(row['policies']['baseline'], row['policies']['current'])
        self.assertEqual(len(result['native_baseline_snapshot_checks']), 2)
        self.assertTrue(all(all(row['fields_equal'].values())
            for row in result['native_baseline_snapshot_checks']))
        self.assertEqual(len(current_maps), 1)
        final_record = self.manifest['records'][-1]
        with np.load(self.directory/final_record['evidence_file'], allow_pickle=False) as evidence:
            current_snapshot = current_maps[0].snapshot(now=float(evidence['snapshot_now']))
        final_checkpoint = self.directory/final_record['snapshot_file']
        with np.load(final_checkpoint, allow_pickle=False) as native:
            for field in ('states', 'probabilities', 'observed', 'last_observed', 'expired'):
                np.testing.assert_array_equal(getattr(current_snapshot, field), native[field])
            self.assertEqual(current_snapshot.timestamp, float(native['timestamp']))
        self.assertEqual(result['baseline_still_free_hit_events'], 3)
        self.assertEqual(result['current_still_free_on_same_events'], 3)
        expected_files = {name for record in self.manifest['records'] for name in
            [record['evidence_file'], *(camera['file'] for camera in record['cameras']),
             *([record['snapshot_file']] if record.get('snapshot_file') else [])]}
        self.assertEqual(set(result['input_sha256']), expected_files)
        for name, digest in result['input_sha256'].items():
            self.assertEqual(digest, hashlib.sha256((self.directory/name).read_bytes()).hexdigest())

    def test_direct_hit_comparison_keeps_the_baseline_cohort_when_candidate_changes_state(self):
        stronger_hits = functools.partial(self.baseline.OccupancyMap, hit_log_odds=4.)
        result = self.evaluate_with(stronger_hits)
        hits = [row for row in result['measurements'] if row['phase'] == 'hit']
        self.assertEqual([row['baseline_still_free_after_hit'] for row in hits], [1, 1, 1])
        self.assertEqual([row['current_states_on_baseline_still_free_hits']['unknown']
                          for row in hits], [1, 0, 0])
        self.assertEqual([row['current_states_on_baseline_still_free_hits']['occupied']
                          for row in hits], [0, 1, 1])
        self.assertEqual(result['baseline_still_free_hit_events'], 3)
        self.assertEqual(result['current_still_free_on_same_events'], 0)

    def test_saved_pixel_masks_and_camera_labels_define_the_retained_surface_denominator(self):
        record = self.manifest['records'][10]
        points, labels, cameras = replay.surface_inputs(self.directory, record)
        np.testing.assert_allclose(points, [[.015, .005, .005], [.035, .015, .005]])
        self.assertEqual(labels, ['/Fixture/Target', '/Fixture/BackdropB'])
        self.assertEqual(cameras, ['A', 'B'])
        result = self.evaluate_with()
        events = result['measurements'][10]['policies']['baseline']['selected_retained_surface_events']
        self.assertEqual(events['input_events'], 2)
        self.assertNotIn('/Robot/Masked', events['per_label'])
        metrics = result['measurements'][10]['policies']['baseline']['physical_center_metrics']
        self.assertEqual(metrics['exposure_references'], 2)
        self.assertEqual(metrics['input_voxels'], 3)
        self.assertEqual(metrics['excluded_voxels'], 1)
        self.assertEqual(metrics['false_free']['count'], 1)

    def test_tampered_checkpoint_state_history_or_metadata_is_rejected(self):
        path = self.directory/self.manifest['records'][12]['snapshot_file']
        with np.load(path, allow_pickle=False) as saved:
            original = {key: saved[key] for key in saved.files}
        for field in ('states', 'probabilities', 'observed', 'last_observed', 'expired',
                      'timestamp', 'bounds_min', 'bounds_max', 'voxel_size'):
            with self.subTest(field=field):
                altered = {key: value.copy() for key, value in original.items()}
                flat = altered[field].reshape(-1)
                selected = 0 if field not in ('last_observed',) else int(np.flatnonzero(np.isfinite(flat))[0])
                flat[selected] = not flat[selected] if flat.dtype == bool else flat[selected]+1
                np.savez(path, **altered)
                with self.assertRaisesRegex(AssertionError, 'differs from original native capture'):
                    self.evaluate_with()
        np.savez(path, **original)

    def test_missing_recorded_clear_is_detected_by_the_next_native_checkpoint(self):
        self.manifest['records'][13]['clear_map_before'] = False
        self.write_manifest()
        with self.assertRaisesRegex(AssertionError, 'differs from original native capture'):
            self.evaluate_with()

    def test_partial_or_pre_evidence_capture_cannot_be_reported_as_a_valid_replay(self):
        self.manifest['status'] = 'RUNNING'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'complete PASS'):
            self.evaluate_with()
        self.manifest['status'] = 'PASS'
        del self.manifest['records'][0]['evidence_file']
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'predates exact integrate-evidence'):
            self.evaluate_with()


if __name__ == '__main__':
    unittest.main()
