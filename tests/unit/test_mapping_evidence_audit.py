"""Diagnostic captures preserve the actual mapping computation and evidence."""
from dataclasses import dataclass, replace
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from robot_skill_stack.world.spatial_mapping import rgbd
from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractionSettings
from robot_skill_stack.world.spatial_mapping.runtime import MappingInput, MappingProcessor
from tests.support.mapping_evidence_audit import AuditedMappingProcessor
from tests.unit.test_spatial_mapping_runtime import gt, settings
from tests.unit.test_spatial_mapping_robot_filter import frame, snapshot


@dataclass
class Clock:
    value: float = 100.2
    def __call__(self):
        return self.value


class MappingEvidenceAuditTests(unittest.TestCase):
    def compare(self, expected, actual):
        self.assertEqual(expected.accepted, actual.accepted)
        self.assertEqual(expected.input_timestamp, actual.input_timestamp)
        self.assertEqual(expected.snapshot.timestamp, actual.snapshot.timestamp)
        self.assertEqual(expected.snapshot.revision, actual.snapshot.revision)
        for name in ('states', 'probabilities', 'observed', 'last_observed', 'expired'):
            np.testing.assert_array_equal(getattr(expected.snapshot, name), getattr(actual.snapshot, name))

    def test_full_ground_truth_capture_preserves_arrays_packet_and_pickled_input(self):
        with tempfile.TemporaryDirectory() as directory:
            options = dict(clock=Clock(), obstacle_settings=ObstacleExtractionSettings())
            audited = AuditedMappingProcessor(settings(), audit_dir=directory, capture_inputs=True, **options)
            # The normal ProcessWorker must be able to initialize its SDK-free
            # interpreter from the processor before any job has run.
            audited = pickle.loads(pickle.dumps(audited))
            payload = MappingInput(ground_truth=gt())
            actual = audited.compute(payload)
            expected = MappingProcessor(settings(), **options).compute(payload)
            self.compare(expected, actual)
            row = json.loads(next(Path(directory).glob('job-*.json')).read_text())
            self.assertEqual(row['capture_format'], 'uncompressed')
            self.assertEqual(row['capture_writer'], 'numpy.savez')
            with zipfile.ZipFile(Path(directory)/row['arrays_npz']) as archive:
                self.assertTrue(all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist()))
            self.assertEqual(row['packet']['timestamp'], actual.obstacles.timestamp)
            with (Path(directory)/row['input_pickle']).open('rb') as stream:
                self.assertEqual(pickle.load(stream).ground_truth.timestamp, 100.)
            with np.load(Path(directory)/row['arrays_npz']) as arrays:
                for name in ('states', 'probabilities', 'observed', 'last_observed', 'expired'):
                    np.testing.assert_array_equal(arrays[name], getattr(actual.snapshot, name))
            self.assertTrue(row['clock_reads'])
            self.assertIn('occupancy_replace', row['timings_seconds'])
            self.assertIn('snapshot', row['timings_seconds'])
            self.assertIn('obstacle_extract', row['timings_seconds'])

    def test_rgbd_records_actual_selected_mask_and_integrated_indices_without_changes(self):
        cfg = settings(source='rgbd', bounds_min=(-.2, -.2, 0.), bounds_max=(.2, .2, 1.5),
                       stride=1, max_rays=3, automatic_robot_filter=True, camera_prim_paths=('/A',))
        image = frame([[.98, 1., 1.02]], camera='/A')
        payload = MappingInput(frames=(image,), robot_snapshots={'/A': snapshot()})
        original = rgbd._traverse_rays
        with tempfile.TemporaryDirectory() as directory:
            processor = AuditedMappingProcessor(cfg, clock=Clock(), audit_dir=directory, capture_inputs=True)
            actual = processor.compute(payload)
            expected = MappingProcessor(cfg, clock=Clock()).compute(payload)
            self.compare(expected, actual)
            self.assertIs(rgbd._traverse_rays, original)
            row = json.loads(next(Path(directory).glob('job-*.json')).read_text())
            with np.load(Path(directory)/row['arrays_npz']) as arrays:
                np.testing.assert_array_equal(arrays['camera_0_selected_y'], [0, 0, 0])
                np.testing.assert_array_equal(arrays['camera_0_selected_x'], [0, 1, 2])
                np.testing.assert_array_equal(arrays['camera_0_automatic_mask'], [[False, True, True]])
                self.assertEqual(len(arrays['integrate_occupied_indices']), 1)
            self.assertEqual(row['diagnostics']['automatic_masked_selected'], 2)
            self.assertIn('robot_filter', row['timings_seconds'])
            self.assertIn('ray_traversal', row['timings_seconds'])
            self.assertIn('occupancy_integrate', row['timings_seconds'])

    def test_full_capture_limit_keeps_scalar_rows_and_timing_only_has_no_dense_files(self):
        for full in (False, True):
            with self.subTest(full=full), tempfile.TemporaryDirectory() as directory:
                processor = AuditedMappingProcessor(settings(), clock=Clock(), audit_dir=directory,
                    capture_inputs=full, max_captures=1)
                processor.compute(MappingInput(ground_truth=gt()))
                processor.compute(MappingInput(ground_truth=gt(stamp=100.1)))
                rows = [json.loads(line) for line in (Path(directory)/'jobs.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertEqual(len(list(Path(directory).glob('*-input.pkl'))), int(full))
                self.assertEqual(len(list(Path(directory).glob('*-arrays.npz'))), int(full))
                self.assertFalse(rows[1]['capture_selected'])
                self.assertTrue(all(row['accepted'] for row in rows))

    def test_capture_write_error_does_not_replace_mapping_result_or_leave_global_hook(self):
        original = rgbd._traverse_rays
        processor = AuditedMappingProcessor(settings(), clock=Clock(), audit_dir='/unused')
        errors = io.StringIO()
        with redirect_stderr(errors), patch.object(processor, '_persist', side_effect=OSError('diagnostic disk full')):
            result = processor.compute(MappingInput(ground_truth=gt()))
        self.assertIn('diagnostic disk full', errors.getvalue())
        self.assertTrue(result.accepted)
        self.assertIs(rgbd._traverse_rays, original)
        self.assertIsNone(processor._active_row)


if __name__ == '__main__':
    unittest.main()
