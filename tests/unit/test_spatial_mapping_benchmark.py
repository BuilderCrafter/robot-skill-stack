"""Exact capture/replay inputs without simulator startup or a native scene."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap
from robot_skill_stack.world.spatial_mapping import robot_filter


SCRIPT = Path(__file__).resolve().parents[2]/'scripts/benchmark_spatial_mapping_performance.py'
SPEC = importlib.util.spec_from_file_location('spatial_performance_benchmark', SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class SpatialMappingBenchmarkTests(unittest.TestCase):
    def test_capture_roundtrip_preserves_original_buffers_and_moving_reset(self):
        mesh = robot_filter.TriangleMesh(np.array([[.4, 0., 0.], [.5, 0., 0.], [.4, .1, 0.]]),
                                         np.array([[0, 1, 2]]))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            capture = benchmark.CaptureRecorder(directory, {'commit': 'test'})
            for index in range(15):
                token, acquired = (index+1., index*.1), 10.+index*.1
                frame = DepthFrame('/World/A', token, np.full((8, 8), .2), np.eye(3),
                    np.eye(4), acquired, np.full((8, 8, 3), index, np.uint8))
                snapshot = robot_filter.RobotSnapshot((robot_filter.MeshInstance(mesh, np.eye(4)),),
                    token[1], token)
                capture.record((frame,), {frame.camera_id: snapshot},
                    {frame.camera_id: (np.array([0]), np.array([0]))},
                    {frame.camera_id: np.zeros((8, 8), bool)})
            capture.manifest['status'] = 'PASS'
            capture.flush()
            manifest, geometries, records = benchmark.load_capture(directory)
            self.assertEqual([record['index'] for record in records if record['clear_map_before']],
                             [0, 3, 6, 9, 12])
            self.assertEqual(len(geometries), 1)
            np.testing.assert_allclose(records[14]['frames'][0].token, (15., 1.4), rtol=0, atol=1e-14)
            self.assertAlmostEqual(records[14]['frames'][0].acquired_at, 11.4)
            np.testing.assert_array_equal(records[14]['frames'][0].rgb, 14)
            adapter = benchmark.RecordedRobotFilter(robot_filter, [mesh])
            adapter.record = records[14]
            result = adapter.masks(records[14]['frames'], pixels_by_camera=records[14]['selected'])
            self.assertTrue(adapter.selection_matches_capture)
            np.testing.assert_array_equal(result['/World/A'], records[14]['production_masks']['/World/A'])
            changed = directory/manifest['records'][0]['cameras'][0]['file']
            changed.write_bytes(changed.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'Changed frame input'):
                benchmark.load_capture(directory)

    def test_signatures_include_probabilities_history_and_timestamp(self):
        occupancy = OccupancyMap(GridSpec((0., 0., 0.), (.03, .03, .03)), clock=lambda: 10.)
        occupancy.integrate([], [[1, 1, 1]], timestamp=10., now=10.)
        query = occupancy.query_many([[.015, .015, .015]], now=10.)
        first = benchmark.snapshot_signature(occupancy.snapshot(now=10.), query)
        # Both updates classify OCCUPIED, but confidence/history must still differ.
        occupancy.integrate([], [[1, 1, 1]], timestamp=10.1, now=10.1)
        second = benchmark.snapshot_signature(occupancy.snapshot(now=10.1),
            occupancy.query_many([[.015, .015, .015]], now=10.1))
        self.assertEqual(first['arrays']['states'], second['arrays']['states'])
        self.assertNotEqual(first['arrays']['probabilities'], second['arrays']['probabilities'])
        self.assertNotEqual(first['arrays']['last_observed'], second['arrays']['last_observed'])
        self.assertNotEqual(first['timestamp'], second['timestamp'])

    def test_isolation_keeps_the_canonical_depth_frame_type(self):
        source = ('from robot_skill_stack.world.spatial_mapping.rgbd import RgbdMapper\n'
                  'from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame\n')
        rewritten = benchmark._rewrite(source)
        self.assertIn('baseline_spatial_mapping.world.spatial_mapping.rgbd', rewritten)
        self.assertIn('robot_skill_stack.world.perception_multiview.acquisition', rewritten)


if __name__ == '__main__':
    unittest.main()
