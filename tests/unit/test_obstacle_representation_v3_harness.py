"""Research activation changes prepared packets without changing RGB-D evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.integrations.isaac.spatial_mapping.config import SpatialMappingConfig
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractionSettings
from robot_skill_stack.world.spatial_mapping.runtime import MappingInput, MappingProcessor
from scripts.benchmark_obstacle_representation_v3 import (
    AuditedRepresentationMappingProcessor, RepresentationInput, RepresentationMappingProcessor,
    RepresentationSpec, configure_extractor, progress_windows, reference_match, shared_input)


class ObstacleRepresentationV3HarnessTests(unittest.TestCase):
    def configuration(self):
        return SpatialMappingConfig(source='rgbd', bounds_min=(0., 0., 0.),
            bounds_max=(.08, .08, .08), automatic_robot_filter=False)

    def payload(self, index):
        transform = np.eye(4)
        transform[:3, 3] = [.015, .015, .005]
        frame = DepthFrame('camera', (index+1., index*.01), np.full((8, 8), .05),
            np.array([[100., 0., 4.], [0., 100., 4.], [0., 0., 1.]]), transform, 10.+index*.01)
        return MappingInput(frames=(frame,))

    def test_owner_rewrap_shares_owned_depth_and_preserves_source_identity(self):
        payload = self.payload(0)
        wrapped = shared_input(payload, active=True, capture=True)
        self.assertIsInstance(wrapped, RepresentationInput)
        self.assertIs(wrapped.frames[0].depth, payload.frames[0].depth)
        self.assertIs(wrapped.frames[0].intrinsics, payload.frames[0].intrinsics)
        self.assertEqual(wrapped.frames[0].token, payload.frames[0].token)
        self.assertEqual(wrapped.frames[0].acquired_at, payload.frames[0].acquired_at)
        self.assertTrue(wrapped.activate_representation)

    def test_baseline_is_geometry_identical_before_and_after_activation(self):
        now = [10.1]
        settings = self.configuration()
        obstacles = ObstacleExtractionSettings()
        original = MappingProcessor(settings, clock=lambda: now[0], obstacle_settings=obstacles)
        research = RepresentationMappingProcessor(settings, clock=lambda: now[0], obstacle_settings=obstacles)
        for index in range(6):
            now[0] = 10.1+index*.01
            payload = self.payload(index)
            first = original.compute(payload)
            second = research.compute(shared_input(payload, active=index >= 3))
            for key in ('states', 'probabilities', 'last_observed', 'observed', 'expired'):
                np.testing.assert_array_equal(getattr(first.snapshot, key), getattr(second.snapshot, key))
            self.assertEqual(first.obstacles.boxes, second.obstacles.boxes)
            self.assertEqual(first.snapshot.timestamp, second.snapshot.timestamp)
            self.assertEqual(second.diagnostics['representation_v3']['active'], index >= 3)
            with self.assertRaises(TypeError):
                second.obstacles.diagnostics['representation_v3']['active'] = False

    def test_candidate_waits_for_activation_and_does_not_reset_temporal_map(self):
        now = [10.1]
        spec = RepresentationSpec('axis_best', 'diversity', 64, 'test')
        processor = RepresentationMappingProcessor(self.configuration(), representation_spec=spec,
            clock=lambda: now[0], obstacle_settings=ObstacleExtractionSettings())
        for index in range(5):
            now[0] = 10.1+index*.01
            completed = processor.compute(shared_input(self.payload(index), active=index == 4))
            if index < 4:
                self.assertEqual(type(processor._extractor.extractor).__name__, 'ObstacleExtractor')
        self.assertEqual(type(processor._extractor.extractor).__name__, 'ExperimentalObstacleExtractor')
        self.assertEqual(completed.snapshot.revision, 5)
        self.assertEqual(completed.diagnostics['representation_v3']['candidate_id'], 'test')

    def test_forensic_capture_selects_phase_and_retains_actual_snapshot_arrays(self):
        now = [10.1]
        with tempfile.TemporaryDirectory() as temporary:
            processor = AuditedRepresentationMappingProcessor(self.configuration(), clock=lambda: now[0],
                obstacle_settings=ObstacleExtractionSettings(), audit_dir=temporary,
                capture_inputs=True, max_captures=2)
            expected = []
            for index in range(5):
                now[0] = 10.1+index*.01
                result = processor.compute(shared_input(self.payload(index), active=index >= 2, capture=index >= 2))
                if 2 <= index <= 3:
                    expected.append(result.snapshot.states.copy())
            rows = [json.loads(line) for line in Path(temporary, 'jobs.jsonl').read_text().splitlines()]
            selected = [row for row in rows if row['capture_selected']]
            self.assertEqual(len(selected), 2)
            self.assertEqual([row['job'] for row in selected], [3, 4])
            for row, states in zip(selected, expected):
                with np.load(Path(temporary, row['arrays_npz'])) as arrays:
                    np.testing.assert_array_equal(arrays['states'], states)
                self.assertTrue(row['packet']['diagnostics']['representation_v3']['active'])

    def test_initial_reference_mismatch_is_explicit_without_moving_the_robot(self):
        config = dict(start=[.45, -.25, .35], goal=[.45, .25, .35],
            orientation=[0., 0., 1., 0.], objects=[], mapping_hz=2., voxel_size=.01)
        reference = dict(metrics=dict(initial_joints=[0.]*9, configuration=config))
        good = reference_match(reference, dict(configuration=config), [0.]*9, tolerance=.01)
        bad = reference_match(reference, dict(configuration=config), [.02]+[0.]*8, tolerance=.01)
        self.assertTrue(good['matched'])
        self.assertFalse(bad['matched'])

    def test_initial_guard_checks_live_calibration_limits_and_frozen_sources(self):
        config = dict(start=[.45, -.25, .35], goal=[.45, .25, .35],
            orientation=[0., 0., 1., 0.], objects=[], mapping_hz=2., voxel_size=.01)
        calibration = dict(camera=dict(intrinsics=np.eye(3).tolist(),
            world_from_camera_ros=np.eye(4).tolist(), resolution=[640, 480], first_acquired_at=100.))
        protocol = dict(layout_id='A', global_ray_budget=1024, sampling_policy='fixed',
            actual_camera_calibration=calibration, backend_settings=dict(max_motion_steps=1000),
            source_hashes=dict(source='frozen'), original_harness_sha256='original', wrapper_sha256='wrapper')
        reference = dict(metrics=dict(initial_joints=[0.]*9, configuration=config,
            backend_settings=protocol['backend_settings']), hashes=protocol['source_hashes'],
            research={key: protocol[key] for key in ('layout_id', 'global_ray_budget', 'sampling_policy', 'actual_camera_calibration')},
            representation_v3=dict(original_harness_sha256='original', wrapper_sha256='wrapper'))
        match = reference_match(reference, dict(configuration=config), [0.]*9, tolerance=.01, runtime_protocol=protocol)
        self.assertTrue(match['matched'])
        protocol = dict(protocol, source_hashes=dict(source='changed'))
        mismatch = reference_match(reference, dict(configuration=config), [0.]*9, tolerance=.01, runtime_protocol=protocol)
        self.assertFalse(mismatch['matched'])
        self.assertFalse(mismatch['checks']['production_source_manifest'])

    def test_low_goal_progress_does_not_become_a_collision_or_local_minimum_claim(self):
        samples = [dict(at=i*.1, position=[.45, 0., .35]) for i in range(31)]
        result = progress_windows(samples)
        self.assertGreater(result['stationary_windows'], 0)
        self.assertGreater(result['longest_low_progress_seconds'], 1.)
        self.assertIn('detours', result['meaning'])

    def test_worker_motion_context_uses_shared_offline_contract(self):
        from scripts.obstacle_representation_v3 import public_motion_context
        first, second = np.eye(4), np.eye(4)
        first[:3, 3], second[:3, 3] = [1., 2., 3.], [4., 5., 6.]
        snapshots = {'z': SimpleNamespace(parts=(SimpleNamespace(world_from_local=first),)),
            'a': SimpleNamespace(parts=(SimpleNamespace(world_from_local=second), SimpleNamespace(world_from_local=first)))}
        payload = SimpleNamespace(activate_representation=True, robot_snapshots=snapshots,
            public_goal=(.45, .25, .35), public_corridor=((.45, -.25, .35), (.45, .25, .35)))
        processor = RepresentationMappingProcessor(self.configuration(),
            representation_spec=RepresentationSpec('baseline', 'motion', 64, 'motion'),
            obstacle_settings=ObstacleExtractionSettings())
        with patch('scripts.obstacle_representation_v3.public_motion_context', wraps=public_motion_context) as builder:
            configure_extractor(processor, payload)
        builder.assert_called_once_with(snapshots, goal=payload.public_goal, corridor=payload.public_corridor)
        expected = public_motion_context(snapshots, goal=payload.public_goal, corridor=payload.public_corridor)
        self.assertEqual(processor._representation_key[-1], expected)


if __name__ == '__main__':
    unittest.main()
