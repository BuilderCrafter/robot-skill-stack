"""Optional descriptors share one accepted mapping publication and generation."""
from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractionSettings
from robot_skill_stack.world.spatial_mapping.runtime import MappingInput, MappingProcessor, MappingService
from tests.unit.test_spatial_mapping_runtime import ManualWorker, gt, settings


class MappingObstacleProductTests(unittest.TestCase):
    def test_default_processor_never_invokes_obstacle_extraction(self):
        processor = MappingProcessor(settings(), clock=lambda: 100.2)
        with patch('robot_skill_stack.world.spatial_mapping.runtime.ObstacleExtractor',
                   side_effect=AssertionError('Direct mapping initialized obstacle work')):
            completed = processor.compute(MappingInput(ground_truth=gt()))
        self.assertTrue(completed.accepted)
        self.assertIsNone(completed.obstacles)
        self.assertNotIn('obstacle_extraction', completed.diagnostics)

    def test_optional_product_preserves_exact_map_arrays_and_original_metadata(self):
        cfg = settings()
        plain = MappingProcessor(cfg, clock=lambda: 100.2).compute(MappingInput(ground_truth=gt()))
        reactive = MappingProcessor(cfg, clock=lambda: 100.2,
            obstacle_settings=ObstacleExtractionSettings()).compute(MappingInput(ground_truth=gt()))
        import numpy as np
        for field in ('states', 'probabilities', 'observed', 'last_observed', 'expired'):
            np.testing.assert_array_equal(getattr(plain.snapshot, field), getattr(reactive.snapshot, field))
        self.assertEqual(reactive.obstacles.timestamp, reactive.snapshot.timestamp)
        self.assertEqual(reactive.obstacles.revision, reactive.snapshot.revision)
        self.assertEqual(reactive.obstacles.generation, reactive.snapshot.generation)
        self.assertIn('obstacle_extraction', reactive.diagnostics)

    def service(self):
        return MappingService(settings(), worker_factory=ManualWorker, clock=lambda: 100.2,
                              obstacle_settings=ObstacleExtractionSettings())

    def publish(self, service, *, generation=0):
        service.submit(MappingInput(generation=generation, ground_truth=gt()))
        service._worker.finish()
        return service.poll()

    def test_owner_read_and_sparse_query_reuse_prepared_packet_without_scan_or_copy(self):
        service = self.service()
        completed = self.publish(service)
        with patch.object(service._processor._extractor, 'extract',
                          side_effect=AssertionError('Owner extracted obstacles')):
            view = service.snapshot()
            self.assertIs(view.obstacles, completed.obstacles)
            self.assertIs(service.snapshot().obstacles, view.obstacles)
            view.query([.15, .05, .05])
        with self.assertRaises(FrozenInstanceError):
            view.obstacles = None

    def test_reset_hides_packet_and_old_read_view_remains_consistent(self):
        service = self.service()
        self.publish(service)
        old = service.snapshot()
        self.assertEqual(old.obstacles.generation, 0)
        service.reset()
        self.assertIsNone(service.snapshot().obstacles)
        self.assertEqual(old.obstacles.generation, 0)
        self.publish(service, generation=1)
        self.assertEqual(service.snapshot().obstacles.generation, 1)

    def test_failure_barrier_cannot_publish_old_obstacle_product(self):
        service = self.service()
        service.submit(MappingInput(ground_truth=gt()))
        service._worker.finish()
        service.note_source_failure('lost robot pose source')
        self.assertIsNone(service.poll())
        self.assertIsNone(service.snapshot().obstacles)
        self.assertFalse(service.health['fresh'])

    def test_close_retains_observably_closed_evidence_and_never_publishes_late_product(self):
        service = self.service()
        self.publish(service)
        old = service.snapshot().obstacles
        service.close()
        self.assertIs(service.snapshot().obstacles, old)
        self.assertFalse(service.health['fresh'])
        self.assertTrue(service.health['closed'])
        self.assertIsNone(service.poll())


if __name__ == '__main__':
    unittest.main()
