"""Optional worker geometry shares the accepted map's publication lifetime."""
import unittest

import numpy as np

from robot_skill_stack.world.spatial_mapping import MappingInput, MappingService, ObstacleExtractionSettings
from robot_skill_stack.world.spatial_mapping.runtime import MappingProcessor
from tests.unit.test_spatial_mapping_runtime import ManualWorker, gt, settings


class MappingObstacleProductTests(unittest.TestCase):
    def test_optional_extraction_preserves_occupancy_and_publishes_the_worker_packet(self):
        plain = MappingProcessor(settings(),clock=lambda:100.2).compute(MappingInput(ground_truth=gt()))
        service = MappingService(settings(),worker_factory=ManualWorker,clock=lambda:100.2,
            obstacle_settings=ObstacleExtractionSettings())
        self.addCleanup(service.close)
        service.submit(MappingInput(ground_truth=gt()))
        service._worker.finish()
        completed = service.poll()
        self.assertIsNone(plain.obstacles)
        for field in ('states','probabilities','observed','last_observed','expired'):
            np.testing.assert_array_equal(getattr(plain.snapshot,field),getattr(completed.snapshot,field))
        view = service.snapshot()
        self.assertIs(view.obstacles,completed.obstacles)

    def test_reset_removes_optional_products_and_keeps_existing_views(self):
        service = MappingService(settings(),worker_factory=ManualWorker,clock=lambda:100.2,
            obstacle_settings=ObstacleExtractionSettings())
        self.addCleanup(service.close)
        service.submit(MappingInput(ground_truth=gt()))
        service._worker.finish()
        completed = service.poll()
        old = service.snapshot()
        service.reset()
        self.assertIsNone(service.snapshot().obstacles)
        self.assertIsNotNone(old.obstacles)
        self.assertIs(old.obstacles,completed.obstacles)
