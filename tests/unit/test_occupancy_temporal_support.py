"""Original observed-cell retention using RGB-D confidence."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractor
from robot_skill_stack.world.spatial_mapping.occupancy import GridSpec, OccupancyMap, OccupancyState
from robot_skill_stack.world.spatial_mapping.runtime import MappingInput, MappingProcessor
from tests.unit.test_spatial_mapping_runtime import gt, settings
from tests.unit.test_spatial_mapping_rgbd import ray


class TemporalObstacleSupportTests(unittest.TestCase):
    def map(self):
        return OccupancyMap(GridSpec((0, 0, 0), (.7, .3, .3), .1), evidence_ttl=5.)

    def reinforce(self, occupancy, indices, count=4):
        for number in range(count):
            stamp = 100.+number*.1
            occupancy.integrate([], indices, timestamp=stamp, now=stamp)
        return occupancy.snapshot(now=100.+(count-1)*.1)

    def test_confidence_requires_repeated_evidence_and_default_remains_spatial(self):
        occupancy = self.map()
        temporal = ObstacleExtractor(temporal_probabilities=True)
        for number in range(4):
            stamp = 100.+number*.1
            occupancy.integrate([], [[1, 1, 1]], timestamp=stamp, now=stamp)
            snapshot = occupancy.snapshot(now=stamp)
            packet = temporal.extract(snapshot, now=stamp, evidence_ttl=5.)
            self.assertEqual(len(packet.boxes), int(number == 3))
        self.assertEqual(packet.diagnostics['strong_sparse_retained'], 1)
        self.assertEqual(packet.diagnostics['spatial_supported_voxels'], 0)
        self.assertEqual(ObstacleExtractor().extract(snapshot, now=stamp, evidence_ttl=5.).boxes, ())

    def test_exact_separate_original_cells_preserve_free_and_unknown_gaps(self):
        occupancy = self.map()
        snapshot = self.reinforce(occupancy, [[1, 1, 1], [5, 1, 1]])
        occupancy.integrate([[3, 1, 1]], [], timestamp=100.4, now=100.4)
        occupancy.integrate([[3, 1, 1]], [], timestamp=100.5, now=100.5)
        snapshot = occupancy.snapshot(now=100.5)
        packet = ObstacleExtractor(temporal_probabilities=True).extract(snapshot, now=100.5, evidence_ttl=5.)
        self.assertEqual([box.id for box in packet.boxes], ['1:2:1:2:1:2', '5:6:1:2:1:2'])
        self.assertEqual(sum(box.voxel_count for box in packet.boxes), 2)
        self.assertEqual(snapshot.states[3, 1, 1], OccupancyState.FREE)
        self.assertEqual(snapshot.states[2, 1, 1], OccupancyState.UNKNOWN)

    def test_unknown_and_expired_evidence_cannot_gain_obstacle_geometry(self):
        snapshot = self.reinforce(self.map(), [[1, 1, 1]])
        unknown = replace(snapshot, _states=np.zeros(snapshot.spec.shape, np.uint8))
        extractor = ObstacleExtractor(temporal_probabilities=True)
        self.assertEqual(extractor.extract(unknown, now=100.4, evidence_ttl=5.).boxes, ())
        extractor = ObstacleExtractor(temporal_probabilities=True)
        live = extractor.extract(snapshot, now=100.4, evidence_ttl=5.)
        self.assertEqual(live.timestamp, snapshot.timestamp)
        self.assertAlmostEqual(live.boxes[0].expires_at, 105.3)
        self.assertEqual(extractor.extract(snapshot, now=105.3001, evidence_ttl=5.).boxes, ())

    def test_ground_truth_saturated_single_cell_is_not_treated_as_temporal(self):
        occupancy = self.map()
        occupancy.replace([[1, 1, 1]], timestamp=100., now=100.)
        snapshot = occupancy.snapshot(now=100.)
        self.assertGreater(snapshot.probabilities[1, 1, 1], .95)
        self.assertEqual(ObstacleExtractor().extract(snapshot, now=100., evidence_ttl=5.).boxes, ())

    def test_processor_enables_probability_semantics_only_for_rgbd(self):
        for source in ('ground_truth', 'rgbd'):
            cfg = settings(source=source, bounds_min=(0., 0., 0.), bounds_max=(1., 1., 1.), stride=1)
            value = MappingInput(ground_truth=gt()) if source == 'ground_truth' else MappingInput(frames=(ray(),))
            from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractionSettings
            obstacle_settings = ObstacleExtractionSettings()
            with self.subTest(source=source), patch('robot_skill_stack.world.spatial_mapping.runtime.ObstacleExtractor', wraps=ObstacleExtractor) as constructor:
                MappingProcessor(cfg, clock=lambda: 100.2, obstacle_settings=obstacle_settings).compute(value)
                constructor.assert_called_once_with(obstacle_settings, temporal_probabilities=source == 'rgbd')


if __name__ == '__main__':
    unittest.main()
