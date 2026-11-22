"""Representative checks for exact, supported occupied-cell cuboids."""
import unittest

import numpy as np

from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap
from robot_skill_stack.world.spatial_mapping.obstacles import (
    ObstacleExtractionSettings, ObstacleExtractor,
)


def snapshot(mask, *, lower=(0., 0., 0.)):
    upper = np.asarray(lower)+np.asarray(mask.shape)*.01
    mapping = OccupancyMap(GridSpec(lower, upper), clock=lambda: 100.)
    mapping.replace(np.argwhere(mask), timestamp=100., now=100.)
    return mapping.snapshot()


def represented(packet, spec):
    centers = spec.indices_to_centers(np.indices(spec.shape).reshape(3, -1).T)
    counts = np.zeros(len(centers), dtype=int)
    for box in packet.boxes:
        lower = np.asarray(box.center)-np.asarray(box.dimensions)/2
        upper = np.asarray(box.center)+np.asarray(box.dimensions)/2
        counts += ((centers >= lower-1e-12) & (centers < upper-1e-12)).all(1)
    return counts.reshape(spec.shape)


class OccupancyObstacleTests(unittest.TestCase):
    def test_solid_cuboid_clips_partial_boundary_cells(self):
        spec = GridSpec((-.01, 0., 0.), (.025, .025, .015))
        mapping = OccupancyMap(spec, clock=lambda: 100.)
        mapping.replace(np.argwhere(np.ones(spec.shape, bool)), timestamp=100.)
        source = mapping.snapshot()
        packet = ObstacleExtractor().extract(source, now=100., evidence_ttl=5.)
        self.assertEqual(len(packet.boxes), 1)
        box = packet.boxes[0]
        np.testing.assert_allclose(box.center, [.0075, .0125, .0075], atol=1e-14)
        np.testing.assert_allclose(box.dimensions, [.035, .025, .015], atol=1e-14)
        self.assertEqual(box.voxel_count, 24)
        self.assertEqual(box.expires_at, 105.)
        self.assertEqual((packet.generation, packet.revision, packet.timestamp),
                         (source.generation, source.revision, source.timestamp))
        np.testing.assert_array_equal(represented(packet, spec), np.ones(spec.shape, int))

    def test_components_keep_holes_and_stride_gaps_but_reject_isolated_noise(self):
        expected = np.zeros((16, 10, 4), bool)
        expected[1:6, 1:3, 1:3] = True
        expected[1:3, 3:7, 1:3] = True
        expected[[9, 11, 13], 5, 2] = True
        noisy = expected.copy()
        noisy[15, 0, 0] = noisy[8, 9, 0] = noisy[9, 9, 0] = True
        source = snapshot(noisy)
        packet = ObstacleExtractor().extract(source, now=100., evidence_ttl=5.)
        self.assertEqual(len(packet.boxes), 5)
        np.testing.assert_array_equal(represented(packet, source.spec), expected)

    def test_temporal_rgbd_support_retains_only_repeated_occupied_hits(self):
        mapping = OccupancyMap(GridSpec((0., 0., 0.), (.09, .01, .01)))
        extractor = ObstacleExtractor(temporal_probabilities=True)
        for number in range(4):
            now = 100.+number*.1
            mapping.integrate([], [[1, 0, 0], [7, 0, 0]], timestamp=now, now=now)
            packet = extractor.extract(mapping.snapshot(now=now), now=now, evidence_ttl=5.)
            self.assertEqual(len(packet.boxes), 2 if number == 3 else 0)
        mapping.integrate([], [[4, 0, 0]], timestamp=100.4, now=100.4)
        for now in (100.5, 100.6):
            mapping.integrate([[3, 0, 0]], [], timestamp=now, now=now)
        source = mapping.snapshot(now=100.6)
        packet = extractor.extract(source, now=100.6, evidence_ttl=5.)
        self.assertEqual([b.id for b in packet.boxes], ['1:2:0:1:0:1', '7:8:0:1:0:1'])
        for box in packet.boxes:
            self.assertAlmostEqual(box.expires_at, 105.3)
        self.assertEqual(ObstacleExtractor().extract(source, now=100.6, evidence_ttl=5.).boxes, ())

    def test_budget_ranks_hazards_before_floor_without_expanding_geometry(self):
        mask = np.zeros((14, 8, 8), bool)
        mask[1:8, 1:6, 0] = True
        mask[9:12, 2, 4] = mask[9:12, 5, 4] = True
        source = snapshot(mask, lower=(0., 0., -.01))
        largest = ObstacleExtractor(ObstacleExtractionSettings(1)).extract(
            source, now=100., evidence_ttl=5.)
        self.assertEqual(largest.boxes[0].voxel_count, 35)
        ranked = ObstacleExtractor(ObstacleExtractionSettings(1, 0.)).extract(
            source, now=100., evidence_ttl=5.)
        self.assertEqual([b.id for b in ranked.boxes], ['9:12:2:3:4:5'])
        self.assertEqual(ranked.diagnostics['omitted_voxels'], 38)
        expected = np.zeros_like(mask)
        expected[9:12, 2, 4] = True
        np.testing.assert_array_equal(represented(ranked, source.spec), expected)
        complete = ObstacleExtractor(ObstacleExtractionSettings(3, 0.)).extract(
            source, now=100., evidence_ttl=5.)
        np.testing.assert_array_equal(represented(complete, source.spec), mask)
        self.assertEqual(complete.diagnostics['omitted_voxels'], 0)

    def test_cache_expires_at_oldest_support_and_refreshes_on_map_revision(self):
        mapping = OccupancyMap(GridSpec((0., 0., 0.), (.03, .01, .01)))
        cells = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        mapping.integrate([], cells, timestamp=100., now=100.)
        mapping.integrate([], cells[-1:], timestamp=102., now=102.)
        source = mapping.snapshot(now=102.)
        extractor = ObstacleExtractor()
        packet = extractor.extract(source, now=102., evidence_ttl=5.)
        self.assertEqual(packet.timestamp, 102.)
        self.assertEqual(packet.boxes[0].expires_at, 105.)
        self.assertIs(extractor.extract(source, now=104.9, evidence_ttl=5.), packet)
        self.assertEqual(extractor.extract(source, now=105., evidence_ttl=5.).boxes, ())
        mapping.integrate([], cells, timestamp=105.1, now=105.1)
        renewed = extractor.extract(mapping.snapshot(now=105.1), now=105.1, evidence_ttl=5.)
        self.assertNotEqual(renewed.revision, packet.revision)
        self.assertEqual(renewed.boxes[0].voxel_count, 3)
        self.assertAlmostEqual(renewed.boxes[0].expires_at, 110.1)


if __name__ == '__main__':
    unittest.main()
