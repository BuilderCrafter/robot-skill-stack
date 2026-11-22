"""Occupancy geometry remains exact, bounded and independent of simulator APIs."""
from dataclasses import FrozenInstanceError
import io
from itertools import product
import unittest

import numpy as np

from robot_skill_stack.runtime.process_worker import _receive, _send
from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap
from robot_skill_stack.world.spatial_mapping.obstacles import (
    ObstacleExtractionSettings, ObstacleExtractor, _cuboids,
)


def snapshot(mask, *, voxel=.01, timestamp=100., now=100., lower=(0., 0., 0.)):
    mask = np.asarray(mask, dtype=bool)
    upper = np.asarray(lower)+np.asarray(mask.shape)*voxel
    mapping = OccupancyMap(GridSpec(lower, upper, voxel), clock=lambda: now,
                           evidence_ttl=5.)
    mapping.replace(np.argwhere(mask), timestamp=timestamp, now=now)
    return mapping.snapshot(now=now)


def represented(packet, spec):
    indices = np.indices(spec.shape).reshape(3, -1).T
    centers = spec.indices_to_centers(indices)
    count = np.zeros(len(centers), dtype=int)
    for box in packet.boxes:
        lower = np.asarray(box.center)-np.asarray(box.dimensions)/2
        upper = np.asarray(box.center)+np.asarray(box.dimensions)/2
        count += np.all((centers >= lower-1e-12) & (centers < upper-1e-12), axis=1)
    return count.reshape(spec.shape)


def reference_supported(mask):
    """Small independent component traversal oracle for the vectorized filter."""
    remaining = set(map(tuple, np.argwhere(mask)))
    result = np.zeros(mask.shape, bool)
    while remaining:
        seed = remaining.pop()
        component, pending = [seed], [seed]
        while pending:
            cell = pending.pop()
            for offset in product(range(-2, 3), repeat=3):
                if offset == (0, 0, 0):
                    continue
                neighbor = tuple(a+b for a, b in zip(cell, offset))
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    pending.append(neighbor)
        if len(component) >= 3:
            result[tuple(np.asarray(component).T)] = True
    return result


class OccupancyObstacleTests(unittest.TestCase):
    def extract(self, mask, *, maximum=128, now=100., stamp=100.):
        source = snapshot(mask, timestamp=stamp, now=now)
        packet = ObstacleExtractor(ObstacleExtractionSettings(maximum)).extract(
            source, now=now, evidence_ttl=5.)
        return source, packet

    def test_solid_cuboid_is_one_exact_zero_padding_proxy(self):
        mask = np.zeros((12, 10, 8), bool)
        mask[2:8, 3:7, 1:6] = True
        source, packet = self.extract(mask)
        self.assertEqual(len(packet.boxes), 1)
        box = packet.boxes[0]
        np.testing.assert_allclose(box.center, [.05, .05, .035], atol=1e-14)
        np.testing.assert_allclose(box.dimensions, [.06, .04, .05], atol=1e-14)
        self.assertEqual(box.voxel_count, 120)
        self.assertEqual(box.expires_at, 105.)
        self.assertEqual(packet.timestamp, source.timestamp)
        self.assertEqual(packet.revision, source.revision)
        self.assertEqual(packet.generation, source.generation)
        self.assertEqual(packet.diagnostics['padding_m'], 0.)
        np.testing.assert_array_equal(represented(packet, source.spec), mask)

    def test_singletons_pairs_and_diagonal_noise_do_not_form_large_objects(self):
        mask = np.zeros((10, 10, 10), bool)
        mask[0, 0, 0] = True
        mask[0, 0, 2] = True  # Empty middle cell must not connect these singletons.
        mask[3, 3, 3:5] = True
        mask[7, 7, 7] = mask[8, 8, 8] = True
        _, packet = self.extract(mask)
        self.assertEqual(packet.boxes, ())
        self.assertEqual(packet.diagnostics['rejected_sparse_voxels'], 6)

    def test_three_cell_component_keeps_its_leaves(self):
        mask = np.zeros((5, 5, 5), bool)
        mask[1, 2, 2] = mask[2, 2, 2] = mask[3, 2, 2] = True
        source, packet = self.extract(mask)
        np.testing.assert_array_equal(represented(packet, source.spec), mask)
        self.assertEqual(packet.diagnostics['represented_voxels'], 3)

    def test_stride_spaced_nearby_hits_survive_without_filling_the_gaps(self):
        mask = np.zeros((9, 5, 5), bool)
        mask[2, 2, 2] = mask[4, 2, 2] = mask[6, 2, 2] = True
        source, packet = self.extract(mask)
        self.assertEqual(len(packet.boxes), 3)
        np.testing.assert_array_equal(represented(packet, source.spec), mask)
        for box in packet.boxes:
            np.testing.assert_allclose(box.dimensions, [.01, .01, .01])
        self.assertEqual(packet.diagnostics['support_radius_voxels'], 2)

    def test_separate_objects_keep_single_cell_clear_gap(self):
        mask = np.zeros((11, 8, 8), bool)
        mask[1:5, 1:7, 1:7] = True
        mask[6:10, 1:7, 1:7] = True
        source, packet = self.extract(mask)
        self.assertEqual(len(packet.boxes), 2)
        actual = represented(packet, source.spec)
        np.testing.assert_array_equal(actual, mask)
        self.assertFalse(actual[5].any())

    def test_irregular_u_shape_does_not_fill_empty_center(self):
        mask = np.zeros((10, 10, 5), bool)
        mask[1:9, 1:3, 1:4] = True
        mask[1:3, 1:9, 1:4] = True
        mask[7:9, 1:9, 1:4] = True
        source, packet = self.extract(mask)
        actual = represented(packet, source.spec)
        np.testing.assert_array_equal(actual, mask)
        self.assertFalse(actual[4:6, 4:8, 1:4].any())

    def test_random_small_inputs_match_independent_component_oracle(self):
        random = np.random.default_rng(184)
        for trial in range(8):
            with self.subTest(trial=trial):
                mask = random.random((8, 9, 7)) < .21
                source, packet = self.extract(mask)
                actual = represented(packet, source.spec)
                self.assertLessEqual(int(actual.max()), 1)
                np.testing.assert_array_equal(actual.astype(bool), reference_supported(mask))

    def test_unknown_free_and_expired_evidence_are_not_obstacles(self):
        spec = GridSpec((0, 0, 0), (.08, .02, .02), .01)
        mapping = OccupancyMap(spec, clock=lambda: 100., evidence_ttl=1.)
        occupied = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        mapping.integrate([], occupied, timestamp=100., now=100.)
        # A separate FREE component and unobserved UNKNOWN cells are excluded.
        mapping.integrate([[4, 0, 0], [5, 0, 0], [6, 0, 0]], [], timestamp=100.1, now=100.1)
        before = ObstacleExtractor().extract(mapping.snapshot(now=100.2), now=100.2, evidence_ttl=1.)
        self.assertEqual(sum(box.voxel_count for box in before.boxes), 3)
        after = ObstacleExtractor().extract(mapping.snapshot(now=101.), now=101., evidence_ttl=1.)
        self.assertEqual(after.boxes, ())

    def test_expiry_uses_oldest_original_support_not_latest_map_time(self):
        mapping = OccupancyMap(GridSpec((0, 0, 0), (.03, .01, .01), .01),
                               clock=lambda: 100., evidence_ttl=5.)
        mapping.integrate([], [[0, 0, 0], [1, 0, 0], [2, 0, 0]], timestamp=100., now=100.)
        mapping.integrate([], [[2, 0, 0]], timestamp=102., now=102.)
        packet = ObstacleExtractor().extract(mapping.snapshot(now=102.), now=102., evidence_ttl=5.)
        self.assertEqual(packet.timestamp, 102.)
        self.assertEqual(packet.boxes[0].expires_at, 105.)

    def test_budget_omits_detail_without_inventing_occupied_volume(self):
        mask = np.zeros((16, 12, 8), bool)
        mask[1:4, 1:4, 1:4] = True
        mask[6:10, 1:5, 1:5] = True
        mask[12:15, 7:10, 1:4] = True
        source, packet = self.extract(mask, maximum=1)
        self.assertEqual(len(packet.boxes), 1)
        self.assertEqual(packet.boxes[0].voxel_count, 64)
        self.assertTrue(packet.diagnostics['truncated'])
        self.assertEqual(packet.diagnostics['omitted_voxels'], 54)
        actual = represented(packet, source.spec).astype(bool)
        self.assertFalse(np.any(actual & ~mask))

    def test_budget_prioritizes_above_support_plane_without_erasing_floor(self):
        mask = np.zeros((14, 8, 8), bool)
        mask[1:8, 1:6, 0] = True
        mask[9:12, 2, 4] = True
        source = snapshot(mask, lower=(0., 0., -.01))
        no_plane = ObstacleExtractor(ObstacleExtractionSettings(1)).extract(
            source, now=100., evidence_ttl=5.)
        self.assertEqual(no_plane.boxes[0].voxel_count, 35)
        ranked = ObstacleExtractor(ObstacleExtractionSettings(1, support_plane_z=0.)).extract(
            source, now=100., evidence_ttl=5.)
        self.assertEqual(ranked.boxes[0].voxel_count, 3)
        self.assertEqual(ranked.diagnostics['floor_candidate_boxes'], 1)
        self.assertEqual(ranked.diagnostics['hazard_candidate_boxes'], 1)
        self.assertEqual(ranked.diagnostics['omitted_floor_voxels'], 35)
        enough = ObstacleExtractor(ObstacleExtractionSettings(2, support_plane_z=0.)).extract(
            source, now=100., evidence_ttl=5.)
        np.testing.assert_array_equal(represented(enough, source.spec), mask)
        self.assertEqual(enough.diagnostics['represented_floor_voxels'], 35)
        self.assertEqual(enough.diagnostics['omitted_floor_voxels'], 0)

    def test_run_budget_retains_only_exact_runs_and_reports_undercoverage(self):
        mask = np.zeros((8, 1, 4), bool)
        mask[0, 0, :3] = True
        mask[2, 0, :4] = True
        mask[4, 0, :2] = True
        boxes, dropped = _cuboids(mask, max_runs=2)
        self.assertEqual(dropped, 2)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(sum(np.prod(box[[1, 3, 5]]-box[[0, 2, 4]]) for box in boxes), 7)

    def test_identity_is_deterministic_and_unchanged_revision_reuses_packet(self):
        mask = np.ones((5, 4, 3), bool)
        source = snapshot(mask)
        extractor = ObstacleExtractor()
        one = extractor.extract(source, now=100., evidence_ttl=5.)
        two = extractor.extract(source, now=100.5, evidence_ttl=5.)
        three = ObstacleExtractor().extract(source, now=100., evidence_ttl=5.)
        self.assertIs(one, two)
        self.assertEqual(one.boxes, three.boxes)
        self.assertEqual(two.extracted_at, 100.)
        self.assertEqual(two.timestamp, 100.)
        expired = extractor.extract(source, now=105., evidence_ttl=5.)
        self.assertEqual(expired.boxes, ())

    def test_partial_last_cell_and_negative_origin_have_exact_map_bounds(self):
        spec = GridSpec((-.01, 0, 0), (.025, .01, .01), .01)
        mapping = OccupancyMap(spec, clock=lambda: 100.)
        mapping.replace(np.argwhere(np.ones(spec.shape, bool)), timestamp=100., now=100.)
        packet = ObstacleExtractor().extract(mapping.snapshot(), now=100., evidence_ttl=5.)
        box = packet.boxes[0]
        np.testing.assert_allclose(np.asarray(box.center)-np.asarray(box.dimensions)/2, spec.bounds_min)
        np.testing.assert_allclose(np.asarray(box.center)+np.asarray(box.dimensions)/2, spec.bounds_max)

    def test_transport_preserves_immutable_packet_and_diagnostics(self):
        _, packet = self.extract(np.ones((3, 2, 2), bool))
        stream = io.BytesIO()
        _send(stream, packet)
        stream.seek(0)
        restored = _receive(stream)
        self.assertEqual(restored.boxes, packet.boxes)
        with self.assertRaises(FrozenInstanceError):
            restored.boxes[0].dimensions = (9., 9., 9.)
        with self.assertRaises(TypeError):
            restored.diagnostics['proxy_count'] = 900

    def test_invalid_bounds_and_time_settings_fail_explicitly(self):
        for value in (0, 129, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ObstacleExtractionSettings(value)
        for plane in (True, float('nan'), float('inf'), 'zero'):
            with self.subTest(plane=plane), self.assertRaises(ValueError):
                ObstacleExtractionSettings(support_plane_z=plane)
        source = snapshot(np.ones((3, 1, 1), bool))
        for now, ttl in ((-1, 5), (float('nan'), 5), (100, 0), (100, float('inf'))):
            with self.assertRaises(ValueError):
                ObstacleExtractor().extract(source, now=now, evidence_ttl=ttl)


if __name__ == '__main__':
    unittest.main()
