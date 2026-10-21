"""Batched traversal against the frozen V2 scalar DDA and set evidence rules."""
from dataclasses import replace
from itertools import product
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap
from robot_skill_stack.world.spatial_mapping.rgbd import RgbdMapper, _ray_voxels, _traverse_rays
from tests.unit.test_spatial_mapping_rgbd import ray


def scalar_voxels(origin, endpoint, spec):
    """Frozen d260377 traversal, intentionally independent of the batched code."""
    origin, endpoint = np.asarray(origin, float), np.asarray(endpoint, float)
    direction = endpoint-origin
    lower, upper = np.asarray(spec.bounds_min), np.asarray(spec.bounds_max)
    enter, leave = 0., 1.
    for axis in range(3):
        if direction[axis] == 0:
            if not lower[axis] <= origin[axis] < upper[axis]:
                return
            continue
        near, far = sorted(((lower[axis]-origin[axis])/direction[axis],
                            (upper[axis]-origin[axis])/direction[axis]))
        enter, leave = max(enter, near), min(leave, far)
    if leave <= enter:
        return
    start = origin+enter*direction
    probe = np.where(direction == 0, start,
                     np.nextafter(start, np.where(direction > 0, np.inf, -np.inf)))
    probe = np.maximum(lower, np.minimum(probe, np.nextafter(upper, lower)))
    cell = np.floor((probe-lower)/spec.voxel_size).astype(int)
    step = np.sign(direction).astype(int)
    crossing, interval = np.full(3, np.inf), np.full(3, np.inf)
    moving = direction != 0
    boundary = lower+(cell+(step > 0))*spec.voxel_size
    crossing[moving] = (boundary[moving]-origin[moving])/direction[moving]
    interval[moving] = spec.voxel_size/abs(direction[moving])
    current = enter
    tolerance = 32*np.finfo(float).eps
    for _ in range(sum(spec.shape)+3):
        if np.any(cell < 0) or np.any(cell >= spec.shape):
            return
        next_crossing = float(crossing.min())
        if min(next_crossing, leave) > current+tolerance:
            yield tuple(int(value) for value in cell)
        if next_crossing >= leave-tolerance:
            return
        crossed = crossing <= next_crossing+1e-12
        cell[crossed] += step[crossed]
        crossing[crossed] += interval[crossed]
        current = next_crossing


def scalar_flat(origin, endpoints, spec):
    cells = [cell for endpoint in endpoints for cell in scalar_voxels(origin, endpoint, spec)]
    return np.ravel_multi_index(np.asarray(cells, int).reshape(-1, 3).T, spec.shape)


class BatchedTraversalTests(unittest.TestCase):
    def assert_traversal(self, origin, endpoints, spec):
        # nextafter(0, -inf) deliberately exercises subnormal ray components;
        # both reference and candidate represent their plane times as infinity.
        with np.errstate(over='ignore'):
            expected = scalar_flat(origin, endpoints, spec)
            actual, visits = _traverse_rays(origin, np.asarray(endpoints), spec, len(expected)+1)
        self.assertEqual(visits, len(expected))
        # Every visit, including duplicate cells, must survive before dedup.
        np.testing.assert_array_equal(np.sort(actual), np.sort(expected))

    def test_faces_edges_corners_nextafter_partial_cells_and_parallel_rays(self):
        for spec in (GridSpec((0, 0, 0), (1, 1, 1), .1),
                     GridSpec((-.15, -.1, -.08), (.17, .24, .29), .1),
                     GridSpec((0, 0, 0), (.07, .01, .01), .01)):
            lower, upper = np.array(spec.bounds_min), np.array(spec.bounds_max)
            middle = (lower+upper)/2
            points = [lower, upper, middle, np.nextafter(lower, -np.inf),
                      np.nextafter(lower, upper), np.nextafter(upper, lower),
                      np.nextafter(upper, np.inf), lower-.1, upper+.1]
            points += [np.where(bits, upper, lower) for bits in product((False, True), repeat=3)]
            for axis in range(3):
                point = middle.copy()
                point[axis] = lower[axis]+spec.voxel_size
                points += [point, np.nextafter(point, lower), np.nextafter(point, upper)]
            for origin in points:
                self.assert_traversal(origin, points, spec)

    def test_one_ray_sequence_preserves_reverse_zero_length_and_near_tie_rules(self):
        spec = GridSpec((0, 0, 0), (1, 1, 1), .1)
        cases = [([.8, .15, .15], [.2, .15, .15]),
                 ([.05]*3, [.95]*3), ([.05]*3, [.05]*3),
                 ([1., .1, .1], [1., .9, .9]),
                 ([0., .1, .1], [-0., .9, .9])]
        for delta in (-2e-12, -1e-13, 0., 1e-13, 2e-12):
            cases.append(([.05]*3, [.95, .95+delta, .95]))
            cases.append(([.95]*3, [.05, .05+delta, .05]))
        for origin, endpoint in cases:
            with self.subTest(origin=origin, endpoint=endpoint):
                self.assertEqual(list(_ray_voxels(origin, endpoint, spec)),
                                 list(scalar_voxels(origin, endpoint, spec)))

    def test_seeded_oblique_rays_and_batches_crossing_internal_chunk_boundary(self):
        rng = np.random.default_rng(8301)
        spec = GridSpec((-.15, -.1, -.08), (.17, .24, .29), .01)
        for origin in rng.uniform(-.3, .4, (7, 3)):
            self.assert_traversal(origin, rng.uniform(-.3, .4, (47, 3)), spec)
        endpoints = np.tile(rng.uniform(-.3, .4, (33, 3)), (17, 1))
        self.assert_traversal([.01]*3, endpoints, spec)

    def test_visit_budget_counts_duplicate_and_endpoint_cells_before_deduplication(self):
        spec = GridSpec((0, 0, 0), (1, .1, .1), .1)
        origin, endpoints = [-.1, .05, .05], np.tile([.95, .05, .05], (300, 1))
        expected = scalar_flat(origin, endpoints, spec)
        self.assertGreater(len(expected), len(np.unique(expected)))
        for budget in (0, 1, 3, len(expected)-1, len(expected), len(expected)+1):
            actual, visits = _traverse_rays(origin, endpoints, spec, budget)
            self.assertEqual(visits, min(len(expected), budget+1))
            self.assertEqual(len(actual), visits)
            if budget >= len(expected):
                np.testing.assert_array_equal(np.sort(actual), np.sort(expected))

    def test_mapper_states_probabilities_and_history_match_scalar_set_evidence(self):
        now = 100.
        spec = GridSpec((0, 0, 0), (1, 1, 1), .1)
        actual, expected = (OccupancyMap(spec, clock=lambda: now, evidence_ttl=.5) for _ in range(2))
        mapper = RgbdMapper(actual, stride=1, max_rays=11, clock=lambda: now)
        selected = {}
        def masks(frames, pixels):
            selected.update(pixels)
            return {frame.camera_id: np.indices(frame.depth.shape).sum(0) % 3 == 0 for frame in frames}
        for number in range(1, 7):
            now += .7 if number == 4 else .1
            frames = tuple(replace(ray(origin=(-.1, .1+.15*i, .15), endpoint=(.7, .1+.15*i, .15),
                camera=key, number=number, acquired=now),
                depth=np.array([[.4, .6, .8, np.nan], [.3, .5, .7, 1.2]]),
                intrinsics=np.array([[4., 0., 2.], [0., 4., 1.], [0., 0., 1.]]))
                for i, key in enumerate('ABCDE'))
            manual = {'A': np.array([[False, True, False, False], [False]*4])}
            self.assertTrue(mapper.update(frames, exclusion_masks=manual, mask_provider=masks, now=now))
            free, occupied, visits = set(), set(), 0
            for frame in frames:
                y, x = selected[frame.camera_id]
                keep = (y+x) % 3 != 0
                y, x = y[keep], x[keep]
                k, t = frame.intrinsics, frame.world_from_camera
                directions = np.c_[(x+.5-k[0, 2])/k[0, 0], (y+.5-k[1, 2])/k[1, 1], np.ones(len(x))]
                endpoints = (directions*frame.depth[y, x, None]) @ t[:3, :3].T+t[:3, 3]
                hits = spec.world_to_indices(endpoints)
                occupied.update(tuple(index) for index in hits if np.all(index >= 0))
                for endpoint in endpoints:
                    cells = list(scalar_voxels(t[:3, 3], endpoint, spec))
                    visits += len(cells)
                    free.update(cells)
            free.difference_update(occupied)
            self.assertTrue(expected.integrate(np.asarray(sorted(free), int).reshape(-1, 3),
                np.asarray(sorted(occupied), int).reshape(-1, 3), timestamp=now, now=now))
            a, b = actual.snapshot(now=now), expected.snapshot(now=now)
            for field in ('states', 'probabilities', 'observed', 'expired', 'last_observed'):
                np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
            self.assertEqual((a.timestamp, a.revision, a.generation, dict(a.health)),
                             (b.timestamp, b.revision, b.generation, dict(b.health)))
            self.assertEqual(mapper.diagnostics['voxel_visits'], visits)

    def test_budget_failure_remains_atomic_at_exact_threshold_across_cameras(self):
        spec = GridSpec((0, 0, 0), (1, 1, 1), .1)
        frames = tuple(ray(origin=(-.1, y, .15), endpoint=(.95, y, .15), camera=key)
                       for key, y in (('A', .15), ('B', .65)))
        visits = sum(len(list(scalar_voxels(frame.world_from_camera[:3, 3],
            [.95, y, .15], spec))) for frame, y in zip(frames, (.15, .65)))
        for budget in (visits-1, visits, visits+1):
            occupancy = OccupancyMap(spec, clock=lambda: 100.)
            occupancy.integrate([], [[0, 0, 0]], timestamp=99., now=100.)
            before = occupancy.snapshot()
            mapper = RgbdMapper(occupancy, stride=1, max_voxel_visits=budget, clock=lambda: 100.)
            accepted = mapper.update(frames, now=100.)
            self.assertEqual(accepted, budget >= visits)
            self.assertEqual(mapper.diagnostics['voxel_visits'], min(visits, budget+1))
            if not accepted:
                np.testing.assert_array_equal(occupancy.snapshot().probabilities, before.probabilities)
                self.assertEqual((occupancy.timestamp, occupancy.revision), (before.timestamp, before.revision))
                self.assertFalse(mapper.update(frames, now=100.))


if __name__ == '__main__':
    unittest.main()
