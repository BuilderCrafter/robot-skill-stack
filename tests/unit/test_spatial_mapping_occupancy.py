"""Source-independent coordinates, evidence, history and immutable map views."""
import math
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap, OccupancyState as State


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class OccupancyTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.spec = GridSpec((0., 0., 0.), (.03, .03, .03))
        self.map = OccupancyMap(self.spec, clock=self.clock)
        self.first, self.second = [[0, 0, 0]], [[1, 0, 0]]
        self.a, self.b = [.005, .005, .005], [.015, .005, .005]

    def update(self, free=(), occupied=()):
        accepted = self.map.integrate(free, occupied, timestamp=self.clock.now)
        self.clock.now += .1
        return accepted

    def test_initial_unknown_is_distinct_from_out_of_bounds(self):
        query = self.map.query(self.a)
        self.assertEqual(query.state, State.UNKNOWN)
        self.assertEqual(query.probability, .5)
        self.assertFalse(query.observed)
        self.assertFalse(query.expired)
        self.assertTrue(math.isnan(query.last_observed))
        self.assertEqual(self.map.health['status'], 'waiting')
        self.assertFalse(self.map.health['fresh'])
        self.assertIsNone(self.map.timestamp)
        snapshot = self.map.snapshot()
        self.assertEqual(snapshot.counts(), dict(unknown=27, free=0, occupied=0, out_of_bounds=0))
        for position in ([-.001, 0, 0], [.03, 0, 0], [np.nan, 0, 0], [np.inf, 0, 0]):
            with self.subTest(position=position):
                outside = self.map.query(position)
                self.assertEqual(outside.state, State.OUT_OF_BOUNDS)
                self.assertTrue(math.isnan(outside.probability))
                self.assertIsNone(outside.index)

    def test_negative_bounds_and_upper_exclusive_coordinate_conversion(self):
        spec = GridSpec((-.02, -.01, 0.), (.025, .014, .023))
        self.assertEqual(spec.shape, (5, 3, 3))
        self.assertEqual(spec.dimensions, spec.shape)
        points = [spec.bounds_min, np.nextafter(spec.bounds_max, spec.bounds_min), spec.bounds_max,
                  [-.020001, 0., .01]]
        np.testing.assert_array_equal(spec.world_to_indices(points),
                                      [[0, 0, 0], [4, 2, 2], [-1, -1, -1], [-1, -1, -1]])
        self.assertEqual(spec.voxel_count, 45)

    def test_partial_boundary_cell_centers_use_actual_clipped_cell_extent(self):
        spec = GridSpec((0, 0, 0), (.025, .014, .01))
        np.testing.assert_allclose(spec.indices_to_centers([[2, 1, 0]]), [[.0225, .012, .005]])
        indices = np.indices(spec.shape).reshape(3, -1).T
        np.testing.assert_array_equal(spec.world_to_indices(spec.indices_to_centers(indices)), indices)
        with self.assertRaises(ValueError):
            spec.indices_to_centers([[-1, 0, 0]])

    def test_rounded_integer_extent_does_not_create_an_empty_extra_voxel(self):
        spec = GridSpec((0, 0, 0), (.07, .01, .01), max_voxels=7)
        self.assertEqual(spec.shape, (7, 1, 1))
        for upper in (.070001, np.nextafter(.07, np.inf)):
            partial = GridSpec((0, 0, 0), (upper, .01, .01))
            self.assertEqual(partial.shape, (8, 1, 1))
            indices = np.indices(partial.shape).reshape(3, -1).T
            centers = partial.indices_to_centers(indices)
            self.assertTrue((centers < partial.bounds_max).all())
            np.testing.assert_array_equal(partial.world_to_indices(centers), indices)

    def test_observed_uncertain_and_unobserved_unknown_remain_distinct(self):
        self.assertTrue(self.update(free=self.first))
        uncertain, never_seen = self.map.query(self.a), self.map.query(self.b)
        self.assertEqual(uncertain.state, State.UNKNOWN)
        self.assertTrue(uncertain.observed)
        self.assertAlmostEqual(uncertain.probability, 1/(1+math.exp(.4)))
        self.assertEqual(uncertain.last_observed, 0.)
        self.assertEqual(never_seen.state, State.UNKNOWN)
        self.assertFalse(never_seen.observed)
        self.assertTrue(self.update(free=self.first))
        self.assertEqual(self.map.query(self.a).state, State.FREE)

    def test_batch_duplicates_add_one_vote_and_occupied_endpoint_wins(self):
        self.update(free=self.first*50, occupied=self.first*100)
        occupied = self.map.query(self.a)
        self.assertEqual(occupied.state, State.OCCUPIED)
        self.assertAlmostEqual(occupied.probability, 1/(1+math.exp(-.85)))
        self.assertEqual(self.map.query(self.b).state, State.UNKNOWN)
        self.assertEqual(self.map.revision, 1)

    def test_probabilities_saturate_and_new_free_rays_can_remove_an_old_obstacle(self):
        for _ in range(30):
            self.update(occupied=self.first)
        self.assertAlmostEqual(self.map.query(self.a).probability, 1/(1+math.exp(-4)))
        for _ in range(30):
            self.update(free=self.first)
        free = self.map.query(self.a)
        self.assertEqual(free.state, State.FREE)
        self.assertAlmostEqual(free.probability, 1/(1+math.exp(4)))

    def test_moving_obstacle_vacates_one_voxel_and_occupies_another(self):
        self.update(occupied=self.first)
        for _ in range(4):
            self.update(free=self.first, occupied=self.second)
        states = self.map.query_many([self.a, self.b]).states
        np.testing.assert_array_equal(states, [State.FREE, State.OCCUPIED])
        self.assertEqual(self.map.counts()['free'], 1)
        self.assertEqual(self.map.voxel_centers(State.OCCUPIED).shape, (1, 3))

    def test_expiry_returns_unknown_without_erasing_observation_history(self):
        self.update(occupied=self.first)
        before = self.map.snapshot()
        original_timestamp = self.map.timestamp
        original_revision = self.map.revision
        self.clock.now = 5.
        self.assertEqual(self.map.expire(), 1)
        self.assertEqual(self.map.expire(), 0)
        expired = self.map.query(self.a)
        self.assertEqual(expired.state, State.UNKNOWN)
        self.assertEqual(expired.probability, .5)
        self.assertTrue(expired.observed)
        self.assertTrue(expired.expired)
        self.assertEqual(expired.last_observed, 0.)
        self.assertEqual(self.map.timestamp, original_timestamp)
        self.assertEqual(self.map.revision, original_revision+1)
        self.assertEqual(before.query(self.a).state, State.OCCUPIED)
        self.assertFalse(before.query(self.a).expired)

    def test_new_measurement_starts_from_unknown_after_previous_evidence_expires(self):
        for _ in range(10):
            self.update(occupied=self.first)
        self.clock.now = 10.
        self.update(free=self.first)
        query = self.map.query(self.a)
        self.assertAlmostEqual(query.probability, 1/(1+math.exp(.4)))
        self.assertEqual(query.state, State.UNKNOWN)
        self.assertFalse(query.expired)
        self.assertEqual(query.last_observed, 10.)

    def test_freshness_and_failure_do_not_advance_source_time_or_revision(self):
        self.update(occupied=self.first)
        timestamp, revision = self.map.timestamp, self.map.revision
        self.clock.now = 1.01
        self.assertEqual(self.map.health['status'], 'stale')
        self.assertFalse(self.map.health['fresh'])
        self.assertEqual(self.map.query(self.a).state, State.OCCUPIED)
        self.map.note_failure(RuntimeError('camera unavailable'))
        self.assertEqual(self.map.health['status'], 'error')
        self.assertEqual(self.map.health['last_error'], 'camera unavailable')
        self.map.note_no_frame('unchanged camera tokens')
        self.assertEqual(self.map.health['status'], 'no_frame')
        self.assertEqual(self.map.health['reason'], 'unchanged camera tokens')
        self.assertEqual(self.map.timestamp, timestamp)
        self.assertEqual(self.map.revision, revision)
        self.update(occupied=self.first)
        self.assertTrue(self.map.health['fresh'])
        self.assertIsNone(self.map.health['last_error'])

    def test_duplicate_and_regressed_batches_cannot_strengthen_evidence(self):
        self.clock.now = 1.
        self.map.integrate([], self.first, timestamp=1.)
        before = self.map.snapshot()
        for timestamp in (1., .5):
            self.assertFalse(self.map.integrate([], self.first, timestamp=timestamp))
            self.assertEqual(self.map.revision, before.revision)
            self.assertEqual(self.map.timestamp, before.timestamp)
            self.assertEqual(self.map.query(self.a).probability, before.query(self.a).probability)
            self.assertFalse(self.map.health['fresh'])

    def test_empty_incremental_data_is_not_an_authoritative_free_observation(self):
        self.assertFalse(self.map.integrate([], [], timestamp=0.))
        self.assertIsNone(self.map.timestamp)
        self.assertEqual(self.map.revision, 0)
        self.assertEqual(self.map.health['reason'], 'no_evidence')
        self.assertEqual(self.map.query(self.a).state, State.UNKNOWN)
        self.assertTrue(self.map.replace([], timestamp=0.))
        self.assertEqual(self.map.query(self.a).state, State.FREE)
        self.assertTrue(self.map.health['fresh'])

    def test_authoritative_replacement_handles_motion_removal_and_unknown_cells(self):
        self.map.replace(self.first, unknown_indices=self.second, timestamp=0.)
        self.assertEqual(self.map.query(self.a).state, State.OCCUPIED)
        self.assertEqual(self.map.query(self.b).state, State.UNKNOWN)
        self.assertFalse(self.map.query(self.b).observed)
        self.clock.now = .1
        self.map.replace(self.second, unknown_indices=self.second, timestamp=.1)
        self.assertEqual(self.map.query(self.a).state, State.FREE)
        self.assertEqual(self.map.query(self.b).state, State.OCCUPIED)
        self.clock.now = .2
        self.map.replace([], unknown_indices=self.second, timestamp=.2)
        hidden = self.map.query(self.b)
        self.assertEqual(hidden.state, State.UNKNOWN)
        self.assertTrue(hidden.observed)
        self.assertEqual(hidden.last_observed, .1)
        self.assertEqual(self.map.counts()['free'], 26)

    def test_snapshot_and_query_exports_are_immutable_independent_and_consistent(self):
        self.update(occupied=self.first)
        snapshot = self.map.snapshot()
        result = snapshot.query_many([self.a, self.b, [1., 0., 0.]])
        np.testing.assert_array_equal(result.states, [State.OCCUPIED, State.UNKNOWN, State.OUT_OF_BOUNDS])
        np.testing.assert_array_equal(result.indices[-1], [-1, -1, -1])
        for array in (snapshot.states, snapshot.probabilities, snapshot.observed,
                      snapshot.last_observed, snapshot.expired, result.states, result.indices):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(TypeError):
            snapshot.health['fresh'] = False
        self.assertEqual(snapshot.metadata['dimensions'], self.spec.shape)
        self.update(free=self.first)
        self.assertEqual(snapshot.query(self.a).state, State.OCCUPIED)
        self.assertNotEqual(snapshot.query(self.a).probability, self.map.query(self.a).probability)

    def test_clear_starts_a_new_generation_without_mutating_old_snapshots(self):
        self.update(occupied=self.first)
        snapshot = self.map.snapshot()
        self.map.clear()
        self.assertEqual(self.map.generation, snapshot.generation+1)
        self.assertEqual(self.map.revision, snapshot.revision+1)
        self.assertIsNone(self.map.timestamp)
        self.assertEqual(self.map.health['status'], 'waiting')
        self.assertFalse(self.map.query(self.a).observed)
        self.assertFalse(self.map.query(self.a).expired)
        self.assertEqual(snapshot.query(self.a).state, State.OCCUPIED)

    def test_invalid_or_outside_indices_reject_the_whole_batch_atomically(self):
        before = self.map.snapshot()
        for bad in ([[3, 0, 0]], [[-1, 0, 0]], [[.5, 0, 0]], [[np.nan, 0, 0]], [[0, 0]],
                    [[True]*3], np.empty((0, 2))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.map.integrate(self.first, bad, timestamp=0.)
            self.assertEqual(self.map.revision, before.revision)
            self.assertEqual(self.map.query(self.a).probability, .5)
        for timestamp in (np.nan, np.inf, -1., 1.):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                self.map.replace(self.first, timestamp=timestamp, now=0.)
        self.assertIsNone(self.map.timestamp)
        self.assertEqual(self.map.query_many([]).states.shape, (0,))
        with self.assertRaises(ValueError):
            self.map.query_many(np.empty((2, 0)))

    def test_invalid_bounds_resolution_and_voxel_budget_fail_before_grid_allocation(self):
        for lower, upper, kwargs in (
                ((0, 0), (1, 1), {}), ((0, 0, 0), (0, 1, 1), {}),
                ((0, 0, 0), (np.inf, 1, 1), {}), ((0, 0, 0), (1, 1, 1), dict(voxel_size=0)),
                ((0, 0, 0), (1, 1, 1), dict(voxel_size=np.nan)),
                ((0, 0, 0), (1, 1, 1), dict(voxel_size=True)),
                ((0, 0, 0), (1, 1, 1), dict(max_voxels=1.5)),
                ((0, 0, 0), (1e100, 1e100, 1e100), {})):
            with self.subTest(lower=lower, upper=upper, kwargs=kwargs), \
                    patch('numpy.zeros') as allocate, self.assertRaises(ValueError):
                GridSpec(lower, upper, **kwargs)
            allocate.assert_not_called()

    def test_invalid_evidence_and_age_settings_fail_before_grid_allocation(self):
        for kwargs in (dict(hit_log_odds=0), dict(miss_log_odds=.1), dict(log_odds_limit=0),
                       dict(p_free=.6), dict(p_occupied=.4), dict(stale_after=0),
                       dict(evidence_ttl=np.inf), dict(clock=None)):
            with self.subTest(kwargs=kwargs), patch('numpy.zeros') as allocate, self.assertRaises(ValueError):
                OccupancyMap(self.spec, **kwargs)
            allocate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
