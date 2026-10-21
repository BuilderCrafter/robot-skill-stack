"""Calibrated ray evidence and complete-snapshot health without simulator SDKs."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.spatial_mapping.occupancy import GridSpec, OccupancyMap, OccupancyState as State
from robot_skill_stack.world.spatial_mapping.rgbd import RgbdMapper, _ray_voxels, _traverse_rays


def ray(origin=(.05, .15, .15), endpoint=(.65, .15, .15), *, camera='A', number=1, acquired=100.):
    origin, endpoint = np.asarray(origin, float), np.asarray(endpoint, float)
    forward = endpoint-origin
    distance = np.linalg.norm(forward)
    forward /= distance
    right = np.cross(forward, [0., 0., 1.])
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1., 0., 0.])
    right /= np.linalg.norm(right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    transform[:3, 3] = origin
    return DepthFrame(camera, (number, number*.1), np.array([[distance]]),
        np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]]), transform, acquired)


class RgbdMappingTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.

    def make(self, *, spec=None, map_options=None, **settings):
        occupancy = OccupancyMap(spec or GridSpec((0., 0., 0.), (1., 1., 1.), .1),
            clock=lambda: self.now, **(map_options or {}))
        mapper = RgbdMapper(occupancy, clock=lambda: self.now, **{'stride': 1, **settings})
        return occupancy, mapper

    def fresh(self, frame, number):
        return replace(frame, token=(number, number*.1), acquired_at=self.now)

    def repeat(self, mapper, frames, count=2):
        for number in range(1, count+1):
            self.now += .1
            current = tuple(self.fresh(frame, number) for frame in frames)
            self.assertTrue(mapper.update(current), mapper.diagnostics)

    def test_one_camera_marks_free_surface_and_unknown_behind_it(self):
        occupancy, mapper = self.make()
        self.repeat(mapper, (ray(),))
        self.assertEqual(occupancy.query((.25, .15, .15)).state, State.FREE)
        self.assertEqual(occupancy.query((.65, .15, .15)).state, State.OCCUPIED)
        behind = occupancy.query((.85, .15, .15))
        self.assertEqual(behind.state, State.UNKNOWN)
        self.assertFalse(behind.observed)
        self.assertTrue(occupancy.health['fresh'])
        self.assertEqual(mapper.diagnostics['rays'], 1)

    def test_clipped_ray_hit_before_bounds_never_clears_behind_the_obstacle(self):
        occupancy, mapper = self.make()
        frame = ray(origin=(-.5, .15, .15), endpoint=(-.1, .15, .15))
        self.assertFalse(mapper.update((frame,)))
        self.assertIsNone(occupancy.timestamp)
        self.assertEqual(occupancy.counts()['unknown'], 1000)
        self.assertEqual(occupancy.health['status'], 'no_frame')

    def test_hit_beyond_bounds_clears_only_the_crossed_segment(self):
        occupancy, mapper = self.make()
        self.repeat(mapper, (ray(origin=(-.5, .15, .15), endpoint=(1.2, .15, .15)),))
        self.assertEqual(occupancy.counts()['occupied'], 0)
        self.assertEqual(occupancy.counts()['free'], 10)
        self.assertEqual(occupancy.query((.95, .15, .15)).state, State.FREE)
        self.assertEqual(occupancy.query((.95, .25, .15)).state, State.UNKNOWN)

    def test_axis_reverse_corner_and_exclusive_upper_boundary_traversal(self):
        spec = GridSpec((0., 0., 0.), (1., 1., 1.), .1)
        self.assertEqual(list(_ray_voxels([.8, .15, .15], [.2, .15, .15], spec)),
                         [(x, 1, 1) for x in range(7, 1, -1)])
        self.assertEqual(list(_ray_voxels([.05]*3, [.95]*3, spec)), [(x, x, x) for x in range(10)])
        self.assertEqual(list(_ray_voxels([1., -.1, .15], [1., .8, .15], spec)), [])
        occupancy, mapper = self.make()
        self.repeat(mapper, (ray(origin=(-.1, .15, .15), endpoint=(0., .15, .15)),))
        self.assertEqual(occupancy.query((.05, .15, .15)).state, State.OCCUPIED)
        self.assertEqual(occupancy.counts()['free'], 0)
        upper, source = self.make()
        self.repeat(source, (ray(origin=(.05, .15, .15), endpoint=(1., .15, .15)),))
        self.assertEqual(upper.counts()['occupied'], 0)
        self.assertEqual(upper.counts()['free'], 10)

    def test_partial_last_voxel_and_negative_world_bounds(self):
        spec = GridSpec((-.15, -.1, -.1), (.17, .1, .1), .1)
        occupancy, mapper = self.make(spec=spec)
        self.repeat(mapper, (ray(origin=(-.3, 0., 0.), endpoint=(.165, 0., 0.)),))
        self.assertEqual(occupancy.query((.165, 0., 0.)).state, State.OCCUPIED)
        self.assertEqual(occupancy.query((-.1, 0., 0.)).state, State.FREE)
        self.assertEqual(occupancy.query((.17, 0., 0.)).state, State.OUT_OF_BOUNDS)

    def test_dda_matches_independent_positive_length_voxel_box_intersections(self):
        spec = GridSpec((-.15, -.1, -.08), (.17, .24, .29), .1)
        indices = np.indices(spec.shape).reshape(3, -1).T
        lower = np.asarray(spec.bounds_min)+indices*spec.voxel_size
        upper = np.minimum(lower+spec.voxel_size, spec.bounds_max)
        rng = np.random.default_rng(17)
        segments = list(rng.uniform(-.4, .5, (40, 2, 3)))
        segments += [np.array([[-.15, -.1, -.08], [.17, .24, .29]]),
                     np.array([[.17, .1, .1], [-.15, .1, .1]])]
        for origin, endpoint in segments:
            direction = endpoint-origin
            enter, leave, valid = np.zeros(len(indices)), np.ones(len(indices)), np.ones(len(indices), bool)
            for axis in range(3):
                if direction[axis] == 0:
                    valid &= (origin[axis] >= lower[:, axis]) & (origin[axis] < upper[:, axis])
                else:
                    first = (lower[:, axis]-origin[axis])/direction[axis]
                    second = (upper[:, axis]-origin[axis])/direction[axis]
                    enter = np.maximum(enter, np.minimum(first, second))
                    leave = np.minimum(leave, np.maximum(first, second))
            expected = {tuple(value) for value in indices[valid & (leave-enter > 1e-13)]}
            self.assertEqual(set(_ray_voxels(origin, endpoint, spec)), expected,
                             (origin.tolist(), endpoint.tolist()))

    def test_actual_calibration_and_half_pixel_centers_determine_hit(self):
        occupancy, mapper = self.make()
        base = ray(origin=(.1, .1, .1), endpoint=(.1, .1, .6))
        frame = replace(base, depth=np.array([[.5, .5]]),
            intrinsics=np.array([[2., 0., .5], [0., 2., .5], [0., 0., 1.]]))
        self.assertTrue(mapper.update((frame,)))
        self.assertEqual(occupancy.query((.1, .1, .6)).state, State.OCCUPIED)
        self.assertEqual(occupancy.query((.35, .1, .6)).state, State.OCCUPIED)

    def test_invalid_pixels_are_ignored_and_an_invalid_camera_rejects_whole_snapshot(self):
        occupancy, mapper = self.make()
        base = ray()
        mixed = replace(base, depth=np.array([[.6, np.nan, np.inf, 0., -1.]]))
        self.assertTrue(mapper.update((mixed,)))
        self.assertEqual(mapper.diagnostics['rays'], 1)
        other, source = self.make()
        bad = replace(ray(camera='B'), depth=np.array([[np.nan]]))
        self.assertFalse(source.update((base, bad)))
        self.assertEqual(other.health['status'], 'error')
        self.assertIsNone(other.timestamp)
        self.assertFalse(source.update((base, replace(bad, depth=np.array([[.6]])))))
        self.assertEqual(other.health['status'], 'no_frame')
        self.assertEqual(other.counts()['occupied'], 0)

    def test_depth_limit_and_valid_nonintersecting_images_produce_no_false_freshness(self):
        occupancy, mapper = self.make(max_depth=.3)
        self.assertFalse(mapper.update((ray(),)))
        self.assertEqual(mapper.diagnostics['rays'], 0)
        self.assertIsNone(occupancy.timestamp)
        self.assertFalse(occupancy.health['fresh'])

    def test_overlapping_cameras_deduplicate_evidence_and_occupied_wins_across_rays(self):
        occupancy, mapper = self.make()
        near = ray(endpoint=(.45, .15, .15), camera='A')
        far = ray(endpoint=(.85, .15, .15), camera='B')
        self.assertTrue(mapper.update((near, far)))
        expected_hit = 1/(1+np.exp(-occupancy.hit_log_odds))
        self.assertAlmostEqual(occupancy.query((.45, .15, .15)).probability, expected_hit)
        self.assertEqual(occupancy.query((.45, .15, .15)).state, State.OCCUPIED)
        self.assertAlmostEqual(occupancy.query((.25, .15, .15)).probability,
                               1/(1+np.exp(-occupancy.miss_log_odds)))
        duplicate_map, duplicate_source = self.make()
        self.assertTrue(duplicate_source.update((near, replace(near, camera_id='B'))))
        self.assertAlmostEqual(duplicate_map.query((.45, .15, .15)).probability, expected_hit)

    def test_an_independent_camera_can_observe_space_behind_another_hit(self):
        occupancy, mapper = self.make()
        first = ray(endpoint=(.45, .15, .15), camera='A')
        second = ray(origin=(.95, .15, .15), endpoint=(.75, .15, .15), camera='B')
        self.repeat(mapper, (first, second))
        self.assertEqual(occupancy.query((.85, .15, .15)).state, State.FREE)
        self.assertEqual(occupancy.query((.65, .15, .15)).state, State.UNKNOWN)

    def test_robot_mask_skips_the_entire_ray_and_can_mask_one_of_many_views(self):
        occupancy, mapper = self.make()
        self.assertFalse(mapper.update((ray(),), {'A': np.ones((1, 1), bool)}))
        self.assertEqual(occupancy.counts()['unknown'], 1000)
        self.assertIsNone(occupancy.timestamp)
        self.now += .1
        frames = (self.fresh(ray(), 2), self.fresh(ray(origin=(.05, .65, .15),
            endpoint=(.65, .65, .15), camera='B'), 2))
        # A new mapper has a complete two-camera contract, rather than silently
        # changing the previous one's configured one-camera source.
        occupancy, mapper = self.make(expected_camera_ids=('A', 'B'))
        self.assertTrue(mapper.update(frames, {'A': np.ones((1, 1), bool)}))
        self.assertFalse(occupancy.query((.25, .15, .15)).observed)
        self.assertFalse(occupancy.query((.85, .15, .15)).observed)
        self.assertEqual(occupancy.query((.65, .65, .15)).state, State.OCCUPIED)

    def test_invalid_mask_is_transactional_and_cannot_relabel_consumed_tokens(self):
        for mask in (np.ones((1, 1), int), np.ones((2, 1), bool)):
            occupancy, mapper = self.make()
            self.assertFalse(mapper.update((ray(),), {'A': mask}))
            self.assertEqual(occupancy.health['status'], 'error')
            self.assertFalse(mapper.update((ray(),), {'A': np.zeros((1, 1), bool)}))
            self.assertIsNone(occupancy.timestamp)

    def test_duplicates_partial_freeze_missing_ids_and_regression_never_refresh(self):
        occupancy, mapper = self.make(expected_camera_ids=('A', 'B', 'C'))
        frames = tuple(ray(camera=key) for key in ('A', 'B', 'C'))
        self.assertTrue(mapper.update(frames))
        before = occupancy.snapshot()
        self.assertFalse(mapper.update(frames))
        self.assertEqual(occupancy.health['status'], 'no_frame')
        self.now += .1
        newer = [self.fresh(frame, 2) for frame in frames]
        self.assertFalse(mapper.update(tuple(newer[:2])+frames[2:]))
        self.assertIn('C', occupancy.health['last_error'])
        self.assertFalse(mapper.update(tuple(newer[:2])))
        self.assertFalse(mapper.update(tuple(newer[:2])+(
            replace(newer[2], token=(.5, .2)),)))
        self.assertEqual(occupancy.timestamp, before.timestamp)
        self.assertEqual(occupancy.revision, before.revision)
        np.testing.assert_array_equal(occupancy.snapshot().probabilities, before.probabilities)
        self.assertTrue(mapper.update(tuple(newer)))

    def test_stale_future_skewed_and_old_acquisition_times_do_not_become_fresh(self):
        for frames in ((ray(acquired=98.),), (ray(acquired=101.),),
                       (ray(), replace(ray(camera='B'), token=(1, .2)))):
            occupancy, mapper = self.make()
            self.assertFalse(mapper.update(frames))
            self.assertEqual(occupancy.health['status'], 'error')
            self.assertIsNone(occupancy.timestamp)
            repaired = tuple(replace(frame, acquired_at=self.now, token=(1, .1)) for frame in frames)
            self.assertFalse(mapper.update(repaired))
        occupancy, mapper = self.make()
        self.assertTrue(mapper.update((ray(acquired=99.8), ray(camera='B', acquired=99.9))))
        self.assertEqual(occupancy.timestamp, 99.8)
        self.now += .1
        # Fresh render tokens do not permit reusing an old acquisition timestamp.
        self.assertFalse(mapper.update((replace(ray(acquired=99.8), token=(2, .2)),
            replace(ray(camera='B', acquired=99.9), token=(2, .2)))))
        self.assertEqual(occupancy.timestamp, 99.8)

    def test_moving_obstacle_is_vacated_by_new_free_evidence_and_ages_to_unknown(self):
        occupancy, mapper = self.make(map_options=dict(evidence_ttl=.5))
        self.assertTrue(mapper.update((ray(endpoint=(.45, .15, .15)),)))
        for number in range(2, 6):
            self.now += .1
            self.assertTrue(mapper.update((ray(endpoint=(.85, .15, .15), number=number, acquired=self.now),)))
        self.assertEqual(occupancy.query((.45, .15, .15)).state, State.FREE)
        self.assertEqual(occupancy.query((.85, .15, .15)).state, State.OCCUPIED)
        self.now += .6
        expired = occupancy.query((.85, .15, .15))
        self.assertEqual(expired.state, State.UNKNOWN)
        self.assertTrue(expired.observed and expired.expired)
        self.assertEqual(expired.probability, .5)

    def test_five_cameras_share_bounded_work_fairly_independent_of_input_order(self):
        first, source = self.make(max_rays=2)
        second, reversed_source = self.make(max_rays=2)
        frames = tuple(ray(origin=(.05, .15+.15*i, .15),
            endpoint=(.65, .15+.15*i, .15), camera=key) for i, key in enumerate('ABCDE'))
        seen = set()
        for number in range(1, 4):
            self.now += .1
            current = tuple(self.fresh(frame, number) for frame in frames)
            self.assertTrue(source.update(current))
            self.assertTrue(reversed_source.update(tuple(reversed(current))))
            self.assertEqual(source.diagnostics['rays'], 2)
            seen.update(key for key, value in source.diagnostics['cameras'].items() if value['rays'])
            self.assertEqual(source.diagnostics['cameras'], reversed_source.diagnostics['cameras'])
        self.assertEqual(seen, set('ABCDE'))
        np.testing.assert_array_equal(first.snapshot().probabilities, second.snapshot().probabilities)

    def test_voxel_visit_budget_aborts_without_partially_applying_evidence(self):
        spec = GridSpec((0., 0., 0.), (1., .1, .1), .1)
        occupancy, mapper = self.make(spec=spec, max_voxel_visits=3)
        occupancy.integrate([], [[0, 0, 0]], timestamp=99., now=self.now)
        before = occupancy.snapshot()
        frame = ray(origin=(-.1, .05, .05), endpoint=(.95, .05, .05))
        self.assertFalse(mapper.update((frame,)))
        self.assertIn('budget', occupancy.health['last_error'])
        self.assertEqual(occupancy.timestamp, before.timestamp)
        self.assertEqual(occupancy.revision, before.revision)
        np.testing.assert_array_equal(occupancy.snapshot().probabilities, before.probabilities)
        self.assertFalse(mapper.update((frame,)))

    def test_default_clock_checks_age_after_traversal_without_partially_updating_map(self):
        occupancy, mapper = self.make(map_options=dict(stale_after=10.))
        occupancy.integrate([], [[0, 0, 0]], timestamp=99., now=self.now)
        before = occupancy.snapshot()
        original = _traverse_rays
        def slow_traversal(*args):
            self.now += 2.
            return original(*args)
        with patch('robot_skill_stack.world.spatial_mapping.rgbd._traverse_rays', slow_traversal):
            self.assertFalse(mapper.update((ray(),)))
        self.assertIn('stale during integration', occupancy.health['last_error'])
        self.assertFalse(occupancy.health['fresh'])
        self.assertEqual(occupancy.timestamp, before.timestamp)
        self.assertEqual(occupancy.revision, before.revision)
        np.testing.assert_array_equal(occupancy.snapshot().probabilities, before.probabilities)

    def test_explicit_evaluation_time_remains_deterministic_and_stamp_is_original(self):
        occupancy, mapper = self.make(map_options=dict(stale_after=10.))
        original = _traverse_rays
        def slow_traversal(*args):
            self.now += .4
            return original(*args)
        with patch('robot_skill_stack.world.spatial_mapping.rgbd._traverse_rays', slow_traversal):
            self.assertTrue(mapper.update((ray(acquired=99.8),), now=100.))
        self.assertEqual(occupancy.timestamp, 99.8)

    def test_invalid_settings_and_fixed_camera_contract_are_explicit(self):
        for settings in (dict(stride=0), dict(stride=True), dict(max_rays=0),
                         dict(max_voxel_visits=0), dict(max_depth=np.nan), dict(max_frame_age=0),
                         dict(max_skew=0), dict(expected_camera_ids=()), dict(expected_camera_ids=('A', 'A'))):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.make(**settings)
        occupancy, mapper = self.make()
        self.assertTrue(mapper.update((ray(),)))
        self.now += .1
        self.assertFalse(mapper.update((ray(camera='A', number=2, acquired=self.now),
                                       ray(camera='B', number=2, acquired=self.now))))
        self.assertEqual(occupancy.health['status'], 'error')


if __name__ == '__main__':
    unittest.main()
