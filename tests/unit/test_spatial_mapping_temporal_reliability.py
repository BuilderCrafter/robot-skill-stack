"""Temporal evidence invariants at the normal 1 cm / five-second TTL settings.

The clock and one-pixel calibrated depth images are synthetic, not measurements
of native freshness or throughput. Each accepted batch is a distinct two-camera
exposure, separated by 0.5 seconds. The offset control ray keeps the map fresh
without observing the obstacle's row during occlusion.
"""
from dataclasses import replace
import math
import unittest

import numpy as np

from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap, OccupancyState as State
from robot_skill_stack.world.spatial_mapping.rgbd import RgbdMapper
from tests.unit.test_spatial_mapping_rgbd import ray


class _RayScene:
    """Small finite scene with independently observable target/control rows."""

    origin = (.005, .025, .025)
    near = (.045, .025, .025)
    moved = (.075, .025, .025)
    hidden = (.085, .025, .025)
    far = (.115, .025, .025)
    control_origin = (.005, .045, .025)
    control_far = (.115, .045, .025)

    def __init__(self):
        self.now, self.number = 100., 0
        self.map = OccupancyMap(GridSpec((0., 0., 0.), (.12, .06, .06)), clock=self.clock)
        self.mapper = RgbdMapper(self.map, stride=1,
            expected_camera_ids=('A', 'B'), clock=self.clock)

    def clock(self):
        return self.now

    def frames(self, endpoint=None, *, overlapping=False):
        self.number += 1
        return (
            ray(self.origin, self.far if endpoint is None else endpoint,
                camera='A', number=self.number, acquired=self.now),
            ray(self.origin if overlapping else self.control_origin,
                self.far if overlapping else self.control_far,
                camera='B', number=self.number, acquired=self.now),
        )

    def batch(self, endpoint=None, *, masked=(), overlapping=False):
        self.now += .5
        frames = self.frames(endpoint, overlapping=overlapping)
        masks = {key: np.ones((1, 1), dtype=bool) for key in masked}
        if not self.mapper.update(frames, masks):
            raise AssertionError(self.mapper.diagnostics)
        return frames

    def query(self, position=None):
        return self.map.query(self.near if position is None else position)


def saturated_free_entry_trace():
    """Return a baseline counterexample trace without requiring its slow policy.

    The native experiment must separately establish whether these directly hit
    voxels are physically occupied. This helper reports accepted endpoint-event
    ordinals; it does not mistake unobserved interior cells for hit surfaces.
    A future prompt-UNKNOWN correction may change the early states.
    """
    scene = _RayScene()
    for _ in range(10):
        scene.batch()
    trace = []
    for ordinal in range(1, 7):
        scene.batch(scene.near)
        query = scene.query()
        trace.append(dict(hit_ordinal=ordinal, time=scene.now, state=query.state.name,
            probability=query.probability, last_observed=query.last_observed))
    return trace


class TemporalReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.scene = _RayScene()

    def seed_occupied(self):
        for _ in range(5):
            self.scene.batch(self.scene.near)
        self.assertEqual(self.scene.query().state, State.OCCUPIED)

    def test_one_visible_miss_preserves_occupied_then_repeated_misses_clear_it(self):
        self.seed_occupied()
        old = self.scene.query()
        probabilities = [old.probability]
        for ordinal in range(1, 13):
            self.scene.batch()
            current = self.scene.query()
            probabilities.append(current.probability)
            self.assertEqual(current.last_observed, self.scene.now)
            self.assertFalse(current.expired)
            if ordinal == 1:
                self.assertEqual(current.state, State.OCCUPIED)
        self.assertTrue(np.all(np.diff(probabilities) < 0))
        self.assertEqual(current.state, State.FREE)
        # The six-second sequence exceeds TTL, but every real miss is evidence.
        self.assertGreater(self.scene.now-old.last_observed, self.scene.map.evidence_ttl)
        self.assertTrue(self.scene.map.health['fresh'])

    def test_motion_clears_the_old_surface_without_filling_hidden_volume(self):
        self.seed_occupied()
        for _ in range(12):
            self.scene.batch(self.scene.moved)
        self.assertEqual(self.scene.query().state, State.FREE)
        self.assertEqual(self.scene.query(self.scene.moved).state, State.OCCUPIED)
        behind = self.scene.query(self.scene.hidden)
        self.assertEqual(behind.state, State.UNKNOWN)
        self.assertFalse(behind.observed)
        self.assertTrue(math.isnan(behind.last_observed))
        self.assertEqual(self.scene.mapper.diagnostics['rays'], 2)

    def test_short_occlusion_preserves_memory_while_another_view_keeps_map_fresh(self):
        self.seed_occupied()
        before = self.scene.query()
        snapshot = self.scene.map.snapshot()
        for _ in range(4):
            self.scene.batch(self.scene.near, masked=('A',))
            hidden = self.scene.query()
            self.assertEqual(hidden.state, State.OCCUPIED)
            self.assertEqual(hidden.probability, before.probability)
            self.assertEqual(hidden.last_observed, before.last_observed)
            self.assertFalse(hidden.expired)
            self.assertEqual(self.scene.map.timestamp, self.scene.now)
            self.assertTrue(self.scene.map.health['fresh'])
            self.assertEqual(self.scene.mapper.diagnostics['cameras']['A']['rays'], 0)
            self.assertEqual(self.scene.mapper.diagnostics['cameras']['B']['rays'], 1)
        self.assertEqual(self.scene.now-before.last_observed, 2.)
        self.assertEqual(snapshot.timestamp, before.last_observed)

    def test_hidden_memory_expires_to_unknown_even_when_global_map_is_fresh(self):
        self.seed_occupied()
        before = self.scene.query()
        snapshot = self.scene.map.snapshot()
        for _ in range(9):
            self.scene.batch(self.scene.near, masked=('A',))
        self.assertEqual(self.scene.query().state, State.OCCUPIED)
        self.scene.batch(self.scene.near, masked=('A',))
        expired = self.scene.query()
        self.assertEqual(self.scene.now-before.last_observed, 5.)
        self.assertEqual(expired.state, State.UNKNOWN)
        self.assertEqual(expired.probability, .5)
        self.assertTrue(expired.observed and expired.expired)
        self.assertEqual(expired.last_observed, before.last_observed)
        self.assertEqual(self.scene.map.timestamp, self.scene.now)
        self.assertTrue(self.scene.map.health['fresh'])
        self.assertEqual(snapshot.query(self.scene.near).state, State.OCCUPIED)
        self.assertFalse(snapshot.query(self.scene.near).expired)

    def test_direct_hits_and_hidden_prior_free_cells_have_different_histories(self):
        for _ in range(10):
            self.scene.batch()
        prior_hidden = self.scene.query(self.scene.hidden)
        self.assertEqual(prior_hidden.state, State.FREE)
        probabilities = [self.scene.query().probability]
        for _ in range(6):
            self.scene.batch(self.scene.near)
            surface, hidden = self.scene.query(), self.scene.query(self.scene.hidden)
            probabilities.append(surface.probability)
            self.assertEqual(surface.last_observed, self.scene.now)
            self.assertEqual(hidden.last_observed, prior_hidden.last_observed)
            self.assertEqual(hidden.probability, prior_hidden.probability)
            self.assertEqual(hidden.state, State.FREE)
        # Require eventual recognition, allowing a safer policy to act earlier.
        self.assertTrue(np.all(np.diff(probabilities) >= 0))
        self.assertEqual(surface.state, State.OCCUPIED)
        for _ in range(4):
            self.scene.batch(self.scene.near)
        hidden = self.scene.query(self.scene.hidden)
        self.assertEqual(hidden.state, State.UNKNOWN)
        self.assertTrue(hidden.expired and hidden.observed)
        self.assertEqual(hidden.last_observed, prior_hidden.last_observed)
        self.assertEqual(self.scene.query().state, State.OCCUPIED)

    def test_conflicting_camera_rays_add_one_global_hit_vote_not_hit_plus_miss(self):
        for ordinal in range(1, 6):
            self.scene.batch(self.scene.near, overlapping=True)
            expected = 1/(1+math.exp(-min(4., ordinal*.85)))
            self.assertAlmostEqual(self.scene.query().probability, expected)
            self.assertEqual(self.scene.query().state, State.OCCUPIED)
        self.scene.batch(overlapping=True)
        self.assertAlmostEqual(self.scene.query().probability, 1/(1+math.exp(-3.6)))
        self.assertEqual(self.scene.query().state, State.OCCUPIED)

    def test_failed_frozen_and_repeated_batches_never_refresh_cell_or_source_time(self):
        self.seed_occupied()
        before = self.scene.query()
        source_time, revision = self.scene.map.timestamp, self.scene.map.revision
        old_frames = self.scene.frames(self.scene.near)
        self.assertFalse(self.scene.mapper.update(old_frames))
        # No clock advance: map rejects the new tokens' unchanged source stamp.
        self.assertEqual(self.scene.map.timestamp, source_time)
        self.assertEqual(self.scene.map.revision, revision)
        def invalid_buffer():
            frames = self.scene.frames(self.scene.near)
            return (replace(frames[0], depth=np.array([[np.nan]])), frames[1])

        attempts = (
            lambda: old_frames,
            lambda: (old_frames[0], self.scene.frames(self.scene.near)[1]),
            lambda: self.scene.frames(self.scene.near)[:1],
            invalid_buffer,
        )
        for attempt in attempts:
            self.scene.now += .5
            self.assertFalse(self.scene.mapper.update(attempt()))
            current = self.scene.query()
            self.assertEqual(current.last_observed, before.last_observed)
            self.assertEqual(current.probability, before.probability)
            self.assertEqual(self.scene.map.timestamp, source_time)
            self.assertEqual(self.scene.map.revision, revision)
        self.scene.now = before.last_observed+5.
        self.assertFalse(self.scene.mapper.update(None))
        expired = self.scene.query()
        self.assertEqual(expired.state, State.UNKNOWN)
        self.assertTrue(expired.observed and expired.expired)
        self.assertEqual(expired.last_observed, before.last_observed)
        self.assertEqual(self.scene.map.timestamp, source_time)
        self.assertFalse(self.scene.map.health['fresh'])

    def test_fresh_tokens_do_not_make_old_acquisition_time_current(self):
        self.seed_occupied()
        before = self.scene.query()
        original_time = self.scene.map.timestamp
        self.scene.now += 2.
        frames = tuple(replace(frame, acquired_at=original_time)
                       for frame in self.scene.frames(self.scene.near))
        self.assertFalse(self.scene.mapper.update(frames))
        self.assertEqual(self.scene.map.health['status'], 'error')
        self.assertEqual(self.scene.map.timestamp, original_time)
        self.assertEqual(self.scene.query().last_observed, before.last_observed)
        repaired = tuple(replace(frame, acquired_at=self.scene.now) for frame in frames)
        self.assertFalse(self.scene.mapper.update(repaired))
        self.assertEqual(self.scene.map.health['status'], 'no_frame')
        self.assertEqual(self.scene.map.timestamp, original_time)


if __name__ == '__main__':
    unittest.main()
