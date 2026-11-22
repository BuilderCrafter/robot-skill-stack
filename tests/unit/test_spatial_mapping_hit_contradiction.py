"""Regression requirements for directly contradicted FREE occupancy evidence.

Native V4 captures demonstrated repeated retained actor endpoints still marked
FREE. A positive vote that would leave a cell classified FREE must instead
withhold that free claim as observed UNKNOWN. These tests do not fill unseen
object interiors, weaken occupied-to-free hysteresis, or change source clocks.
"""
import math
import unittest

import numpy as np

from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap, OccupancyState as State


class HitContradictionTests(unittest.TestCase):
    target = [[0, 0, 0]]
    hidden = [[1, 0, 0]]
    target_point = (.005, .005, .005)
    hidden_point = (.015, .005, .005)

    def setUp(self):
        self.make_map()

    def make_map(self, **options):
        self.now = 100.
        self.map = OccupancyMap(GridSpec((0., 0., 0.), (.03, .01, .01)),
                                clock=lambda: self.now, **options)

    def update(self, *, free=(), occupied=()):
        self.now += .1
        self.assertTrue(self.map.integrate(free, occupied, timestamp=self.now))
        return self.map.query(self.target_point)

    def prime_free(self, *, include_hidden=False):
        cells = self.target+self.hidden if include_hidden else self.target
        # Twelve misses ensure exact clipping, avoiding a near-limit float tie.
        for _ in range(12):
            self.update(free=cells)
        self.assertEqual(self.map.query(self.target_point).state, State.FREE)

    def test_first_hit_withholds_saturated_free_claim_second_hit_is_occupied(self):
        self.prime_free()
        before = self.map.snapshot()
        first = self.update(occupied=self.target)
        self.assertEqual(first.state, State.UNKNOWN)
        self.assertEqual(first.probability, .5)
        self.assertTrue(first.observed)
        self.assertFalse(first.expired)
        self.assertEqual(first.last_observed, self.now)
        self.assertEqual(self.map.timestamp, self.now)
        self.assertEqual(before.query(self.target_point).state, State.FREE)
        second = self.update(occupied=self.target)
        self.assertEqual(second.state, State.OCCUPIED)
        self.assertGreaterEqual(second.probability, self.map.p_occupied)

    def test_new_unknown_and_already_uncertain_hit_updates_remain_unchanged(self):
        new_hit = self.update(occupied=self.target)
        self.assertEqual(new_hit.state, State.OCCUPIED)
        self.assertAlmostEqual(new_hit.probability, 1/(1+math.exp(-.85)))
        self.make_map()
        uncertain = self.update(free=self.target)
        self.assertEqual(uncertain.state, State.UNKNOWN)
        positive = self.update(occupied=self.target)
        self.assertEqual(positive.state, State.UNKNOWN)
        self.assertAlmostEqual(positive.probability, 1/(1+math.exp(-.45)))
        # This result already withholds a free claim, so retain its evidence.
        self.assertNotEqual(positive.probability, .5)

    def test_one_miss_and_repeated_miss_hysteresis_are_not_weakened(self):
        for _ in range(5):
            self.update(occupied=self.target)
        for ordinal in range(1, 13):
            current = self.update(free=self.target)
            self.assertAlmostEqual(current.probability, 1/(1+math.exp(-(4.-ordinal*.4))))
            if ordinal <= 8:
                self.assertEqual(current.state, State.OCCUPIED)
            elif ordinal < 12:
                self.assertEqual(current.state, State.UNKNOWN)
            else:
                self.assertEqual(current.state, State.FREE)
            self.assertEqual(current.last_observed, self.now)
            self.assertFalse(current.expired)

    def test_single_false_hit_withholds_free_then_two_misses_restore_it(self):
        self.prime_free()
        first = self.update(occupied=self.target)
        self.assertEqual(first.state, State.UNKNOWN)
        self.assertEqual(first.probability, .5)
        self.assertEqual(self.update(free=self.target).state, State.UNKNOWN)
        self.assertEqual(self.update(free=self.target).state, State.FREE)

    def test_configured_free_boundary_including_exact_tie_matches_classifier(self):
        # After exact saturation at -4, a +2 hit yields probability sigmoid(-2).
        # Only settings that still classify that result FREE need neutralization.
        boundary = float(1/(1+np.exp(2.)))
        for threshold, neutralize in ((float(np.nextafter(boundary, 0.)), False),
                                      (boundary, True),
                                      (float(np.nextafter(boundary, .5)), True)):
            with self.subTest(p_free=threshold):
                self.make_map(p_free=threshold, hit_log_odds=2.)
                self.prime_free()
                after = self.update(occupied=self.target)
                self.assertEqual(after.state, State.UNKNOWN)
                self.assertEqual(after.probability, .5 if neutralize else boundary)

    def test_unhit_hidden_free_cell_keeps_its_own_evidence_until_ttl(self):
        self.prime_free(include_hidden=True)
        hidden_before = self.map.query(self.hidden_point)
        self.assertEqual(self.update(occupied=self.target).state, State.UNKNOWN)
        self.update(occupied=self.target)
        self.now = hidden_before.last_observed+4.9
        self.assertTrue(self.map.integrate([], self.target, timestamp=self.now))
        hidden = self.map.query(self.hidden_point)
        self.assertEqual(hidden.state, State.FREE)
        self.assertEqual(hidden.probability, hidden_before.probability)
        self.assertEqual(hidden.last_observed, hidden_before.last_observed)
        self.assertFalse(hidden.expired)
        self.now = hidden_before.last_observed+5.
        expired = self.map.query(self.hidden_point)
        self.assertEqual(expired.state, State.UNKNOWN)
        self.assertEqual(expired.probability, .5)
        self.assertTrue(expired.expired and expired.observed)
        self.assertEqual(expired.last_observed, hidden_before.last_observed)
        self.assertTrue(self.map.health['fresh'])
        self.assertEqual(self.map.query(self.target_point).state, State.OCCUPIED)

    def test_hit_after_expired_free_starts_from_unknown_evidence(self):
        self.prime_free()
        old = self.map.query(self.target_point)
        self.now = old.last_observed+5.
        self.assertEqual(self.map.query(self.target_point).state, State.UNKNOWN)
        after = self.update(occupied=self.target)
        self.assertEqual(after.state, State.OCCUPIED)
        self.assertFalse(after.expired)
        self.assertAlmostEqual(after.probability, 1/(1+math.exp(-.85)))


if __name__ == '__main__':
    unittest.main()
