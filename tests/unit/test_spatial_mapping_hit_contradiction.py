"""A measured hit withholds a contradictory FREE claim without filling hidden cells."""
import math
import unittest

from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap, OccupancyState as State


class HitContradictionTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.map = OccupancyMap(GridSpec((0.,0.,0.),(.03,.01,.01)),clock=lambda:self.now)
        self.point = (.005,.005,.005)

    def update(self, *, free=(), occupied=()):
        self.now += .1
        self.assertTrue(self.map.integrate(free,occupied,timestamp=self.now))
        return self.map.query(self.point)

    def test_first_hit_withholds_saturated_free_and_second_hit_is_occupied(self):
        for _ in range(12):
            self.update(free=[[0,0,0],[1,0,0]])
        hidden_before = self.map.query((.015,.005,.005))
        self.assertAlmostEqual(self.map.query(self.point).probability,1/(1+math.exp(4.)))
        first = self.update(occupied=[[0,0,0]])
        self.assertEqual(first.state,State.UNKNOWN)
        self.assertEqual(first.probability,.5)
        self.assertTrue(first.observed and not first.expired)
        second = self.update(occupied=[[0,0,0]])
        self.assertEqual(second.state,State.OCCUPIED)
        self.assertAlmostEqual(second.probability,1/(1+math.exp(-.85)))
        hidden = self.map.query((.015,.005,.005))
        self.assertEqual((hidden.probability,hidden.last_observed),
                         (hidden_before.probability,hidden_before.last_observed))

    def test_real_misses_preserve_then_clear_an_old_occupied_surface(self):
        for _ in range(5):
            self.update(occupied=[[0,0,0]])
        first = self.update(free=[[0,0,0]])
        self.assertEqual(first.state,State.OCCUPIED)
        self.assertAlmostEqual(first.probability,1/(1+math.exp(-3.6)))
        for _ in range(11):
            last = self.update(free=[[0,0,0]])
        self.assertEqual(last.state,State.FREE)
        self.assertAlmostEqual(last.probability,1/(1+math.exp(.8)))

    def test_hit_after_expired_free_uses_no_old_prior_even_without_an_expiry_read(self):
        for _ in range(12):
            self.update(free=[[0,0,0]])
        self.now += 5.
        hit = self.update(occupied=[[0,0,0]])
        self.assertEqual(hit.state,State.OCCUPIED)
        self.assertFalse(hit.expired)
        self.assertAlmostEqual(hit.probability,1/(1+math.exp(-.85)))


if __name__ == '__main__':
    unittest.main()
