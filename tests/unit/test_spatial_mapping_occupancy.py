"""Voxel coordinates, temporal evidence and independent immutable snapshots."""
import math
import unittest

import numpy as np

from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap, OccupancyState as State


class OccupancyTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.spec = GridSpec((0.,0.,0.),(.03,.01,.01))
        self.map = OccupancyMap(self.spec,clock=lambda:self.now)
        self.a,self.b,self.c = (.005,.005,.005),(.015,.005,.005),(.025,.005,.005)

    def test_upper_exclusive_coordinates_partial_cells_and_grid_budget(self):
        spec = GridSpec((-.02,-.01,0.),(.025,.014,.023))
        self.assertEqual(spec.shape,(5,3,3))
        np.testing.assert_array_equal(spec.world_to_indices([spec.bounds_min,
            np.nextafter(spec.bounds_max,spec.bounds_min),spec.bounds_max]),
            [[0,0,0],[4,2,2],[-1,-1,-1]])
        partial = GridSpec((0.,0.,0.),(.025,.014,.01))
        np.testing.assert_allclose(partial.indices_to_centers([[2,1,0]]),[[.0225,.012,.005]])
        indices = np.indices(partial.shape).reshape(3,-1).T
        np.testing.assert_array_equal(partial.world_to_indices(partial.indices_to_centers(indices)),indices)
        self.assertEqual(GridSpec((0.,0.,0.),(.07,.01,.01),max_voxels=7).shape,(7,1,1))
        with self.assertRaises(ValueError):
            GridSpec((0.,0.,0.),(100.,100.,100.),max_voxels=100)

    def test_unobserved_uncertain_free_and_outside_are_distinct(self):
        initial = self.map.query_many([self.a,self.spec.bounds_max,[np.nan,0.,0.]])
        np.testing.assert_array_equal(initial.states,[State.UNKNOWN,State.OUT_OF_BOUNDS,State.OUT_OF_BOUNDS])
        self.assertFalse(initial.observed.any())
        self.assertEqual(initial.probabilities[0],.5)
        self.assertTrue(math.isnan(initial.probabilities[1]))
        self.map.integrate([[0,0,0]],[],timestamp=self.now)
        first = self.map.query(self.a)
        self.assertEqual(first.state,State.UNKNOWN)
        self.assertTrue(first.observed)
        self.assertAlmostEqual(first.probability,1/(1+math.exp(.4)))
        self.now += .1
        self.map.integrate([[0,0,0]],[],timestamp=self.now)
        self.assertEqual(self.map.query(self.a).state,State.FREE)

    def test_duplicates_hit_precedence_and_replayed_time_do_not_add_votes(self):
        self.assertTrue(self.map.integrate([[0,0,0]]*5,[[0,0,0]]*7,timestamp=self.now))
        before = self.map.query(self.a)
        self.assertEqual(before.state,State.OCCUPIED)
        self.assertAlmostEqual(before.probability,1/(1+math.exp(-.85)))
        self.assertFalse(self.map.integrate([],[[0,0,0]],timestamp=self.now))
        self.assertFalse(self.map.integrate([],[[0,0,0]],timestamp=self.now-.1))
        self.assertEqual(self.map.revision,1)
        self.assertEqual(self.map.query(self.a).probability,before.probability)

    def test_expiry_preserves_history_and_new_votes_start_without_expired_prior(self):
        self.map.integrate([],[[0,0,0]],timestamp=self.now)
        self.now += .5
        self.map.integrate([],[[1,0,0]],timestamp=self.now)
        before = self.map.snapshot()
        self.now = 105.1
        expired = self.map.query(self.a)
        self.assertEqual(expired.state,State.UNKNOWN)
        self.assertEqual(expired.probability,.5)
        self.assertTrue(expired.observed and expired.expired)
        self.assertEqual(expired.last_observed,100.)
        self.assertEqual(self.map.query(self.b).state,State.OCCUPIED)
        self.assertEqual((self.map.timestamp,self.map.revision),(100.5,2))
        self.map.integrate([[0,0,0]],[],timestamp=self.now)
        fresh = self.map.query(self.a)
        self.assertAlmostEqual(fresh.probability,1/(1+math.exp(.4)))
        self.assertFalse(fresh.expired)
        self.assertEqual(before.query(self.a).state,State.OCCUPIED)

    def test_gt_replacement_distinguishes_empty_evidence_and_unknown_history(self):
        self.assertFalse(self.map.integrate([],[],timestamp=self.now))
        self.assertIsNone(self.map.timestamp)
        self.assertTrue(self.map.replace([[0,0,0]],unknown_indices=[[1,0,0]],timestamp=self.now))
        np.testing.assert_array_equal(self.map.query_many([self.a,self.b,self.c]).states,
                                      [State.OCCUPIED,State.UNKNOWN,State.FREE])
        self.assertFalse(self.map.query(self.b).observed)
        self.now += .1
        self.map.replace([[1,0,0]],unknown_indices=[[1,0,0]],timestamp=self.now)
        self.assertEqual(self.map.query(self.b).state,State.OCCUPIED)
        last_seen = self.now
        self.now += .1
        self.map.replace([],unknown_indices=[[1,0,0]],timestamp=self.now)
        hidden = self.map.query(self.b)
        self.assertEqual(hidden.state,State.UNKNOWN)
        self.assertTrue(hidden.observed)
        self.assertEqual(hidden.last_observed,last_seen)

    def test_snapshot_and_sparse_exports_stay_immutable_when_the_map_updates(self):
        self.map.integrate([],[[0,0,0]],timestamp=self.now)
        snapshot = self.map.snapshot()
        batch = snapshot.query_many([self.a,self.b,[1.,0.,0.]])
        np.testing.assert_array_equal(batch.states,[State.OCCUPIED,State.UNKNOWN,State.OUT_OF_BOUNDS])
        for array in (snapshot.states,snapshot.probabilities,snapshot.observed,snapshot.last_observed,
                      snapshot.expired,batch.states,batch.last_observed):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        self.now += .1
        self.map.replace([[1,0,0]],timestamp=self.now)
        self.assertEqual(snapshot.query(self.a).state,State.OCCUPIED)
        self.assertEqual(self.map.query(self.a).state,State.FREE)
        self.assertEqual(snapshot.timestamp,100.)


if __name__ == '__main__':
    unittest.main()
