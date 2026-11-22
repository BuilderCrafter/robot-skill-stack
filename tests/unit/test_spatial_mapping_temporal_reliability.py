"""Moving and occluded surfaces with the normal 10 mm grid and five-second TTL."""
import unittest

import numpy as np

from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap, OccupancyState as State
from robot_skill_stack.world.spatial_mapping.rgbd import RgbdMapper
from tests.unit.test_spatial_mapping_rgbd import ray


class _RayScene:
    origin = (.005,.025,.025)
    near = (.045,.025,.025)
    moved = (.075,.025,.025)
    hidden = (.085,.025,.025)
    far = (.115,.025,.025)

    def __init__(self):
        self.now,self.number = 100.,0
        self.map = OccupancyMap(GridSpec((0.,0.,0.),(.12,.06,.06)),clock=lambda:self.now)
        self.mapper = RgbdMapper(self.map,stride=1,clock=lambda:self.now)

    def batch(self, endpoint, *, occluded=False):
        self.now += .5
        self.number += 1
        frames = (ray(self.origin,endpoint,camera='A',number=self.number,acquired=self.now),
                  ray((.005,.045,.025),(.115,.045,.025),camera='B',number=self.number,acquired=self.now))
        def masks(frames, pixels):
            return {f.camera_id:np.full(f.depth.shape,occluded and f.camera_id=='A',bool) for f in frames}
        if not self.mapper.update(frames,mask_provider=masks):
            raise AssertionError(self.map.health)

    def query(self, position=None):
        return self.map.query(self.near if position is None else position)


class TemporalReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.scene = _RayScene()
        for _ in range(5):
            self.scene.batch(self.scene.near)

    def test_moving_surface_clears_with_real_misses_without_filling_hidden_space(self):
        self.assertEqual(self.scene.query().state,State.OCCUPIED)
        self.scene.batch(self.scene.moved)
        self.assertEqual(self.scene.query().state,State.OCCUPIED)
        for _ in range(11):
            self.scene.batch(self.scene.moved)
        self.assertEqual(self.scene.query().state,State.FREE)
        self.assertEqual(self.scene.query(self.scene.moved).state,State.OCCUPIED)
        hidden = self.scene.query(self.scene.hidden)
        self.assertEqual(hidden.state,State.UNKNOWN)
        self.assertFalse(hidden.observed)

    def test_masked_memory_expires_to_unknown_while_another_view_keeps_source_fresh(self):
        before = self.scene.query()
        for _ in range(9):
            self.scene.batch(self.scene.near,occluded=True)
        current = self.scene.query()
        self.assertEqual(current.state,State.OCCUPIED)
        self.assertEqual(current.probability,before.probability)
        self.assertEqual(current.last_observed,before.last_observed)
        self.scene.batch(self.scene.near,occluded=True)
        expired = self.scene.query()
        self.assertEqual(self.scene.now-before.last_observed,5.)
        self.assertEqual(expired.state,State.UNKNOWN)
        self.assertEqual(expired.probability,.5)
        self.assertTrue(expired.observed and expired.expired)
        self.assertEqual(expired.last_observed,before.last_observed)
        self.assertEqual(self.scene.map.timestamp,self.scene.now)
        self.assertTrue(self.scene.map.health['fresh'])


if __name__ == '__main__':
    unittest.main()
