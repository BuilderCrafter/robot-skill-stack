"""Depth projection, camera sampling and atomic occupancy integration."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.spatial_mapping.occupancy import GridSpec, OccupancyMap, OccupancyState as State
from robot_skill_stack.world.spatial_mapping.rgbd import RgbdMapper, _traverse_rays


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

    def make(self, *, spec=None, **settings):
        occupancy = OccupancyMap(spec or GridSpec((0.,0.,0.),(1.,1.,1.),.1),clock=lambda:self.now)
        return occupancy,RgbdMapper(occupancy,clock=lambda:self.now,**({'stride':1} | settings))

    def test_calibrated_half_pixel_projection_free_hits_and_ignored_depths(self):
        occupancy,mapper = self.make(max_depth=1.)
        image = replace(ray(origin=(.1,.1,.1),endpoint=(.1,.1,.6)),
            depth=np.array([[.5,.5,np.nan,0.,2.]]),intrinsics=np.array([[2.,0.,.5],[0.,2.,.5],[0.,0.,1.]]))
        for number in (1,2):
            self.now += .1
            self.assertTrue(mapper.update((replace(image,token=(number,number*.1),acquired_at=self.now),)))
        self.assertEqual(occupancy.query((.1,.1,.6)).state,State.OCCUPIED)
        self.assertEqual(occupancy.query((.35,.1,.6)).state,State.OCCUPIED)
        self.assertEqual(occupancy.query((.1,.1,.35)).state,State.FREE)
        self.assertFalse(occupancy.query((.1,.1,.85)).observed)
        distant,source = self.make(max_depth=.3)
        self.assertFalse(source.update((ray(),)))
        self.assertIsNone(distant.timestamp)
        self.assertFalse(distant.snapshot().observed.any())

    def test_camera_evidence_is_deduplicated_and_occupied_wins_over_free(self):
        near,far = ray(endpoint=(.45,.15,.15)),ray(endpoint=(.85,.15,.15),camera='B')
        for second in (far,replace(near,camera_id='B')):
            occupancy,mapper = self.make()
            self.assertTrue(mapper.update((near,second)))
            self.assertAlmostEqual(occupancy.query((.45,.15,.15)).probability,1/(1+np.exp(-.85)))
            self.assertAlmostEqual(occupancy.query((.25,.15,.15)).probability,1/(1+np.exp(.4)))

    def test_ray_budget_is_shared_before_robot_masking_without_backfill(self):
        occupancy,mapper = self.make(max_rays=4)
        dense = replace(ray(),depth=np.full((1,8),.6))
        other = ray(origin=(.05,.65,.15),endpoint=(.65,.65,.15),camera='B')
        selected = {}
        def masks(frames, pixels):
            selected.update(pixels)
            return {f.camera_id:np.full(f.depth.shape,f.camera_id=='A',bool) for f in frames}
        self.assertTrue(mapper.update((other,dense),mask_provider=masks))
        np.testing.assert_array_equal(selected['A'][1],[1,4,6])
        self.assertEqual(len(selected['B'][1]),1)
        self.assertFalse(occupancy.query((.25,.15,.15)).observed)
        self.assertFalse(occupancy.query((.85,.15,.15)).observed)
        self.assertEqual(occupancy.query((.65,.65,.15)).state,State.OCCUPIED)

    def test_repeated_frozen_and_stale_batches_preserve_the_original_publication(self):
        occupancy,mapper = self.make()
        frames = (ray(acquired=99.8),ray(camera='B',acquired=99.9))
        self.assertTrue(mapper.update(frames))
        before = occupancy.snapshot()
        self.assertEqual(before.timestamp,99.8)
        self.assertFalse(mapper.update(frames))
        self.now += .1
        newer = tuple(replace(f,token=(2,.2),acquired_at=self.now) for f in frames)
        self.assertFalse(mapper.update((newer[0],frames[1])))
        self.assertFalse(mapper.update(tuple(replace(f,acquired_at=98.) for f in newer)))
        self.assertFalse(mapper.update(newer))  # Bad exposures cannot be repaired and replayed.
        self.assertEqual((occupancy.timestamp,occupancy.revision),(before.timestamp,before.revision))
        np.testing.assert_array_equal(occupancy.snapshot().probabilities,before.probabilities)
        self.assertTrue(mapper.update(tuple(replace(f,token=(3,.3)) for f in newer)))

    def test_unsynchronized_or_future_observations_are_not_usable_evidence(self):
        for frames in ((ray(),replace(ray(camera='B'),token=(1,.2))),(ray(acquired=101.),)):
            occupancy,mapper = self.make()
            self.assertFalse(mapper.update(frames))
            self.assertIsNone(occupancy.timestamp)
            self.assertIsNotNone(occupancy.error)

    def test_visit_limit_counts_both_cameras_and_rejection_is_atomic(self):
        spec = GridSpec((0.,0.,0.),(1.,.1,.1),.1)
        frames = tuple(ray(origin=(-.1,.05,.05),endpoint=(.95,.05,.05),camera=key) for key in ('A','B'))
        for budget in (19,20):
            occupancy,mapper = self.make(spec=spec,max_voxel_visits=budget)
            occupancy.integrate([],[[0,0,0]],timestamp=99.,now=self.now)
            before = occupancy.snapshot()
            self.assertEqual(mapper.update(frames),budget==20)
            if budget == 19:
                self.assertEqual((occupancy.timestamp,occupancy.revision),(before.timestamp,before.revision))
                np.testing.assert_array_equal(occupancy.snapshot().probabilities,before.probabilities)
            else:
                self.assertEqual(occupancy.query((.95,.05,.05)).state,State.OCCUPIED)

    def test_completion_age_is_checked_and_offline_evaluation_time_is_explicit(self):
        for explicit in (False,True):
            self.now = 100.
            occupancy,mapper = self.make()
            def slow(*args):
                self.now += 2.
                return _traverse_rays(*args)
            with patch('robot_skill_stack.world.spatial_mapping.rgbd._traverse_rays',slow):
                accepted = mapper.update((ray(acquired=99.8),),now=100. if explicit else None)
            self.assertEqual(accepted,explicit)
            self.assertEqual(occupancy.timestamp,99.8 if explicit else None)


if __name__ == '__main__':
    unittest.main()
