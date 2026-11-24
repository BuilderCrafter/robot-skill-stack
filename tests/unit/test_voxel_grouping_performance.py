"""Exact voxel ordering/centroids across the optimized integer-key boundary."""
import unittest

import numpy as np

from robot_skill_stack.world.common import point_clouds as clouds


def structured_reference(points, voxel=.002, max_points=6000):
    points=np.asarray(points,float).reshape(-1,3)
    points=points[np.isfinite(points).all(1)]
    if not len(points):return points
    _,inverse=np.unique(np.floor(points/voxel+1e-9).astype(np.int64),axis=0,return_inverse=True)
    counts=np.bincount(inverse)
    result=np.column_stack([np.bincount(inverse,weights=points[:,i])/counts for i in range(3)])
    return result if len(result)<=max_points else result[np.linspace(0,len(result)-1,max_points,dtype=int)]


class VoxelGroupingTests(unittest.TestCase):
    def test_centroids_order_duplicates_invalid_pixels_and_cap_are_bitwise_identical(self):
        rng=np.random.default_rng(20261114)
        raw=rng.normal(0,.4,(12000,3))
        raw=np.concatenate([raw,raw[:100],[[np.nan,0,0],[0,np.inf,0]]])
        for voxel,maximum in ((.001,12000),(.01,57),(.5,1)):
            with self.subTest(voxel=voxel,maximum=maximum):
                np.testing.assert_array_equal(clouds.voxelize(raw,voxel,maximum),
                                              structured_reference(raw,voxel,maximum))

    def test_empty_invalid_only_and_single_point_clouds(self):
        for points in (np.empty((0,3)), np.full((2,3),np.nan), np.array([[.1,.2,.3]])):
            with self.subTest(points=points):
                np.testing.assert_array_equal(clouds.voxelize(points),structured_reference(points))

    def test_integer_range_overflow_falls_back_without_colliding_voxels(self):
        points=np.array([[-1e9,-1e9,-1e9],[1e9,1e9,1e9],[-1e9,1e9,1e9],
                         [1e9,-1e9,1e9],[-1e9,-1e9,-1e9]])
        np.testing.assert_array_equal(clouds.voxelize(points,1.,100),structured_reference(points,1.,100))
        # Each key fits int64, but subtracting extrema in int64 would overflow.
        points=np.array([[-8e18,0.,0.],[8e18,0.,0.],[0.,0.,0.]])
        np.testing.assert_array_equal(clouds.voxelize(points,1.,100),structured_reference(points,1.,100))

    def test_axis_degeneracy_negative_offsets_and_voxel_boundaries(self):
        points=np.array([[-.004,-.002,0.],[-.004,-.002,0.],[-.002,-.002,0.],
                         [0.,-.002,0.],[.002-1e-13,-.002,0.],[.002+1e-13,-.002,0.]])
        np.testing.assert_array_equal(clouds.voxelize(points),structured_reference(points))
        with self.assertRaises(ValueError):clouds.voxelize(points,0.)
        with self.assertRaises(ValueError):clouds.voxelize(points,max_points=0)

if __name__=='__main__':unittest.main()
