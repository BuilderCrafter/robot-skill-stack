"""Representative DDA boundaries and an independent segment/voxel reference."""
import unittest

import numpy as np

from robot_skill_stack.world.spatial_mapping import GridSpec
from robot_skill_stack.world.spatial_mapping.rgbd import _traverse_rays


def intersected_cells(origin, endpoint, spec):
    """Intersect the segment with each small voxel box, without DDA stepping."""
    origin,endpoint = np.asarray(origin),np.asarray(endpoint)
    indices = np.indices(spec.shape).reshape(3,-1).T
    lower = np.asarray(spec.bounds_min)+indices*spec.voxel_size
    upper = np.minimum(lower+spec.voxel_size,spec.bounds_max)
    enter,leave,valid = np.zeros(len(indices)),np.ones(len(indices)),np.ones(len(indices),bool)
    for axis,direction in enumerate(endpoint-origin):
        if direction == 0:
            valid &= (lower[:,axis]<=origin[axis]) & (origin[axis]<upper[:,axis])
        else:
            first,second = (lower[:,axis]-origin[axis])/direction,(upper[:,axis]-origin[axis])/direction
            enter = np.maximum(enter,np.minimum(first,second))
            leave = np.minimum(leave,np.maximum(first,second))
    return np.ravel_multi_index(indices[valid & (leave-enter>1e-13)].T,spec.shape)


class BatchedTraversalTests(unittest.TestCase):
    def test_reverse_diagonal_exclusive_boundary_and_shortened_final_cell(self):
        cube = GridSpec((0.,0.,0.),(1.,1.,1.),.1)
        partial = GridSpec((-.15,-.1,-.08),(.17,.24,.29),.1)
        cases = ((cube,[.8,.15,.15],[.2,.15,.15],[(x,1,1) for x in range(7,1,-1)]),
                 (cube,[.05]*3,[.95]*3,[(x,x,x) for x in range(10)]),
                 (cube,[1.,-.1,.15],[1.,.8,.15],[]),
                 (partial,[-.3,.05,.05],[.165,.05,.05],[(x,1,1) for x in range(4)]))
        for spec,origin,endpoint,expected in cases:
            flat,_ = _traverse_rays(origin,np.array([endpoint]),spec,100)
            actual = [tuple(cell) for cell in np.column_stack(np.unravel_index(flat,spec.shape))]
            self.assertEqual(actual,expected)

    def test_oblique_batch_across_chunk_boundary_matches_independent_intersections(self):
        spec = GridSpec((-.15,-.1,-.08),(.17,.24,.29),.1)
        origin = np.array([-.2,.03,.05])
        endpoints = np.tile(np.array([[.1,.2,.24],[.3,-.2,.1],[.05,.15,-.2]]),(86,1))
        expected = np.concatenate([intersected_cells(origin,endpoint,spec) for endpoint in endpoints])
        actual,visits = _traverse_rays(origin,endpoints,spec,len(expected))
        self.assertEqual(visits,len(expected))
        np.testing.assert_array_equal(np.sort(actual),np.sort(expected))


if __name__ == '__main__':
    unittest.main()
