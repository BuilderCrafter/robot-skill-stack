"""Known optical-depth hits across multiple BVH root packets."""
import unittest

import numpy as np

from robot_skill_stack.world.spatial_mapping.robot_filter import _mesh_depth
from tests.unit.test_spatial_mapping_robot_filter import box_mesh


class RobotFilterPacketTests(unittest.TestCase):
    def test_nearest_surface_misses_and_existing_depth_limits(self):
        mesh = box_mesh(lower=(-1.,-1.,1.),upper=(1.,1.,2.))
        directions = np.tile([[0.,0.,1.],[1.,1.,1.],[2.,0.,1.],[0.,0.,-1.],[0.,0.,0.]],(26,1))
        limits = np.tile([np.inf,np.inf,np.inf,2.,.25],26)
        result = _mesh_depth(mesh,np.zeros(3),directions,limits)
        np.testing.assert_array_equal(result,np.tile([1.,1.,np.inf,2.,.25],26))
        np.testing.assert_array_equal(limits,np.tile([np.inf,np.inf,np.inf,2.,.25],26))


if __name__ == '__main__':
    unittest.main()
