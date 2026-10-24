"""Exact independent mesh/cuboid reference features used by native evaluation."""
import unittest
import numpy as np

from tests.support.reactive_geometry import (
    triangle_box_squared,mesh_box_distance,CollisionMeshReference,reduce_clearances,convex_polygon_triangles)
from robot_skill_stack.world.spatial_mapping.robot_filter import TriangleMesh


class ReactiveReferenceGeometryTests(unittest.TestCase):
    def test_cooked_convex_face_fan_retains_float32_roundoff_and_checks_topology(self):
        vertices=np.array([[0.,0.,0.],[.001,0.,0.],[.001,.001,0.],[0.,.001,2e-9]],dtype=np.float32)
        np.testing.assert_array_equal(convex_polygon_triangles(vertices,np.array([4]),np.arange(4)),
                                      [[0,1,2],[0,2,3]])
        for counts,indices in (([4],[0,1,2]),([3],[0,1,9]),([2],[0,1])):
            with self.assertRaises(ValueError):
                convex_polygon_triangles(vertices,np.array(counts),np.array(indices))

    def test_diagnostic_serializer_keeps_unavailable_sdk_values_standard_json(self):
        import json
        from dataclasses import dataclass
        from types import MappingProxyType
        from scripts.diagnose_reactive_motion import serial
        @dataclass(frozen=True)
        class Metadata:
            values: object
        value=Metadata(MappingProxyType(dict(position=np.array([np.nan,np.inf,-np.inf,1.]),
                                             timestamp=np.float64(np.nan))))
        encoded=json.dumps(serial(value),allow_nan=False)
        self.assertEqual(json.loads(encoded),dict(values=dict(position=[None,None,None,1.],timestamp=None)))

    def test_triangle_face_crossing_box_has_zero_distance_without_near_vertices(self):
        triangles=np.array([[[-3.,-3.,0.],[3.,-3.,0.],[0.,3.,0.]]])
        self.assertEqual(triangle_box_squared(triangles,np.ones(3))[0],0.)

    def test_face_parallel_and_edge_feature_distances(self):
        triangles=np.array([[[-3.,-3.,2.],[3.,-3.,2.],[0.,3.,2.]],
                            [[2.,2.,-.5],[2.,2.,.5],[3.,3.,0.]]])
        np.testing.assert_allclose(triangle_box_squared(triangles,np.ones(3)),[1.,2.])

    def test_mesh_rotated_and_translated_clearance_and_solid_containment(self):
        vertices=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
        triangles=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],
            [0,4,5],[0,5,1],[2,3,7],[2,7,6],
            [0,2,6],[0,6,4],[1,5,7],[1,7,3]])
        mesh=TriangleMesh(vertices,triangles)
        transform=np.eye(4);transform[0,3]=3.
        self.assertAlmostEqual(mesh_box_distance(mesh,transform,np.ones(3)),1.)
        transform[:3,:3]=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
        self.assertAlmostEqual(mesh_box_distance(mesh,transform,np.ones(3)),1.)
        self.assertEqual(mesh_box_distance(mesh,np.eye(4),np.full(3,.1)),0.)

    def test_exact_duplicate_and_witness_pruning_preserve_bruteforce_minimum(self):
        mesh=TriangleMesh(np.array([[0.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),np.array([[0,1,2]]))
        part=CollisionMeshReference('/mesh','/body','wrist',mesh,np.eye(4))
        transforms=[]
        for x in (3.,3.,2.5,2.5,2.,2.,3.):
            transform=np.eye(4);transform[0,3]=x
            transforms.append(np.array([transform]))
        boxes=[[('/box',np.eye(4),np.ones(3))] for _ in transforms]
        expected=min(mesh_box_distance(mesh,poses[0],np.full(3,.5)) for poses in transforms)
        reduced=reduce_clearances([part],['/body'],transforms,boxes)
        self.assertEqual(reduced['minimum_m'],expected)
        self.assertEqual(reduced['sampled_poses'],len(transforms))
        self.assertGreater(reduced['exact_duplicate_pose_skips'],0)
        self.assertLess(reduced['mesh_box_evaluations'],len(transforms))


if __name__=='__main__':
    unittest.main()
