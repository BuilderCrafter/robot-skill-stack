"""Compare the new batched raster with independent triangle-ray intersections."""
from dataclasses import replace
import unittest
import numpy as np
from robot_skill_stack.world.perception.frame import PerceptionFrame
from robot_skill_stack.world.perception.v2.self_filter import RobotSurfaceModel
from robot_skill_stack.world.perception.v2.mesh_raster import raster_depth
from robot_skill_stack.world.perception.v2.mesh_shapes import box_triangles, round_triangles


class RasterTests(unittest.TestCase):
    def setUp(self):
        self.frame=PerceptionFrame(1,np.zeros((100,140,3),np.uint8),np.ones((100,140)),
                                  np.array([[112.,0,61.3],[0,116.,45.7],[0,0,1]]),np.eye(4),100.)

    def compare(self, meshes, poses, frame=None):
        m=RobotSurfaceModel(meshes); f=self.frame if frame is None else frame
        reference=m.depth_reference(f,poses); actual=m.depth(f,poses)
        np.testing.assert_array_equal(np.isfinite(actual),np.isfinite(reference))
        np.testing.assert_allclose(actual,reference,atol=2e-6,rtol=2e-6)
        return actual

    def test_box_front_back_and_two_sided_faces(self):
        t=box_triangles(.18); pose=np.eye(4)[None];pose[:,2,3]=.7
        self.compare([('box',t)],pose)
        self.compare([('box',t[:,::-1])],pose)

    def test_oblique_perspective_and_multiple_shapes(self):
        rng=np.random.default_rng(918)
        for _ in range(12):
            pose=np.eye(4)[None]; angle=rng.uniform(-1.4,1.4); c,s=np.cos(angle),np.sin(angle)
            pose[0,:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
            pose[0,:3,3]=[rng.uniform(-.15,.15),rng.uniform(-.15,.15),rng.uniform(.4,1.2)]
            for kind in ['Sphere','Cylinder','Capsule']:
                self.compare([('link',round_triangles(kind,.07,.12,segments=32))],pose)

    def test_large_triangle_clipped_to_view(self):
        t=np.array([[[-9,-9,1],[9,-9,1],[0,9,2.]]])
        self.compare([('large',t)],np.eye(4)[None])

    def test_triangle_crossing_camera_plane(self):
        t=box_triangles(.4);pose=np.eye(4)[None];pose[:,2,3]=.1
        self.compare([('near',t)],pose)

    def test_all_behind_camera(self):
        pose=np.eye(4)[None];pose[:,2,3]=-2
        self.assertFalse(np.isfinite(self.compare([('back',box_triangles(.4))],pose)).any())

    def test_small_and_offscreen_faces(self):
        for x in [-12.,-.5,0.,.5,12.]:
            for size in [.0001,.02,.5]:
                pose=np.eye(4)[None];pose[0,:3,3]=[x,0,.7]
                self.compare([('link',box_triangles(size))],pose)

    def test_nearest_surface_not_draw_order(self):
        a,b=box_triangles(.2),box_triangles(.1)
        poses=np.tile(np.eye(4),(2,1,1));poses[:,2,3]=[.6,.8]
        actual=self.compare([('a',a),('b',b)],poses)
        swapped=self.compare([('b',b),('a',a)],poses[::-1])
        np.testing.assert_array_equal(actual,swapped)

    def test_nonidentity_camera_pose(self):
        pose=np.eye(4)[None];pose[0,:3,3]=[.3,.2,.8]
        cam=np.eye(4);cam[:3,3]=[.2,.1,-.1]
        self.compare([('link',box_triangles(.2))],pose,replace(self.frame,world_from_camera=cam))

    def test_tiny_batches_do_not_change_output(self):
        t=round_triangles('Sphere',.1,segments=24)+[0,0,.7]
        a=raster_depth(t[(t[:,:,2]>0).all(1)],self.frame.intrinsics,self.frame.depth.shape,budget=53)
        b=raster_depth(t[(t[:,:,2]>0).all(1)],self.frame.intrinsics,self.frame.depth.shape)
        np.testing.assert_allclose(a,b)

    def test_nonpositive_z_rejected_by_direct_raster(self):
        with self.assertRaisesRegex(ValueError,'camera plane'):
            raster_depth(box_triangles(.1),self.frame.intrinsics,self.frame.depth.shape)

    def test_invalid_model_pose_rejected(self):
        m=RobotSurfaceModel([('a',box_triangles(.1))])
        with self.assertRaises(ValueError): m.depth(self.frame,np.tile(np.eye(4),(2,1,1)))

    def test_dense_surface_fidelity(self):
        self.compare([('dense',round_triangles('Sphere',.1,segments=96))],
                     np.array([[[1,0,0,0],[0,1,0,0],[0,0,1,.7],[0,0,0,1.]]]))


if __name__=='__main__': unittest.main(verbosity=2)
