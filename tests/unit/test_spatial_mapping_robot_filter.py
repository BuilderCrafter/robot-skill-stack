"""Selected robot masks, exposure association and immutable worker geometry."""
from dataclasses import replace
from itertools import product
import unittest

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap, OccupancyState
from robot_skill_stack.world.spatial_mapping.rgbd import RgbdMapper
from robot_skill_stack.world.spatial_mapping.robot_filter import MeshInstance, RobotSelfFilter, RobotSnapshot, TriangleMesh


def box_mesh(lower=(-.2, -.2, 1.), upper=(.2, .2, 1.2)):
    vertices = np.array(list(product(*zip(lower, upper))))
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                      [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return TriangleMesh(vertices, faces)


def frame(depth=1., *, camera='A', token=(1, .1), transform=None, intrinsics=None):
    depth = np.atleast_2d(np.asarray(depth, float))
    height, width = depth.shape
    k = np.array([[1000., 0., width/2], [0., 1000., height/2], [0., 0., 1.]])
    return DepthFrame(camera, token, depth, k if intrinsics is None else intrinsics,
                      np.eye(4) if transform is None else transform, 100.)


def snapshot(parts=None, token=(1, .1), stamp=None):
    parts = (MeshInstance(box_mesh(), np.eye(4)),) if parts is None else tuple(parts)
    return RobotSnapshot(parts, token[1] if stamp is None else stamp, token)



def pixels(frames):
    """Select every pixel in small synthetic test images."""
    return {f.camera_id:tuple(axis.ravel() for axis in np.indices(f.depth.shape)) for f in frames}


class RobotSelfFilterTests(unittest.TestCase):
    def test_depth_comparison_masks_self_and_behind_but_preserves_foreground_and_budget_skips(self):
        current = frame([[.98,1.,1.02,np.nan,0.,1.]])
        selected = {'A':(np.zeros(5,int),np.arange(5))}
        masks = RobotSelfFilter().masks((current,),{'A':snapshot()},pixels_by_camera=selected)
        np.testing.assert_array_equal(masks['A'],[[False,True,True,False,False,False]])
        empty = {'A':(np.empty(0,int),np.empty(0,int))}
        self.assertFalse(RobotSelfFilter().masks((current,),{'A':snapshot()},pixels_by_camera=empty)['A'].any())

    def test_triangle_aabb_is_not_a_robot_silhouette(self):
        mesh = TriangleMesh(np.array([[-.2,-.2,1.],[.2,-.2,1.],[-.2,.2,1.]]),np.array([[0,1,2]]))
        current = frame(np.ones((2,2)),intrinsics=np.array([[5.,0.,1.],[0.,5.,1.],[0.,0.,1.]]))
        masks = RobotSelfFilter().masks((current,),{'A':snapshot([MeshInstance(mesh,np.eye(4))])},
                                       pixels_by_camera=pixels((current,)))
        self.assertTrue(masks['A'][0,0])
        self.assertFalse(masks['A'][1,1])

    def test_optical_depth_and_affine_part_transform(self):
        # Optical Z is 1 m for this off-axis hit; Euclidean distance is longer.
        part = MeshInstance(box_mesh((.7,-.1,1.),(.9,.1,1.1)),np.eye(4))
        current = frame(1.,intrinsics=np.array([[1.,0.,-.3],[0.,1.,.5],[0.,0.,1.]]))
        source = RobotSelfFilter()
        self.assertTrue(source.masks((current,),{'A':snapshot([part])},pixels_by_camera=pixels((current,)))['A'][0,0])
        affine = np.diag([2.,.5,1.5,1.]); affine[2,3] = .2
        part = MeshInstance(box_mesh(),affine)
        current = frame(1.7)
        self.assertTrue(source.masks((current,),{'A':snapshot([part])},pixels_by_camera=pixels((current,)))['A'][0,0])

    def test_matching_poses_and_camera_views_with_time_and_token_refusal(self):
        mesh = box_mesh()
        old = snapshot([MeshInstance(mesh,np.eye(4))])
        moved = np.eye(4); moved[0,3] = .5
        new = snapshot([MeshInstance(mesh,moved)],token=(2,.2))
        source = RobotSelfFilter()
        first = frame()
        self.assertTrue(source.masks((first,),{'A':old},pixels_by_camera=pixels((first,)))['A'][0,0])
        current = (frame(token=(2,.2)),frame(camera='B',token=(2,.2),transform=moved))
        result = source.masks(current,{'A':new,'B':new},pixels_by_camera=pixels(current))
        self.assertFalse(result['A'][0,0])
        self.assertTrue(result['B'][0,0])
        for wrong in (replace(new,simulation_time=.3),replace(new,frame_token=(3,.2))):
            with self.assertRaises(ValueError):
                source.masks(current[:1],{'A':wrong},pixels_by_camera=pixels(current[:1]))

    def test_mesh_and_pose_data_are_owned_and_immutable(self):
        vertices = np.array([[-1.,-1.,1.],[1.,-1.,1.],[0.,1.,1.]])
        triangles,transform = np.array([[0,1,2]]),np.eye(4)
        mesh = TriangleMesh(vertices,triangles)
        part = MeshInstance(mesh,transform)
        vertices[:],triangles[:],transform[2,3] = 9,0,9
        np.testing.assert_array_equal(mesh.triangles,[[0,1,2]])
        np.testing.assert_array_equal(mesh.vertices[:,2],[1.,1.,1.])
        self.assertEqual(part.world_from_local[2,3],0.)
        for array in (mesh.vertices,mesh.triangles,mesh._lower,part.world_from_local,part._local_from_world):
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_mapper_selected_robot_rays_do_not_clear_space_behind_the_robot(self):
        occupancy = OccupancyMap(GridSpec((-.1,-.1,.1),(.1,.1,2.),.1),clock=lambda:100.)
        mapper,source = RgbdMapper(occupancy,stride=1,clock=lambda:100.),RobotSelfFilter()
        def masks(frames, selection):
            robot = {'A':snapshot(token=frames[0].token)}
            return source.masks(frames,robot,pixels_by_camera=selection)
        self.assertFalse(mapper.update((frame(),),mask_provider=masks))
        self.assertFalse(occupancy.snapshot().observed.any())
        self.assertTrue(mapper.update((frame(.8,token=(2,.2)),),mask_provider=masks))
        self.assertEqual(occupancy.query((0.,0.,.8)).state,OccupancyState.OCCUPIED)
        self.assertFalse(occupancy.query((0.,0.,1.5)).observed)


if __name__ == '__main__':
    unittest.main()
