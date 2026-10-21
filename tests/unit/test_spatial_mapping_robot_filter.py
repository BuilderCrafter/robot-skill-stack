"""Independent mesh/ray, occlusion and pose-history tests without simulator SDKs."""
from dataclasses import replace
from itertools import product
import unittest

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap, OccupancyState
from robot_skill_stack.world.spatial_mapping.rgbd import RgbdMapper
from robot_skill_stack.world.spatial_mapping.robot_filter import (
    MeshInstance, RobotSelfFilter, RobotSnapshot, TriangleMesh,
)


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


class RobotSelfFilterTests(unittest.TestCase):
    def test_depth_comparison_retains_foreground_and_withholds_self_and_behind(self):
        source = RobotSelfFilter(stride=1)
        masks = source.masks((frame([[.98, 1., 1.02, np.nan, 0.]]),), {'A': snapshot()})
        np.testing.assert_array_equal(masks['A'], [[False, True, True, False, False]])
        detail = source.diagnostics['cameras']['A']
        self.assertEqual((detail['self_match'], detail['behind_withheld'], detail['foreground_kept']), (1, 1, 1))
        self.assertEqual(detail['valid_pixels'], 3)
        # An obstacle only3mm in front is unresolved under the5mm model margin;
        # tightening the explicit tolerance changes that case, without dilation.
        self.assertTrue(source.masks((frame(.997),), {'A': snapshot()})['A'][0, 0])
        narrow = RobotSelfFilter(depth_tolerance=.001, stride=1)
        self.assertFalse(narrow.masks((frame(.997),), {'A': snapshot()})['A'][0, 0])

    def test_triangle_aabb_overlap_is_not_a_robot_silhouette_mask(self):
        mesh = TriangleMesh(np.array([[-.2, -.2, 1.], [.2, -.2, 1.], [-.2, .2, 1.]]), np.array([[0, 1, 2]]))
        model = snapshot([MeshInstance(mesh, np.eye(4))])
        k = np.array([[5., 0., 1.], [0., 5., 1.], [0., 0., 1.]])
        masks = RobotSelfFilter(stride=1).masks((frame(np.ones((2, 2)), intrinsics=k),), {'A': model})
        self.assertTrue(masks['A'][0, 0])
        self.assertFalse(masks['A'][1, 1])  # Inside AABB, outside actual triangle.

    def test_optical_z_parameter_half_pixels_and_affine_mesh_transform(self):
        # Euclidean distance is1.2806m, while the correct optical depth is1m.
        part = MeshInstance(box_mesh((.7, -.1, 1.), (.9, .1, 1.1)), np.eye(4))
        k = np.array([[1., 0., -.3], [0., 1., .5], [0., 0., 1.]])
        source = RobotSelfFilter(stride=1)
        self.assertTrue(source.masks((frame(1., intrinsics=k),), {'A': snapshot([part])})['A'][0, 0])
        affine = np.diag([2., .5, 1.5, 1.]); affine[2, 3] = .2
        part = MeshInstance(box_mesh(), affine)
        self.assertTrue(source.masks((frame(1.7),), {'A': snapshot([part])})['A'][0, 0])
        rotation = np.array([[0., -1., 0., .3], [1., 0., 0., -.1], [0., 0., 1., .4], [0., 0., 0., 1.]])
        part = MeshInstance(box_mesh(), rotation)
        self.assertTrue(source.masks((frame(1., transform=rotation),), {'A': snapshot([part])})['A'][0, 0])

    def test_nearest_robot_surface_wins_over_farther_link(self):
        parts = [MeshInstance(box_mesh((-.2, -.2, z), (.2, .2, z+.2)), np.eye(4)) for z in (2., 1.)]
        source = RobotSelfFilter(stride=1)
        self.assertTrue(source.masks((frame(2.),), {'A': snapshot(parts)})['A'][0, 0])
        detail = source.diagnostics['cameras']['A']
        self.assertEqual(detail['self_match'], 0)
        self.assertEqual(detail['behind_withheld'], 1)

    def test_open_finger_gap_and_closed_fingers_have_distinct_accurate_masks(self):
        fingers = [MeshInstance(box_mesh((a, -.05, 1.), (b, .05, 1.2)), np.eye(4))
                   for a, b in ((-.12, -.04), (.04, .12))]
        closed = [MeshInstance(box_mesh((a, -.05, 1.), (b, .05, 1.2)), np.eye(4))
                  for a, b in ((-.05, 0.), (0., .05))]
        k = np.array([[10., 0., 1.5], [0., 10., .5], [0., 0., 1.]])
        source = RobotSelfFilter(stride=1)
        np.testing.assert_array_equal(source.masks((frame([[1., 1., 1.]], intrinsics=k),),
            {'A': snapshot(fingers)})['A'], [[True, False, True]])
        new_frame = frame([[1., 1., 1.]], token=(2, .2), intrinsics=k)
        np.testing.assert_array_equal(source.masks((new_frame,),
            {'A': snapshot(closed, token=(2, .2))})['A'], [[False, True, False]])

    def test_camera_specific_views_use_the_same_mesh_without_extra_geometry(self):
        model = snapshot()
        shifted = np.eye(4); shifted[0, 3] = 2.
        opposite = np.diag([-1., 1., -1., 1.]); opposite[2, 3] = 3.
        frames = (frame(camera='A'), frame(camera='B', transform=shifted),
                  frame(1.8, camera='C', transform=opposite))
        source = RobotSelfFilter(stride=1)
        masks = source.masks(frames, dict.fromkeys(('A', 'B', 'C'), model))
        self.assertTrue(masks['A'][0, 0]); self.assertFalse(masks['B'][0, 0]); self.assertTrue(masks['C'][0, 0])
        self.assertEqual(set(source.diagnostics['cameras']), {'A', 'B', 'C'})
        again = source.masks(tuple(reversed(frames)), dict.fromkeys(('C', 'B', 'A'), model))
        for key in masks:
            np.testing.assert_array_equal(masks[key], again[key])

    def test_moving_robot_uses_matching_old_snapshot_instead_of_latest_pose(self):
        mesh = box_mesh()
        old = MeshInstance(mesh, np.eye(4))
        moved = np.eye(4); moved[0, 3] = .5
        new = MeshInstance(mesh, moved)
        source = RobotSelfFilter(stride=1)
        self.assertTrue(source.masks((frame(),), {'A': snapshot([old])})['A'][0, 0])
        current = frame(token=(2, .2))
        self.assertFalse(source.masks((current,), {'A': snapshot([new], token=(2, .2))})['A'][0, 0])
        self.assertIs(old.mesh, new.mesh)  # BVH is reused when link pose changes.
        with self.assertRaisesRegex(ValueError, 'render time/token'):
            source.masks((frame(),), {'A': snapshot([new], token=(2, .2))})

    def test_stale_wrong_token_or_missing_view_geometry_fails_entire_batch(self):
        source = RobotSelfFilter(stride=1)
        for supplied in ({'A': snapshot(stamp=.12)}, {'A': snapshot(token=(2, .1))}, {},
                         {'A': snapshot(), 'extra': snapshot()}):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                source.masks((frame(),), supplied)
            self.assertEqual(source.diagnostics['state'], 'error')
        bad = frame(camera='B'); bad.intrinsics[0, 0] = 0
        with self.assertRaises(ValueError):
            source.masks((frame(), bad), {'A': snapshot(), 'B': snapshot()})
        self.assertEqual(source.diagnostics['cameras'], {})  # Prevalidate all views.

    def test_only_mapping_stride_grid_is_evaluated(self):
        source = RobotSelfFilter(stride=2)
        mask = source.masks((frame(np.ones((3, 3))),), {'A': snapshot()})['A']
        expected = np.zeros((3, 3), bool); expected[::2, ::2] = True
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(source.diagnostics['cameras']['A']['sampled_pixels'], 4)

    def test_selected_pixels_match_full_masks_without_labeling_budget_skips(self):
        source = RobotSelfFilter(stride=2)
        current = frame([[1., 1., .98], [1., 1., 1.], [1.02, 1., np.nan]])
        full = source.masks((current,), {'A': snapshot()})['A']
        selection = {'A': (np.array([2, 0]), np.array([0, 2]))}
        selected = source.masks((current,), {'A': snapshot()}, pixels_by_camera=selection)['A']
        expected = np.zeros((3, 3), bool); expected[2, 0] = full[2, 0]
        expected[0, 2] = full[0, 2]
        np.testing.assert_array_equal(selected, expected)
        self.assertFalse(selected[0, 0])  # A budget skip is not a robot label.
        detail = source.diagnostics['cameras']['A']
        self.assertEqual((detail['sampled_pixels'], detail['self_match'],
                          detail['behind_withheld'], detail['foreground_kept']), (2, 0, 1, 1))
        empty = (np.array([], int), np.array([], int))
        masks = source.masks((current, frame(camera='B')), {'A': snapshot(), 'B': snapshot()},
                             pixels_by_camera={'A': empty, 'B': (np.array([0]), np.array([0]))})
        self.assertFalse(masks['A'].any())
        self.assertTrue(masks['B'][0, 0])
        self.assertEqual(source.diagnostics['cameras']['A']['sampled_pixels'], 0)

    def test_invalid_selected_pixels_fail_before_any_camera_is_intersected(self):
        source = RobotSelfFilter(stride=2)
        frames = (frame(np.ones((3, 3))), frame(np.ones((3, 3)), camera='B'))
        good = (np.array([0]), np.array([0]))
        invalid = [
            (np.array([0., 2.]), np.array([0, 2])),
            (np.array([0, 0]), np.array([0, 0])),
            (np.array([0]), np.array([3])),
            (np.array([-2]), np.array([0])),
            (np.array([1]), np.array([0])),
            (np.array([[0]]), np.array([0])),
            (np.array([0, 2]), np.array([0])),
        ]
        for pair in invalid:
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                source.masks(frames, {'A': snapshot(), 'B': snapshot()},
                             pixels_by_camera={'A': good, 'B': pair})
            self.assertEqual(source.diagnostics['state'], 'error')
            self.assertEqual(source.diagnostics['cameras'], {})
        for selections in ({'A': good}, {'A': good, 'B': good, 'extra': good}):
            with self.assertRaises(ValueError):
                source.masks(frames, {'A': snapshot(), 'B': snapshot()},
                             pixels_by_camera=selections)

    def test_meshes_and_transforms_are_owned_immutable_and_invalid_models_fail(self):
        mesh = box_mesh()
        part = MeshInstance(mesh, np.eye(4))
        for array in (mesh.vertices, mesh.triangles, mesh._lower, part.world_from_local):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(ValueError):
            TriangleMesh(np.array([[np.nan, 0., 0.]]), np.array([[0, 0, 0]]))
        with self.assertRaises(ValueError):
            TriangleMesh(np.zeros((3, 3)), np.array([[0, 1, 2]]))
        with self.assertRaises(ValueError):
            MeshInstance(mesh, np.zeros((4, 4)))
        with self.assertRaises(ValueError):
            RobotSnapshot((), .1)
        with self.assertRaises(ValueError):
            RobotSnapshot((part,), np.nan)
        for options in (dict(stride=0), dict(depth_tolerance=-.1), dict(pose_tolerance=np.inf)):
            with self.assertRaises(ValueError):
                RobotSelfFilter(**options)

    def test_generated_mask_skips_whole_mapper_ray_without_clearing_behind_robot(self):
        now = 100.
        occupancy = OccupancyMap(GridSpec((-.1, -.1, .1), (.1, .1, 2.), .1), clock=lambda: now)
        mapper = RgbdMapper(occupancy, stride=1, clock=lambda: now)
        source = RobotSelfFilter(stride=1)
        current = frame()
        masks = source.masks((current,), {'A': snapshot()})
        self.assertFalse(mapper.update((current,), masks))
        for point in ((0., 0., .5), (0., 0., 1.), (0., 0., 1.5)):
            query = occupancy.query(point)
            self.assertEqual(query.state, OccupancyState.UNKNOWN)
            self.assertFalse(query.observed)
        # A real foreground surface is retained as occupancy, while rays still
        # stop before both the robot and the volume behind it.
        current = frame(.8, token=(2, .2))
        masks = source.masks((current,), {'A': snapshot(token=(2, .2))})
        self.assertTrue(mapper.update((current,), masks))
        self.assertEqual(occupancy.query((0., 0., .8)).state, OccupancyState.OCCUPIED)
        self.assertFalse(occupancy.query((0., 0., 1.5)).observed)


if __name__ == '__main__':
    unittest.main()
