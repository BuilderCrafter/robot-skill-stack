import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.integrations.isaac.config import SceneGeometry
from robot_skill_stack.world.common.camera import DepthFrame, FrameSynchronizer
from robot_skill_stack.world.common.point_clouds import crop_workspace, fuse_clouds, project_depth, voxelize


def camera(number=1, stamp=.1):
    return Mock(get_frame_token=Mock(return_value=(number, stamp)),
                get_depth=Mock(return_value=np.ones((4, 4))),
                get_rgb=Mock(return_value=None),
                get_intrinsics=Mock(return_value=np.eye(3)),
                get_world_from_camera_transform=Mock(return_value=np.eye(4)))


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.cameras = {key: camera() for key in 'ABC'}
        self.rig = FrameSynchronizer(self.cameras)

    def advance(self, number=2, stamp=.2):
        for cam in self.cameras.values():
            cam.get_frame_token.return_value = (number, stamp)

    def test_projection_matches_calibration_and_filters_invalid_depth(self):
        transform = np.eye(4)
        transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        transform[:3, 3] = [2, 3, 4]
        frame = DepthFrame('A', (1, .1), np.array([[2., np.nan], [0., -1.]]),
                           np.array([[2, 0, .5], [0, 2, .5], [0, 0, 1.]]), transform, 10.)
        points, pixels = project_depth(frame, stride=1)
        np.testing.assert_allclose(points, [[2, 3, 6]])
        np.testing.assert_array_equal(pixels, [[0, 0]])

    def test_capture_preserves_all_camera_data_and_owns_independent_buffers(self):
        for index, cam in enumerate(self.cameras.values()):
            cam.get_depth.return_value[:] = index + 1
            cam.get_rgb.return_value = np.full((4, 4, 3), index, dtype=np.uint8)
            cam.get_intrinsics.return_value = np.diag([index + 2., index + 3., 1.])
            cam.get_world_from_camera_transform.return_value[:3, 3] = [index, -index, .5]
            cam.get_frame_token.return_value = (1, .1 + index * .001)
        with patch('time.monotonic', return_value=123.):
            frames = self.rig.capture()
        self.assertEqual([frame.camera_id for frame in frames], list(self.cameras))
        for index, (frame, cam) in enumerate(zip(frames, self.cameras.values())):
            self.assertEqual(frame.token, (1., .1 + index * .001))
            self.assertEqual(frame.acquired_at, 123.)
            for name, getter in (('depth', cam.get_depth), ('rgb', cam.get_rgb),
                                 ('intrinsics', cam.get_intrinsics),
                                 ('world_from_camera', cam.get_world_from_camera_transform)):
                expected = getter.return_value.copy()
                np.testing.assert_array_equal(getattr(frame, name), expected)
                getter.return_value[:] = 99
                np.testing.assert_array_equal(getattr(frame, name), expected)

    def test_repeated_batches_read_nothing_and_a_frozen_camera_blocks_the_batch(self):
        self.assertEqual(len(self.rig.capture()), 3)
        self.assertIsNone(self.rig.capture())
        self.cameras['A'].get_frame_token.return_value = (2, .2)
        with self.assertRaisesRegex(RuntimeError, 'frozen'):
            self.rig.capture()
        for cam in self.cameras.values():
            cam.get_depth.assert_called_once()
        self.advance()
        self.assertEqual(len(self.rig.capture()), 3)

    def test_missing_or_unrendered_camera_blocks_capture_before_buffer_reads(self):
        for token in (None, (0, 0), (1, np.nan)):
            with self.subTest(token=token):
                self.cameras['C'].get_frame_token.return_value = token
                with self.assertRaisesRegex(RuntimeError, 'Missing'):
                    self.rig.capture()
        for cam in self.cameras.values():
            cam.get_depth.assert_not_called()
        self.advance()
        self.assertEqual(len(self.rig.capture()), 3)

    def test_missing_or_unusable_depth_consumes_the_complete_attempt(self):
        for depth in (None, np.full((4, 4), np.nan), np.zeros((4, 4))):
            with self.subTest(depth=depth):
                self.advance(1, .1)
                rig = FrameSynchronizer(self.cameras)
                self.cameras['C'].get_depth.return_value = depth
                with self.assertRaisesRegex(RuntimeError, 'depth'):
                    rig.capture()
                self.cameras['C'].get_depth.return_value = np.ones((4, 4))
                self.assertIsNone(rig.capture())
                self.advance()
                self.assertEqual(len(rig.capture()), 3)

    def test_skew_and_regressed_clocks_cannot_refresh_observations(self):
        self.cameras['C'].get_frame_token.return_value = (1, .15)
        with self.assertRaisesRegex(RuntimeError, 'skew'):
            self.rig.capture()
        for cam in self.cameras.values():
            cam.get_depth.assert_not_called()
        self.advance(1, .1)
        self.rig.capture()
        for token in ((1, .2), (2, .05)):
            with self.subTest(token=token):
                self.advance()
                self.cameras['C'].get_frame_token.return_value = token
                with self.assertRaisesRegex(RuntimeError, 'regressed'):
                    self.rig.capture()
        for cam in self.cameras.values():
            cam.get_depth.assert_called_once()
        self.advance()
        self.assertEqual(len(self.rig.capture()), 3)

    def test_camera_advancing_during_capture_invalidates_the_attempt(self):
        self.cameras['C'].get_frame_token.side_effect = [(1, .1), (2, .2)]
        with self.assertRaisesRegex(RuntimeError, 'updated during'):
            self.rig.capture()
        self.cameras['C'].get_frame_token.side_effect = None
        self.assertIsNone(self.rig.capture())
        self.advance()
        self.assertEqual(len(self.rig.capture()), 3)

    def test_bad_calibration_is_rejected_and_the_attempt_cannot_be_replayed(self):
        for getter, invalid in (
            (self.cameras['C'].get_intrinsics, np.zeros((3, 3))),
            (self.cameras['C'].get_world_from_camera_transform, np.diag([2., 1., 1., 1.])),
            (self.cameras['C'].get_world_from_camera_transform, np.diag([-1., 1., 1., 1.])),
        ):
            with self.subTest(calibration=invalid):
                rig = FrameSynchronizer(self.cameras)
                original, getter.return_value = getter.return_value, invalid
                with self.assertRaisesRegex(ValueError, 'calibration'):
                    rig.capture()
                getter.return_value = original
                self.assertIsNone(rig.capture())

    def test_all_owned_cameras_close_once_and_the_first_error_is_preserved(self):
        first = RuntimeError('first cleanup failure')
        self.cameras['C'].close.side_effect = first
        self.cameras['B'].close.side_effect = RuntimeError('second cleanup failure')
        with self.assertRaises(RuntimeError) as raised:
            self.rig.close()
        self.assertIs(raised.exception, first)
        self.rig.close()
        for cam in self.cameras.values():
            cam.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.rig.capture()

    def test_voxels_bound_duplicate_density_and_crop_removes_table_and_background(self):
        points = np.array([[.4, 0, .02], [.4001, 0, .02], [.45, .1, .05]])
        fused = fuse_clouds([points, np.repeat(points, 100, axis=0)])
        self.assertEqual(len(fused), 2)
        np.testing.assert_allclose(fused, voxelize(points), atol=1e-12)
        scene = SceneGeometry(0., [.2, -.4, 0], [.7, .4, .22])
        cloud = np.r_[fused, [[.5, 0, 0], [1, 0, .1], [np.nan, 0, 0]]]
        np.testing.assert_allclose(crop_workspace(cloud, scene), fused)


if __name__ == '__main__':
    unittest.main()
