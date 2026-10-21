from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame, FrameSynchronizer
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, fuse_clouds, project_depth, voxelize
from robot_skill_stack.integrations.isaac.config import SceneGeometry


def camera(number=1, stamp=.1):
    result = Mock()
    result.get_frame_token.return_value = (str(number), str(stamp))
    result.get_depth.return_value = np.ones((4, 4))
    result.get_rgb.return_value = None
    result.get_intrinsics.return_value = np.eye(3)
    result.get_world_from_camera_transform.return_value = np.eye(4)
    return result


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = camera(), camera()
        self.rig = FrameSynchronizer({'A': self.a, 'B': self.b})

    def test_projection_matches_known_pinhole_rigid_transform_and_invalid_depth(self):
        transform = np.eye(4)
        transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        transform[:3, 3] = [2, 3, 4]
        frame = DepthFrame('A', (1, .1), np.array([[2., np.nan], [0., -1.]]),
                           np.array([[2, 0, .5], [0, 2, .5], [0, 0, 1.]]), transform, 10.)
        points, pixels = project_depth(frame, stride=1)
        np.testing.assert_allclose(points, [[2, 3, 6]])
        np.testing.assert_array_equal(pixels, [[0, 0]])
        with self.assertRaises(ValueError):
            replace(frame, world_from_camera=np.zeros((4, 4)))

    def test_distinct_synchronized_frames_are_copied_and_use_monotonic_time(self):
        with patch('time.monotonic', return_value=100.):
            frames = self.rig.capture()
        self.assertEqual([f.acquired_at for f in frames], [100., 100.])
        self.a.get_depth.return_value[:] = 7
        self.assertTrue((frames[0].depth == 1).all())
        self.assertIsNone(self.rig.capture())

    def test_missing_frozen_and_skewed_cameras_are_explicit_failures(self):
        self.rig.capture()
        self.a.get_frame_token.return_value = ('2', '.2')
        with self.assertRaisesRegex(RuntimeError, 'frozen'):
            self.rig.capture()
        self.b.get_frame_token.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'Missing'):
            self.rig.capture()
        self.b.get_frame_token.return_value = ('2', '.3')
        with self.assertRaisesRegex(RuntimeError, 'skew'):
            self.rig.capture()
        self.assertEqual(self.rig.status['state'], 'unsynchronized')

    def test_clock_reversal_and_buffer_update_cannot_be_fresh_samples(self):
        self.rig.capture()
        for cam in (self.a, self.b):
            cam.get_frame_token.return_value = ('1', '.05')
        with self.assertRaisesRegex(RuntimeError, 'regressed'):
            self.rig.capture()
        self.rig = FrameSynchronizer({'A': self.a, 'B': self.b})
        self.b.get_frame_token.side_effect = [('2', '.2'), ('3', '.3')]
        self.a.get_frame_token.return_value = ('2', '.2')
        with self.assertRaisesRegex(RuntimeError, 'updated during'):
            self.rig.capture()

    def test_invalid_buffers_are_consumed_and_all_resources_close_once(self):
        self.b.get_depth.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'depth'):
            self.rig.capture()
        self.b.get_depth.return_value = np.ones((4, 4))
        self.assertIsNone(self.rig.capture())
        self.b.close.side_effect = RuntimeError('cleanup failure')
        with self.assertRaisesRegex(RuntimeError, 'cleanup failure'):
            self.rig.close()
        self.rig.close()
        self.a.close.assert_called_once()
        self.b.close.assert_called_once()

    def test_all_invalid_depth_is_a_source_failure_not_silent_single_view_fusion(self):
        self.b.get_depth.return_value[:] = np.nan
        with self.assertRaisesRegex(RuntimeError, 'Invalid multi-view depth'):
            self.rig.capture()
        self.assertEqual(self.rig.status['state'], 'invalid_depth')
        self.assertIsNone(self.rig.capture())

    def test_voxels_bound_duplicate_density_and_crop_removes_table_and_background(self):
        points = np.array([[.4, 0, .02], [.4001, 0, .02], [.45, .1, .05]])
        fused = fuse_clouds([points, np.repeat(points, 100, axis=0)])
        self.assertEqual(len(fused), 2)
        np.testing.assert_allclose(fused, voxelize(points), atol=1e-12)
        scene = SceneGeometry(0., [.2, -.4, 0], [.7, .4, .22])
        cloud = np.r_[fused, [[.5, 0, 0], [1, 0, .1], [np.nan, 0, 0]]]
        np.testing.assert_allclose(crop_workspace(cloud, scene), fused)


class VariableCameraAcquisitionTests(unittest.TestCase):
    def make_rig(self, count):
        cameras = {f'/World/Camera{index+1}': camera() for index in range(count)}
        return FrameSynchronizer(cameras), cameras

    @staticmethod
    def advance(cameras, number=2, stamp=.2):
        for cam in cameras.values():
            cam.get_frame_token.return_value = (str(number), str(stamp))

    def test_three_and_five_camera_snapshots_preserve_every_calibration_and_timestamp(self):
        for count in (3, 5):
            with self.subTest(count=count):
                rig, cameras = self.make_rig(count)
                for index, cam in enumerate(cameras.values()):
                    cam.get_depth.return_value = np.full((4, 4), index+1.)
                    cam.get_rgb.return_value = np.full((4, 4, 3), index, dtype=np.uint8)
                    cam.get_intrinsics.return_value = np.diag([index+2., index+3., 1.])
                    cam.get_world_from_camera_transform.return_value[:3, 3] = [index, -index, .5]
                    cam.get_frame_token.return_value = ('1', str(.1+index*.001))
                with patch('time.monotonic', return_value=123.):
                    frames = rig.capture()
                self.assertEqual(len(frames), count)
                self.assertEqual([frame.camera_id for frame in frames], list(cameras))
                self.assertEqual([frame.acquired_at for frame in frames], [123.]*count)
                for index, (frame, cam) in enumerate(zip(frames, cameras.values())):
                    self.assertEqual(frame.token, (1., .1+index*.001))
                    np.testing.assert_array_equal(frame.depth, cam.get_depth.return_value)
                    np.testing.assert_array_equal(frame.rgb, cam.get_rgb.return_value)
                    np.testing.assert_array_equal(frame.intrinsics, cam.get_intrinsics.return_value)
                    np.testing.assert_array_equal(frame.world_from_camera,
                                                  cam.get_world_from_camera_transform.return_value)
                    cam.get_depth.return_value[:] = 99
                    cam.get_rgb.return_value[:] = 99
                    cam.get_intrinsics.return_value[0, 0] = 99
                    cam.get_world_from_camera_transform.return_value[0, 3] = 99
                    self.assertEqual(frame.depth[0, 0], index+1.)
                    self.assertEqual(frame.rgb[0, 0, 0], index)
                    self.assertEqual(frame.intrinsics[0, 0], index+2.)
                    self.assertEqual(frame.world_from_camera[0, 3], index)
                self.assertEqual(rig.status['state'], 'synchronized')

    def test_repeated_three_and_five_camera_buffers_publish_nothing_and_do_not_read_depth(self):
        for count in (3, 5):
            with self.subTest(count=count):
                rig, cameras = self.make_rig(count)
                self.assertEqual(len(rig.capture()), count)
                for _ in range(2):
                    self.assertIsNone(rig.capture())
                    self.assertEqual(rig.status['state'], 'unchanged')
                    self.assertEqual(rig.status['cameras'], list(cameras))
                for cam in cameras.values():
                    cam.get_depth.assert_called_once()

    def test_frozen_third_camera_blocks_the_whole_batch_until_all_are_fresh(self):
        for count in (3, 5):
            with self.subTest(count=count):
                rig, cameras = self.make_rig(count)
                rig.capture()
                self.advance(cameras)
                cameras['/World/Camera3'].get_frame_token.return_value = ('1', '.1')
                with self.assertRaisesRegex(RuntimeError, 'frozen.*Camera3'):
                    rig.capture()
                self.assertEqual(rig.status, dict(state='frozen', cameras=['/World/Camera3']))
                for cam in cameras.values():
                    cam.get_depth.assert_called_once()
                self.advance(cameras)
                self.assertEqual(len(rig.capture()), count)

    def test_missing_third_render_frame_is_not_partial_capture(self):
        rig, cameras = self.make_rig(5)
        cameras['/World/Camera3'].get_frame_token.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'Missing.*Camera3'):
            rig.capture()
        self.assertEqual(rig.status, dict(state='missing', cameras=['/World/Camera3']))
        for cam in cameras.values():
            cam.get_depth.assert_not_called()
        self.advance(cameras, number=1, stamp=.1)
        self.assertEqual(len(rig.capture()), 5)

    def test_missing_third_depth_consumes_the_attempt_and_never_relabels_old_buffers(self):
        rig, cameras = self.make_rig(5)
        cameras['/World/Camera3'].get_depth.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'depth.*Camera3'):
            rig.capture()
        self.assertEqual(rig.status, dict(state='missing_depth', cameras=['/World/Camera3']))
        cameras['/World/Camera3'].get_depth.return_value = np.ones((4, 4))
        self.assertIsNone(rig.capture())
        self.advance(cameras)
        self.assertEqual(len(rig.capture()), 5)

    def test_invalid_third_or_fifth_depth_blocks_capture_and_consumes_every_token(self):
        for index in (3, 5):
            for invalid in (np.nan, np.inf, 0., -1.):
                with self.subTest(camera=index, invalid=invalid):
                    rig, cameras = self.make_rig(5)
                    key = f'/World/Camera{index}'
                    cameras[key].get_depth.return_value[:] = invalid
                    with self.assertRaisesRegex(RuntimeError, f'Invalid.*Camera{index}'):
                        rig.capture()
                    self.assertEqual(rig.status, dict(state='invalid_depth', cameras=[key]))
                    cameras[key].get_depth.return_value[:] = 1.
                    self.assertIsNone(rig.capture())
                    self.advance(cameras)
                    self.assertEqual(len(rig.capture()), 5)

    def test_skew_from_third_camera_is_checked_across_the_entire_rig(self):
        rig, cameras = self.make_rig(5)
        cameras['/World/Camera3'].get_frame_token.return_value = ('1', '.15')
        with self.assertRaisesRegex(RuntimeError, 'skew'):
            rig.capture()
        self.assertEqual(rig.status['state'], 'unsynchronized')
        self.assertAlmostEqual(rig.status['skew_seconds'], .05)
        for cam in cameras.values():
            cam.get_depth.assert_not_called()
        self.advance(cameras, number=1, stamp=.1)
        self.assertEqual(len(rig.capture()), 5)

    def test_third_camera_regressed_render_number_or_timestamp_blocks_new_batch(self):
        for token in (('1', '.2'), ('2', '.1'), ('2', '.05')):
            with self.subTest(token=token):
                rig, cameras = self.make_rig(5)
                rig.capture()
                self.advance(cameras)
                cameras['/World/Camera3'].get_frame_token.return_value = token
                with self.assertRaisesRegex(RuntimeError, 'regressed'):
                    rig.capture()
                self.assertEqual(rig.status['state'], 'clock_regression')
                for cam in cameras.values():
                    cam.get_depth.assert_called_once()

    def test_third_camera_update_during_acquisition_cannot_return_a_mixed_snapshot(self):
        rig, cameras = self.make_rig(5)
        cameras['/World/Camera3'].get_frame_token.side_effect = [('1', '.1'), ('2', '.2')]
        with self.assertRaisesRegex(RuntimeError, 'updated during'):
            rig.capture()
        self.assertEqual(rig.status['state'], 'changed_during_acquisition')
        cameras['/World/Camera3'].get_frame_token.side_effect = None
        self.advance(cameras)
        self.assertEqual(len(rig.capture()), 5)

    def test_every_camera_calibration_is_validated_before_returning_a_batch(self):
        rig, cameras = self.make_rig(5)
        cameras['/World/Camera5'].get_intrinsics.return_value = np.zeros((3, 3))
        with self.assertRaisesRegex(ValueError, 'calibration'):
            rig.capture()
        cameras['/World/Camera5'].get_intrinsics.return_value = np.eye(3)
        self.assertIsNone(rig.capture())

    def test_all_cameras_close_once_even_when_multiple_cleanup_operations_fail(self):
        for count in (3, 5):
            with self.subTest(count=count):
                rig, cameras = self.make_rig(count)
                order = []
                def close(key):
                    order.append(key)
                    if key in ('/World/Camera2', f'/World/Camera{count}'):
                        raise RuntimeError(f'cleanup failed for {key}')
                for key, cam in cameras.items():
                    cam.close.side_effect = lambda key=key: close(key)
                with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                    rig.close()
                rig.close()
                self.assertEqual(order, list(reversed(cameras)))
                for cam in cameras.values():
                    cam.close.assert_called_once()
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    rig.capture()


if __name__ == '__main__':
    unittest.main()
