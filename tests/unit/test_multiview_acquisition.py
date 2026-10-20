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
