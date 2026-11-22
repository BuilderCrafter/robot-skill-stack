import unittest
from unittest.mock import Mock

import numpy as np

from robot_skill_stack.world.perception_multiview.shared_frames import SharedFramePool


def camera():
    return Mock(get_frame_token=Mock(return_value=(1., .1)),
                get_depth=Mock(return_value=np.ones((2, 3))),
                get_rgb=Mock(return_value=None),
                get_intrinsics=Mock(return_value=np.eye(3)),
                get_world_from_camera_transform=Mock(return_value=np.eye(4)))


class SharedFramesTests(unittest.TestCase):
    def setUp(self):
        self.cameras = {key: camera() for key in 'ABC'}
        self.clock = Mock(return_value=10.)
        self.pool = SharedFramePool(self.cameras, clock=self.clock)

    def advance(self, *keys):
        self.pool.begin_update()
        for key in keys:
            self.cameras[key].get_frame_token.return_value = (2., .2)

    def test_partial_overlap_reuses_exact_owned_frame_and_original_timestamp(self):
        semantic = self.pool.batch('semantic', 'AB')
        self.clock.return_value = 11.
        self.pool.begin_update()
        mapping = self.pool.batch('mapping', 'AC')
        self.assertIs(semantic[0], mapping[0])
        self.assertEqual(mapping[0].acquired_at, 10.)
        self.assertEqual(mapping[1].acquired_at, 11.)
        self.assertEqual(self.cameras['A'].get_depth.call_count, 1)
        self.assertEqual(self.pool.shared_reads, 1)
        self.cameras['A'].get_depth.return_value[:] = 9
        np.testing.assert_array_equal(mapping[0].depth, 1)
        with self.assertRaises(ValueError):
            mapping[0].depth.setflags(write=True)

    def test_frozen_extra_camera_does_not_block_semantic_subset(self):
        self.pool.batch('semantic', 'AB')
        self.pool.batch('mapping', 'AC')
        self.advance('A', 'B')
        with self.assertRaisesRegex(RuntimeError, 'frozen'):
            self.pool.batch('mapping', 'AC')
        self.assertIsNotNone(self.pool.batch('semantic', 'AB'))

    def test_failed_complete_attempt_cannot_be_repaired_and_replayed(self):
        self.cameras['B'].get_depth.return_value = None
        with self.assertRaises(RuntimeError):
            self.pool.batch('semantic', 'AB')
        self.cameras['B'].get_depth.return_value = np.ones((2, 3))
        self.assertIsNone(self.pool.batch('semantic', 'AB'))
        with self.assertRaises(RuntimeError):
            self.pool.batch('another-consumer', 'AB')
        self.advance('A', 'B')
        self.assertIsNotNone(self.pool.batch('semantic', 'AB'))

    def test_no_duplicates_skew_and_clock_regression(self):
        self.pool.batch('semantic', 'AB')
        self.assertIsNone(self.pool.batch('semantic', 'AB'))
        self.advance('A', 'B')
        self.cameras['B'].get_frame_token.return_value = (2., .4)
        with self.assertRaisesRegex(RuntimeError, 'skew'):
            self.pool.batch('semantic', 'AB')
        self.cameras['B'].get_frame_token.return_value = (2., .2)
        self.assertIsNotNone(self.pool.batch('semantic', 'AB'))
        self.cameras['B'].get_frame_token.return_value = (.5, .05)
        self.cameras['A'].get_frame_token.return_value = (.5, .05)
        with self.assertRaisesRegex(RuntimeError, 'regressed'):
            self.pool.batch('semantic', 'AB')
        self.pool.reset()
        self.assertIsNotNone(self.pool.batch('semantic', 'AB'))

    def test_close_never_closes_borrowed_cameras(self):
        self.pool.close()
        self.pool.close()
        for value in self.cameras.values():
            value.close.assert_not_called()
        with self.assertRaises(RuntimeError):
            self.pool.batch('semantic', 'AB')

    def test_stop_and_play_cannot_retimestamp_the_retained_render_buffer(self):
        self.pool.batch('semantic', 'AB')
        self.pool.reset()
        self.pool.reset()
        self.clock.return_value = 20.
        self.assertIsNone(self.pool.batch('semantic', 'AB'))
        self.assertEqual(self.cameras['A'].get_depth.call_count, 1)
        self.advance('A', 'B')
        self.assertEqual(self.pool.batch('semantic', 'AB')[0].acquired_at, 20.)

    def test_inflight_pre_stop_exposure_cannot_seed_the_restarted_clock(self):
        self.pool.batch('semantic', 'AB', simulation_time=.1)
        self.pool.reset()
        self.pool.reset()
        for key in 'AB':
            self.cameras[key].get_frame_token.return_value = (2., 12.)
        with self.assertRaisesRegex(RuntimeError, 'current simulation generation'):
            self.pool.batch('semantic', 'AB', simulation_time=.05)
        # The legitimate post-reset stream has a lower simulation timestamp.
        # It is still the first admitted exposure in the new generation.
        for key in 'AB':
            self.cameras[key].get_frame_token.return_value = (3., .1)
        self.clock.return_value = 20.
        frames = self.pool.batch('semantic', 'AB', simulation_time=.12)
        self.assertEqual(frames[0].token, (3., .1))
        self.assertEqual(frames[0].acquired_at, 20.)

    def test_frozen_rejected_exposure_stays_quarantined_when_simulation_catches_up(self):
        self.pool.batch('semantic', 'AB', simulation_time=.1)
        self.pool.reset()
        for key in 'AB':
            self.cameras[key].get_frame_token.return_value = (2., 12.)
        with self.assertRaises(RuntimeError):
            self.pool.batch('semantic', 'AB', simulation_time=.05)
        self.clock.return_value = 30.
        self.assertIsNone(self.pool.batch('semantic', 'AB', simulation_time=13.))
        self.assertEqual(self.cameras['A'].get_depth.call_count, 1)
        for key in 'AB':
            self.cameras[key].get_frame_token.return_value = (3., 13.1)
        self.assertIsNotNone(self.pool.batch('semantic', 'AB', simulation_time=13.1))
        self.assertEqual(len(self.pool._quarantined_tokens), 2)


if __name__ == '__main__':
    unittest.main()
