"""Fresh-source selection, fixed-time queries and retained camera evidence."""
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.world.common.camera import DepthFrame
from robot_skill_stack.world.spatial_mapping.frozen import FrozenRgbdSourceError, capture_frozen_rgbd
from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractionSettings
from robot_skill_stack.world.spatial_mapping.occupancy import OccupancyState
from robot_skill_stack.world.spatial_mapping.robot_filter import MeshInstance, RobotSnapshot, TriangleMesh
from robot_skill_stack.world.spatial_mapping.runtime import MappingInput, MappingService
from tests.unit.test_spatial_mapping_runtime import ManualWorker, settings


class FrozenRgbdCaptureTests(unittest.TestCase):
    def service(self, *, maximum=64, ttl=5., support=None, **changes):
        values = dict(enabled=True, source='rgbd', automatic_robot_filter=True, stride=1,
            bounds_min=(-.05,-.05,.8), bounds_max=(.05,.05,1.05), voxel_size=.01,
            max_frame_age=1., stale_after=1., evidence_ttl=ttl, camera_prim_paths=('/A','/B'))
        values.update(changes)
        self.clock = [100.2]
        source = MappingService(settings(**values), worker_factory=ManualWorker, clock=lambda:self.clock[0],
            obstacle_settings=ObstacleExtractionSettings(maximum, support))
        self.addCleanup(source.close)
        return source

    def payload(self, source, number=1, acquired=100., depth=1.):
        k = np.array([[100.,0.,1.5],[0.,100.,1.5],[0.,0.,1.]])
        frames = tuple(DepthFrame(key, (number, number*.1), np.full((3,3),depth), k.copy(),
            np.eye(4), acquired) for key in ('/A','/B'))
        mesh = TriangleMesh(np.array([[-.2,-.2,2.],[.2,-.2,2.],[.2,.2,2.],[-.2,.2,2.]]),
            np.array([[0,1,2],[0,2,3]]))
        robot = {frame.camera_id:RobotSnapshot((MeshInstance(mesh,np.eye(4)),),
            frame.token[1], frame.token) for frame in frames}
        return MappingInput(generation=source.generation, frames=frames, robot_snapshots=robot)

    def publish(self, source, number=1, acquired=100., depth=1.):
        source.submit(self.payload(source, number, acquired, depth))
        source._worker.finish()
        result = source.poll()
        self.assertTrue(result.accepted)
        return result

    def prepared(self, **options):
        source = self.service(**options)
        self.publish(source)
        self.publish(source,2,100.1)
        self.assertTrue(source.snapshot().obstacles.boxes)
        return source

    def test_fresh_publication_reuses_geometry_and_original_camera_records(self):
        source = self.prepared()
        view = source.snapshot()
        with patch.object(source, 'snapshot', return_value=view):
            self.assertIs(capture_frozen_rgbd(source), view)
        self.assertEqual(view.timestamp,100.1)
        self.assertEqual(view.evaluated_at,100.2)
        self.assertAlmostEqual(view.evaluated_at-view.timestamp,.1)
        self.assertEqual([f['camera_id'] for f in view.source_frames],['/A','/B'])
        self.assertEqual(view.source_frames[0]['token'],(2,.2))
        self.assertEqual(view.source_frames[0]['acquired_at'],100.1)
        np.testing.assert_array_equal(view.source_frames[0]['world_from_camera'],np.eye(4))
        points = [[0.,0.,.9],[0.,0.,1.],[.04,.04,.9],[.5,0.,1.]]
        np.testing.assert_array_equal(view.query_many(points).states,
            [OccupancyState.FREE,OccupancyState.OCCUPIED,OccupancyState.UNKNOWN,OccupancyState.OUT_OF_BOUNDS])

    def test_missing_stale_and_failed_sources_are_refused(self):
        for source in (None, self.service()):
            with self.assertRaises(FrozenRgbdSourceError):
                capture_frozen_rgbd(source)
        source = self.prepared()
        self.clock[0] = 102.
        with self.assertRaises(FrozenRgbdSourceError) as caught:
            capture_frozen_rgbd(source)
        self.assertEqual(caught.exception.code,'stale')
        source = self.prepared()
        source.note_source_failure('camera acquisition failed')
        with self.assertRaises(FrozenRgbdSourceError) as caught:
            capture_frozen_rgbd(source)
        self.assertEqual(caught.exception.code,'unavailable')

    def test_empty_expired_and_over_budget_cuboids_are_refused(self):
        source = self.service()
        depth = np.full((3,3),np.nan)
        depth[1,1] = 1.
        self.publish(source,depth=depth)
        with self.assertRaises(FrozenRgbdSourceError):
            capture_frozen_rgbd(source)
        source = self.prepared(ttl=.2)
        self.clock[0] = min(b.expires_at for b in source.snapshot().obstacles.boxes)
        with self.assertRaises(FrozenRgbdSourceError) as caught:
            capture_frozen_rgbd(source)
        self.assertEqual(caught.exception.code,'unusable')
        source = self.service()
        self.publish(source,depth=np.array([[1.,1.,1.],[.98,.98,.98],[.98,.98,.98]]))
        self.assertGreater(len(source.snapshot().obstacles.boxes),1)
        with self.assertRaises(FrozenRgbdSourceError):
            capture_frozen_rgbd(source,max_obstacles=1)

    def test_later_publication_reset_and_close_do_not_change_the_frozen_world(self):
        source = self.prepared()
        view = capture_frozen_rgbd(source)
        boxes = view.obstacles.boxes
        self.publish(source,3,100.15,depth=1.01)
        source.reset()
        source.close()
        self.clock[0] = 1000.
        self.assertIs(view.obstacles.boxes,boxes)
        self.assertEqual(view.timestamp,100.1)
        self.assertAlmostEqual(view.evaluated_at-view.timestamp,.1)
        self.assertEqual(view.source_frames[0]['token'],(2,.2))
        self.assertEqual(view.query([0.,0.,1.]).state,OccupancyState.OCCUPIED)


if __name__ == '__main__':
    unittest.main()
