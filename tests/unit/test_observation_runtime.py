"""Execute the production owner coordinator against recording SDK boundaries."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.runtime.worker import WorkResult
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.provider import ObservationBatch
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.common.camera import DepthFrame


class ControlledWorker:
    def __init__(self, process, **kwargs):
        self.process, self.pending, self.result = process, None, None
        self.closed = False

    def submit(self, payload, tag=None):
        self.pending = payload, tag
        return True

    def finish(self, error=None):
        payload, tag = self.pending
        self.result = WorkResult(tag, self.process(payload) if error is None else None, error, 100., 100., 100.1)
        self.pending = None

    def poll(self):
        result, self.result = self.result, None
        return result

    def invalidate(self):
        self.pending = self.result = None

    def close(self):
        self.closed = True
        self.invalidate()


class ObservationRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.now = patch('time.monotonic', return_value=100.).start()
        self.addCleanup(patch.stopall)
        self.stage = object()
        usd = ModuleType('omni.usd')
        usd.get_context = lambda: NS(get_stage=lambda: self.stage)
        omni = ModuleType('omni')
        omni.usd = usd
        name = 'observation_runtime_under_test'
        path = Path('robot_skill_stack/integrations/isaac/observation_runtime.py')
        spec = importlib.util.spec_from_file_location(name, path)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'omni': omni, 'omni.usd': usd}):
            spec.loader.exec_module(self.module)
        self.config = load_scene_config('config/scenes/multiview_playground.toml')
        self.frames = tuple(DepthFrame(key, (1., .1), np.ones((2, 3)), np.eye(3),
            np.eye(4), 100.) for key in self.config.multi_view.camera_prim_paths)
        self.pool = NS(batch=Mock(return_value=self.frames), reset=Mock(),
            begin_update=Mock(), close=Mock(), cameras={'A': NS(reset_clock=Mock(), close=Mock())},
            status={'semantic': {}})
        self.provider = NS(rig=NS(status={}), processor=NS(compute=lambda payload: payload),
            acquire=Mock(side_effect=lambda frames: NS(timestamp=min(f.acquired_at for f in frames))),
            associate=Mock(side_effect=lambda result, context: ObservationBatch([
                ObjectObservation('cube', pose=Pose([.4, 0, .025]))], result.timestamp)),
            forget=Mock(), close=Mock())
        self.model = WorldModel()
        self.updater = WorldModelUpdater(self.model, self.provider)
        self.world = NS(is_playing=Mock(return_value=True), current_time=1.)
        self.owned_camera = NS(close=Mock())
        self.runtime = self.make_runtime()

    def make_runtime(self, **kwargs):
        runtime = self.module.ObservationRuntime(self.world, self.config, self.updater,
            pool=self.pool, stage=self.stage, clock=lambda: self.now.return_value,
            worker_factory=ControlledWorker, owned_cameras=(self.owned_camera,), **kwargs)
        self.addCleanup(runtime.close)
        return runtime

    def submit_then_complete(self):
        self.runtime.tick()
        self.runtime._semantic_worker.finish()
        self.pool.batch.return_value = None
        self.now.return_value = 100.1

    def test_semantic_publication_and_worker_failure(self):
        self.submit_then_complete()
        self.assertFalse(self.model.objects())
        self.runtime.tick()
        self.assertEqual(self.model.require('cube').last_seen, 100.)
        self.now.return_value, self.pool.batch.return_value = 100.3, self.frames
        self.runtime.tick()
        self.runtime._semantic_worker.finish(error='compute failed')
        self.runtime.tick()
        self.provider.associate.assert_called_once()
        self.assertEqual(self.updater.last_error, 'compute failed')

    def test_context_changes_and_expiry_are_rejected_before_tracking(self):
        self.model.register(WorldObject('other'))
        self.model.require('other').last_seen = 100.
        self.submit_then_complete()
        self.model.set_held('other')
        self.model.set_held(None)
        self.runtime.tick()
        self.provider.associate.assert_not_called()
        self.now.return_value, self.pool.batch.return_value = 100.3, self.frames
        self.runtime.tick()  # Submit again with the current context.
        self.runtime._semantic_worker.finish()
        self.now.return_value, self.pool.batch.return_value = 101.1, None
        self.runtime.tick()
        self.provider.associate.assert_not_called()
        self.assertFalse(self.model.exists('cube'))

    def test_acquisition_failure_invalidates_older_success_before_polling(self):
        self.submit_then_complete()
        self.now.return_value = 100.3
        self.pool.batch.side_effect = RuntimeError('camera failed')
        self.runtime.tick()
        self.provider.associate.assert_not_called()
        self.assertIn('camera failed', self.updater.last_error)

    def test_stop_play_discards_old_work_and_requires_fresh_acquisition(self):
        self.submit_then_complete()
        self.world.is_playing.return_value = False
        self.runtime.tick()
        self.world.is_playing.return_value = True
        self.runtime.tick()
        self.provider.associate.assert_not_called()
        self.assertEqual(self.pool.reset.call_count, 2)
        self.pool.cameras['A'].reset_clock.assert_called_once()
        self.now.return_value = 100.3
        self.pool.batch.return_value = tuple(replace(frame, token=(2., .2), acquired_at=100.3) for frame in self.frames)
        self.runtime.tick()
        self.runtime._semantic_worker.finish()
        self.runtime.tick()
        self.assertTrue(self.model.exists('cube'))

    def test_close_releases_owned_resources_and_leaves_borrowed_cameras_to_provider(self):
        self.submit_then_complete()
        self.runtime.close()
        self.runtime.close()
        self.runtime.tick()
        self.assertTrue(self.runtime._semantic_worker.closed)
        self.owned_camera.close.assert_called_once()
        self.pool.close.assert_called_once()
        self.pool.cameras['A'].close.assert_not_called()
        self.provider.close.assert_not_called()
        self.provider.associate.assert_not_called()

    def test_shared_cameras_reuse_frames_when_mapping_is_due_between_semantic_updates(self):
        from tests.unit.test_shared_frames import camera
        self.config = replace(self.config, spatial_mapping=replace(self.config.spatial_mapping,
            enabled=True, source='rgbd', camera_prim_paths=self.config.multi_view.camera_prim_paths))
        cameras = {path: camera() for path in self.config.multi_view.camera_prim_paths}
        self.pool = self.module.SharedFramePool(cameras, clock=lambda: self.now.return_value)
        service = Mock(generation=0)
        runtime = self.make_runtime(spatial_mapping=service)
        runtime.tick()
        prepared = self.provider.acquire.call_args.kwargs['frames']
        self.assertIs(service.submit.call_args.args[0].frames[0].depth, prepared[0].depth)
        service.submit.reset_mock()
        # At 0.5 s mapping is due; semantic acquisition is next due at 0.6 s.
        for stamp in (100.2, 100.4):
            self.now.return_value = stamp
            runtime.tick()
        for source in cameras.values():
            source.get_frame_token.return_value = (2., .2)
        self.now.return_value = 100.5
        runtime.tick()
        service.submit.assert_not_called()
        for source in cameras.values():
            source.get_depth.assert_called_once()
        self.now.return_value = 100.61
        runtime.tick()
        frames = self.provider.acquire.call_args.kwargs['frames']
        self.assertEqual(service.submit.call_args.args[0].frames[0].token, frames[0].token)
        self.assertEqual(service.submit.call_args.args[0].frames[0].acquired_at, frames[0].acquired_at)

    def test_robot_source_failure_blocks_mapping_publication_until_recovery(self):
        self.config = replace(self.config, spatial_mapping=replace(self.config.spatial_mapping,
            enabled=True, source='rgbd', camera_prim_paths=self.config.multi_view.camera_prim_paths))
        service = Mock(generation=0)
        robot = NS(source_error='Robot model changed', prepare=Mock(return_value={}), close=Mock())
        runtime = self.make_runtime(spatial_mapping=service, robot_filter=robot)
        runtime.tick()
        runtime.tick()
        service.note_source_failure.assert_called_with(robot.source_error)
        service.poll.assert_not_called()
        self.assertEqual(self.provider.acquire.call_count, 1)  # Semantic source remains independent.
        robot.source_error = None
        runtime.tick()
        service.poll.assert_called_once()


if __name__ == '__main__':
    unittest.main()
