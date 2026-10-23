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
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame


class ControlledWorker:
    def __init__(self, process, **kwargs):
        self.process, self.pending, self.result = process, None, None
        self.invalidations, self.closed = 0, False

    def submit(self, payload, tag=None):
        self.pending = payload, tag
        return True

    def finish(self):
        payload, tag = self.pending
        self.result = WorkResult(tag, self.process(payload), None, 100., 100., 100.1)
        self.pending = None

    def poll(self):
        result, self.result = self.result, None
        return result

    def invalidate(self):
        self.invalidations += 1
        self.pending = self.result = None

    def close(self):
        self.closed = True
        self.invalidate()

    @property
    def health(self):
        return dict(closed=self.closed, pending=self.pending is not None)


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
        self.module.BoundedWorker = ControlledWorker
        self.config = load_scene_config('config/scenes/multiview_playground.toml')
        self.frames = tuple(DepthFrame(key, (1., .1), np.ones((2, 3)), np.eye(3),
            np.eye(4), 100.) for key in self.config.multi_view.camera_prim_paths)
        self.pool = NS(batch=Mock(return_value=self.frames), reset=Mock(),
            begin_update=Mock(), close=Mock(), cameras={'A': NS(reset_clock=Mock())},
            status={'semantic': {}}, acquisitions=0, shared_reads=0, failures=0)
        self.provider = NS(rig=NS(status={}), processor=NS(compute=lambda payload: payload),
            acquire=Mock(side_effect=lambda frames: NS(timestamp=min(f.acquired_at for f in frames))),
            associate=Mock(side_effect=lambda result, context: ObservationBatch([
                ObjectObservation('cube', pose=Pose([.4, 0, .025]))], result.timestamp)),
            forget=Mock(), close=Mock())
        self.model = WorldModel()
        self.updater = WorldModelUpdater(self.model, self.provider)
        self.world = NS(is_playing=Mock(return_value=True), current_time=1.)
        self.runtime = self.module.ObservationRuntime(self.world, self.config, self.updater,
            pool=self.pool, stage=self.stage, clock=lambda: self.now.return_value)
        self.addCleanup(self.runtime.close)

    def submit_then_complete(self):
        self.runtime.tick()
        self.runtime._semantic_worker.finish()
        self.pool.batch.return_value = None
        self.now.return_value = 100.1

    def test_acquisition_and_publication_are_separate_without_physics_registration(self):
        self.runtime.tick()
        self.assertFalse(self.model.objects())
        self.provider.associate.assert_not_called()
        self.runtime._semantic_worker.finish()
        self.pool.batch.return_value = None
        self.now.return_value = 100.1
        self.runtime.tick()
        self.assertEqual(self.model.require('cube').last_seen, 100.)
        self.assertEqual(self.runtime.diagnostics['semantic']['published'], 1)
        self.assertAlmostEqual(self.runtime.diagnostics['semantic']['last']['publish_age_seconds'], .1)

    def test_held_context_aba_rejects_before_tracking(self):
        self.model.register(WorldObject('other'))
        self.model.require('other').last_seen = 100.
        self.submit_then_complete()
        self.model.set_held('other')
        self.model.set_held(None)
        self.runtime.tick()
        self.provider.associate.assert_not_called()
        self.assertEqual(self.runtime.diagnostics['semantic']['rejected'], 1)
        self.assertFalse(self.model.exists('cube'))

    def test_acquisition_failure_barrier_precedes_polling_old_success(self):
        self.submit_then_complete()
        self.now.return_value = 100.3
        self.pool.batch.side_effect = RuntimeError('camera failed')
        self.runtime.tick()
        self.provider.associate.assert_not_called()
        self.assertIn('camera failed', self.updater.last_error)
        self.assertEqual(self.runtime.diagnostics['semantic']['source_failures'], 1)

    def test_expired_result_does_not_enter_tracker(self):
        self.submit_then_complete()
        self.now.return_value = 101.1
        self.runtime.tick()
        self.provider.associate.assert_not_called()
        self.assertFalse(self.model.objects())

    def test_stop_play_installs_generation_barrier_and_does_not_replay_old_work(self):
        self.submit_then_complete()
        self.world.is_playing.return_value = False
        self.runtime.tick()
        self.assertEqual(self.runtime.diagnostics['state'], 'stopped')
        self.world.is_playing.return_value = True
        self.runtime.tick()
        self.assertEqual(self.runtime.diagnostics['generation'], 2)
        self.provider.associate.assert_not_called()
        self.assertEqual(self.pool.reset.call_count, 2)
        self.pool.cameras['A'].reset_clock.assert_called_once()

    def test_stage_replacement_reports_failure_and_discards_completed_work(self):
        self.submit_then_complete()
        self.stage = object()
        self.runtime._on_update()
        self.assertEqual(self.runtime.diagnostics['state'], 'source_failure')
        self.assertIn('reinitialize', self.runtime.diagnostics['error'])
        self.provider.associate.assert_not_called()
        self.assertIsNone(self.runtime._semantic_worker.poll())

    def test_disable_retains_camera_resources_and_invalidates_work(self):
        self.submit_then_complete()
        self.runtime.set_enabled(semantic=False, mapping=False)
        self.runtime.tick()
        self.provider.associate.assert_not_called()
        self.pool.close.assert_not_called()

    def test_close_invalidates_workers_but_provider_remains_updater_owned(self):
        self.submit_then_complete()
        self.runtime.close()
        self.runtime.close()
        self.runtime.tick()
        self.provider.associate.assert_not_called()
        self.provider.close.assert_not_called()
        self.pool.close.assert_called_once()
        self.assertTrue(self.runtime._semantic_worker.closed)

    def test_independent_mapping_camera_validation(self):
        config = replace(self.config, world=replace(self.config.world, provider='ground_truth'),
            spatial_mapping=replace(self.config.spatial_mapping, enabled=True, source='rgbd'))
        with self.assertRaisesRegex(ValueError, 'camera_prim_paths'):
            self.module.mapping_settings(config)
        config = replace(config, spatial_mapping=replace(config.spatial_mapping,
            camera_prim_paths=('/World/IndependentCamera',)))
        self.assertEqual(self.module.mapping_settings(config).camera_prim_paths,
                         ('/World/IndependentCamera',))

    def test_known_robot_model_failure_blocks_older_map_completion_immediately(self):
        service = NS(note_source_failure=Mock(), poll=Mock(), close=Mock())
        robot_filter = NS(source_error='Robot model changed; reinitialize', close=Mock())
        self.runtime.spatial_mapping = service
        self.runtime.robot_filter = robot_filter
        self.runtime._mapping_enabled = True
        self.runtime._mapping_due = 101.  # The next scheduled acquisition is later.
        self.runtime._semantic_enabled = False
        self.runtime.tick()
        self.runtime.tick()
        service.note_source_failure.assert_called_once_with(robot_filter.source_error)
        service.poll.assert_not_called()
        robot_filter.source_error = None
        self.runtime.tick()
        service.poll.assert_called_once()


if __name__ == '__main__':
    unittest.main()
