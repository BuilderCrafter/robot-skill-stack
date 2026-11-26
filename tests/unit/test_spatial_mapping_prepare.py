"""The robot adapter hands SDK-free payloads to workers and recovers after Play."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from tests.unit.test_spatial_mapping_robot_adapter import Scene, boundaries, frame
from tests.unit.test_spatial_mapping_robot_filter import pixels
from robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter import IsaacRobotSelfFilter
from robot_skill_stack.world.spatial_mapping.robot_filter import RobotSelfFilter


class RobotPrepareTests(unittest.TestCase):
    def test_prepared_payload_outlives_adapter_and_can_be_filtered_in_worker(self):
        scene = Scene()
        with boundaries(scene):
            source = IsaacRobotSelfFilter(scene, scene.root.path)
            frames = (frame(),)
            with patch.object(RobotSelfFilter, 'masks', side_effect=AssertionError('intersection in prepare')):
                prepared = source.prepare(frames)
            source.close()
            with patch.dict('sys.modules', {'pxr': None, 'omni': None, 'isaacsim': None,
                                           'isaacsim.core.simulation_manager': None}):
                with ThreadPoolExecutor(max_workers=1) as worker:
                    masks = worker.submit(RobotSelfFilter().masks, frames, prepared,
                        pixels_by_camera=pixels(frames)).result(timeout=1.)
            self.assertTrue(masks['A'].all())

    def test_prepare_rejects_worker_thread_before_any_sdk_read(self):
        scene = Scene()
        with boundaries(scene) as (_, _, manager):
            source = IsaacRobotSelfFilter(scene, scene.root.path)
            with patch.object(manager, 'get_simulation_time', side_effect=AssertionError('SDK read')) as read:
                with ThreadPoolExecutor(max_workers=1) as worker:
                    with self.assertRaises(RuntimeError):
                        worker.submit(source.prepare, (frame(),)).result(timeout=1.)
                read.assert_not_called()
            source.close()

    def test_stop_play_discards_old_poses_and_rebinds_without_rebuilding_meshes(self):
        scene = Scene()
        with boundaries(scene) as (state, _, manager):
            source = IsaacRobotSelfFilter(scene, scene.root.path)
            mesh = source.prepare((frame(),))['A'].parts[0].mesh
            source.reset_history()
            self.assertFalse(source.rebind_after_play())  # Pause may retain its view.
            with self.assertRaises(RuntimeError):
                source.prepare((frame(),))
            manager.get_physics_sim_view.return_value = None
            with self.assertRaises(RuntimeError):
                source.rebind_after_play()
            self.assertIsNotNone(source.source_error)
            new_view = NS(prim_paths=[scene.body.path], get_transforms=Mock(return_value=state.raw))
            manager.get_physics_sim_view.return_value = NS(create_rigid_body_view=lambda paths: new_view)
            self.assertTrue(source.rebind_after_play())
            new_view.get_transforms.assert_not_called()
            with self.assertRaises(RuntimeError):
                source.prepare((frame(),))
            state.stamp, state.raw[0, 2] = 1.1, 1.2
            state.callback(.1)
            prepared = source.prepare((frame(1.1, 1.2),))['A']
            self.assertEqual(prepared.simulation_time, 1.1)
            self.assertIs(prepared.parts[0].mesh, mesh)
            self.assertEqual(prepared.parts[0].world_from_local[2, 3], 1.2)
            with self.assertRaises(RuntimeError):
                source.prepare((frame(),))
            source.close()
