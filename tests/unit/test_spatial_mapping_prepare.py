"""SDK preparation remains owner-only; its immutable payload is worker-safe."""
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import Mock,patch
import numpy as np
from tests.unit.test_spatial_mapping_robot_adapter import Scene,boundaries,frame,ADAPTER
from robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter import IsaacRobotSelfFilter
from robot_skill_stack.world.spatial_mapping.robot_filter import RobotSelfFilter
from tests.unit.test_spatial_mapping_robot_filter import pixels

class RobotPrepareTests(unittest.TestCase):
    def test_preparation_does_not_intersect_and_payload_outlives_adapter(self):
        scene=Scene()
        with boundaries(scene):
            source=IsaacRobotSelfFilter(scene,scene.root.path,stride=1)
            frames=(frame(),)
            with patch.object(RobotSelfFilter, 'masks', side_effect=AssertionError('intersection in prepare')):
                prepared=source.prepare(frames)
            with self.assertRaises(TypeError): prepared['extra']=prepared['A']
            self.assertEqual(prepared['A'].frame_token,frames[0].token)
            source.close()
            with patch(ADAPTER+'._usd_modules',side_effect=AssertionError('worker USD access')):
                with ThreadPoolExecutor(max_workers=1) as worker:
                    masks=worker.submit(RobotSelfFilter().masks,frames,prepared,
                        pixels_by_camera=pixels(frames)).result(timeout=1.)
            self.assertTrue(masks['A'].all())

    def test_old_frame_preparation_matches_history_and_owns_transforms(self):
        scene=Scene()
        with boundaries(scene) as (state,_,_):
            source=IsaacRobotSelfFilter(scene,scene.root.path,stride=1)
            state.stamp,state.raw[0,2]=1.1,1.2
            state.callback(.1)
            old=source.prepare((frame(1.,1.),))
            current=source.prepare((frame(1.1,1.2),))
            state.raw[0,2]=4.
            self.assertTrue(RobotSelfFilter().masks((frame(1.,1.),),old,
                pixels_by_camera=pixels((frame(1.,1.),)))['A'].all())
            self.assertTrue(RobotSelfFilter().masks((frame(1.1,1.2),),current,
                pixels_by_camera=pixels((frame(1.1,1.2),)))['A'].all())
            with self.assertRaises(ValueError): old['A'].parts[0].world_from_local.setflags(write=True)
            self.assertIs(old['A'].parts[0].mesh,current['A'].parts[0].mesh)
            source.close()

    def test_wrong_exposure_missing_history_and_closed_source_fail_without_latest_fallback(self):
        scene=Scene()
        with boundaries(scene):
            source=IsaacRobotSelfFilter(scene,scene.root.path,stride=1)
            with self.assertRaisesRegex(RuntimeError,'No robot pose'):
                source.prepare((frame(.5),))
            source.close()
            with self.assertRaisesRegex(RuntimeError,'closed'):
                source.prepare((frame(),))

    def test_prepare_rejects_worker_thread_before_any_sdk_state_read(self):
        scene=Scene()
        with boundaries(scene) as (_,_,manager):
            source=IsaacRobotSelfFilter(scene,scene.root.path,stride=1)
            owner=threading.get_ident(); original=manager.get_simulation_time
            def read_time():
                self.assertEqual(threading.get_ident(),owner,'SDK access outside owner thread')
                return original()
            manager.get_simulation_time=Mock(side_effect=read_time)
            with ThreadPoolExecutor(max_workers=1) as worker:
                future=worker.submit(source.prepare,(frame(),))
                with self.assertRaisesRegex(RuntimeError,'thread|owner'):
                    future.result(timeout=1.)
            manager.get_simulation_time.assert_not_called()
            source.close()

    def test_arbitrary_camera_count_does_not_read_live_physx_per_camera(self):
        scene=Scene()
        with boundaries(scene) as (_,view,_):
            source=IsaacRobotSelfFilter(scene,scene.root.path,stride=1)
            view.get_transforms.reset_mock()
            frames=tuple(frame(camera_id=str(i)) for i in range(5))
            prepared=source.prepare(frames)
            self.assertEqual(set(prepared),set(str(i) for i in range(5)))
            view.get_transforms.assert_not_called()  # Only the post-step history captures live poses.
            source.close()

    def test_reset_history_hides_pre_stop_exposures_until_new_physics(self):
        scene = Scene()
        with boundaries(scene) as (state, _, _):
            source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
            source.reset_history()
            with self.assertRaisesRegex(RuntimeError, 'empty'):
                source.prepare((frame(),))
            state.stamp = 1.1
            state.callback(.1)
            with self.assertRaisesRegex(RuntimeError, 'No robot pose'):
                source.prepare((frame(),))
            self.assertEqual(source.prepare((frame(1.1),))['A'].simulation_time, 1.1)
            source.close()




class RobotViewRebindTests(unittest.TestCase):
    def test_new_manager_view_rebinds_exact_bodies_without_geometry_or_latest_pose(self):
        from types import SimpleNamespace as NS
        scene = Scene()
        with boundaries(scene) as (state, old_view, manager):
            source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
            mesh = source._parts[0].mesh
            new_view = NS(prim_paths=[scene.body.path], count=1,
                          get_transforms=Mock(return_value=state.raw))
            new_sim = NS(is_valid=True, create_rigid_body_view=Mock(return_value=new_view))
            manager.get_physics_sim_view.return_value = new_sim
            with patch.object(source, '_geometry', side_effect=AssertionError('geometry rebuild')):
                self.assertTrue(source.rebind_after_play())
            self.assertIs(source._parts[0].mesh, mesh)
            self.assertIs(source._view, new_view)
            new_view.get_transforms.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'empty'):
                source.prepare((frame(),))
            state.stamp = 1.1
            state.callback(.1)
            self.assertEqual(source.prepare((frame(1.1),))['A'].simulation_time, 1.1)
            source.close()

    def test_missing_manager_view_withholds_old_history_and_same_live_pause_view_is_noop(self):
        scene = Scene()
        with boundaries(scene) as (_, old_view, manager):
            source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
            current = manager.get_physics_sim_view.return_value
            self.assertFalse(source.rebind_after_play())
            self.assertIs(source._view, old_view)
            manager.get_physics_sim_view.return_value = None
            with self.assertRaisesRegex(RuntimeError, 'Waiting for a new initialized'):
                source.rebind_after_play()
            with self.assertRaisesRegex(RuntimeError, 'Waiting for a new initialized'):
                source.prepare((frame(),))
            self.assertFalse(source._history)
            manager.get_physics_sim_view.return_value = current
            source.close()

    def test_mismatched_new_body_paths_reject_and_invalidate_model(self):
        from types import SimpleNamespace as NS
        scene = Scene()
        with boundaries(scene) as (_, old_view, manager):
            source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
            wrong = NS(prim_paths=['/Other/Body'], count=1)
            manager.get_physics_sim_view.return_value = NS(create_rigid_body_view=lambda paths: wrong)
            with self.assertRaisesRegex(RuntimeError, 'paths changed'):
                source.rebind_after_play()
            self.assertIs(source._view, old_view)
            with self.assertRaisesRegex(RuntimeError, 'reinitialize'):
                source.prepare((frame(),))
            source.close()

    def test_model_notice_invalidates_geometry_visibility_and_resync_but_allows_body_motion(self):
        from types import SimpleNamespace as NS
        changes = [('resync', '/Robots/Panda/Arm'),
                   ('resync', '/Robots/Panda/Arm/Visual.faceVertexIndices'),
                   ('info', '/Robots/Panda/Arm/Visual.points'),
                   ('info', '/Robots/Panda/Arm/Visual.faceVertexIndices'),
                   ('info', '/Robots/Panda/Arm/Visual.visibility'),
                   ('info', '/Robots.purpose'), ('info', '/Robots/Panda/Arm/Visual.xformOp:translate'),
                   ('info', '/Robots/Panda/Arm.xformOp:scale')]
        for category, path in changes:
            with self.subTest(path=path):
                scene = Scene()
                with boundaries(scene):
                    source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
                    allowed = NS(GetResyncedPaths=lambda: [scene.body.path+'.xformOp:translate'], GetChangedInfoOnlyPaths=lambda:
                                 [scene.body.path+'.xformOp:translate', '/Unrelated.points'])
                    source._model_changed(allowed, scene)
                    self.assertIsNone(source._model_error)
                    notice = NS(GetResyncedPaths=lambda: [path] if category == 'resync' else [],
                                GetChangedInfoOnlyPaths=lambda: [path] if category == 'info' else [])
                    source._model_changed(notice, scene)
                    with self.assertRaisesRegex(RuntimeError, 'reinitialize'):
                        source.rebind_after_play()
                    with self.assertRaisesRegex(RuntimeError, 'reinitialize'):
                        source.prepare((frame(),))
                    self.assertFalse(source._history)
                    listener = source._model_notice
                    source.close()
                    listener.Revoke.assert_called_once()


class RobotSourceErrorTests(unittest.TestCase):
    def test_cached_pose_model_and_closed_errors_are_readonly_without_sdk_reads(self):
        from types import SimpleNamespace as NS
        scene = Scene()
        with boundaries(scene) as (state, view, manager):
            source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
            self.assertIsNone(source.source_error)
            state.error = RuntimeError('invalidated backend view')
            state.callback(.1)
            with patch.object(manager, 'get_simulation_time', side_effect=AssertionError('SDK read')), \
                    patch.object(manager, 'get_physics_sim_view', side_effect=AssertionError('SDK read')), \
                    patch.object(view, 'get_transforms', side_effect=AssertionError('SDK read')):
                self.assertEqual(source.source_error, 'invalidated backend view')
                notice = NS(GetResyncedPaths=lambda: [], GetChangedInfoOnlyPaths=lambda:
                            [scene.mesh.path+'.points'])
                source._model_changed(notice, scene)
                self.assertIn('authored model changed', source.source_error)
                with self.assertRaises(AttributeError):
                    source.source_error = None
                source.close()
                self.assertEqual(source.source_error, 'Robot self-filter is closed')
