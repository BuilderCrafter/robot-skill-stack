"""Execute production bootstrap with recording SDK boundaries; no simulator emulation."""
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch
import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.observations import ObjectObservation
from tests.unit.test_runtime_lifecycle import StopPlayWorld
from tests.unit.test_robot_filter_adapter import fake_runtime
from tests.unit import test_perception_v2 as v2_tests
from tests.support.fake_omni_ui import extension_environment
from tests.unit.test_control_panel_ui import bundle_

ROOT=Path(__file__).resolve().parents[2]
ISAAC=ROOT/'robot_skill_stack/integrations/isaac'


@contextmanager
def bootstrap_fixture():
    env=NS(world=None, providers=[], fail_backend=False, init_calls=0)
    async def next_update(): await asyncio.sleep(0)
    app=ModuleType('omni.kit.app'); app.get_app=lambda: NS(next_update_async=next_update)
    kit,omni,usd=ModuleType('omni.kit'),ModuleType('omni'),ModuleType('omni.usd')
    kit.app,omni.kit,omni.usd=app,kit,usd
    usd.get_context=lambda: NS(get_stage=lambda: NS(GetPrimAtPath=lambda p:NS(IsValid=lambda:True)))
    class World(StopPlayWorld):
        def __init__(self, **kw):
            super().__init__(); env.world=self; self.context=None; self.playing=False
            self.scene=NS(get_object=lambda name:NS(name=name),add=lambda obj:obj)
        @classmethod
        def instance(cls): return env.world
        def get_physics_context(self): return self.context
        def is_playing(self): return self.playing
        async def initialize_simulation_context_async(self):
            env.init_calls+=1; await next_update(); self.context=object()
        async def reset_async(self):
            self.stop(); await next_update(); self.play(); self.playing=True
        async def play_async(self):
            self.play(); self.playing=True; await next_update()
    def backend(**kw):
        if env.fail_backend: raise ValueError('injected backend failure')
        return NS(grasp_min_width=.005,grasp_max_width=.075,
                  get_end_effector_pose=lambda:Pose([0,0,.2],[0,0,1,0]),check_reachability=lambda p:True)
    def module(name, **kw):
        m=ModuleType(name); m.__dict__.update(kw); return m
    modules={'omni':omni,'omni.kit':kit,'omni.kit.app':app,'omni.usd':usd,
             'isaacsim':module('isaacsim'),'isaacsim.core':module('isaacsim.core'),
             'isaacsim.core.api':module('isaacsim.core.api',World=World),
             'isaacsim.core.prims':module('isaacsim.core.prims',SingleXFormPrim=Mock()),
             'isaacsim.robot.manipulators.examples.franka':module('franka',Franka=Mock())}
    for path,symbol,value in (
        ('manipulation.franka_backend','IsaacFrankaBackend',backend),
        ('world.ground_truth_provider','IsaacGroundTruthProvider',Mock()),
        ('sensors.rgbd_camera','IsaacRgbdCamera',Mock())):
        name='robot_skill_stack.integrations.isaac.'+path
        modules[name]=module(name,**{symbol:value})
    name='bootstrap_under_test'
    spec=importlib.util.spec_from_file_location(name,ISAAC/'bootstrap.py')
    m=importlib.util.module_from_spec(spec); modules[name]=m
    with patch.dict(sys.modules,modules):
        spec.loader.exec_module(m)
        cfg=load_scene_config(ROOT/'config/scenes/playground.toml')
        cfg=replace(cfg,world=replace(cfg.world,provider='ground_truth'))
        async def make_provider(world,config):
            p=NS(observe=Mock(return_value=[ObjectObservation('cube',pose=Pose([.45,0,.025]),
                                  class_name='cube',size=[.05]*3,visible=True,source='ground_truth')]),
                 close=Mock(),camera=NS(close=Mock()))
            env.providers.append(p); return p
        with patch.object(m,'load_scene_config',return_value=cfg), patch.object(m,'_create_state_provider',side_effect=make_provider):
            yield m,env


class BootstrapTests(unittest.TestCase):
    def run_case(self,fn):
        async def run():
            with bootstrap_fixture() as (m,e): await fn(m,e)
        asyncio.run(run())

    def test_ground_truth_populates_and_ticks(self):
        async def fn(m,e):
            b=await m.build_runtime('profile.toml'); self.assertIsNotNone(b.world_model.get('cube'))
            count=b.world_updater.update_count; e.world.tick(); self.assertEqual(b.world_updater.update_count,count+1)
            b.close(); self.assertEqual(e.world._physics_functions,{})
        self.run_case(fn)

    def test_build_failure_closes_camera_and_provider_no_callback(self):
        async def fn(m,e):
            e.fail_backend=True
            with self.assertRaisesRegex(ValueError,'injected'): await m.build_runtime('x.toml')
            self.assertEqual(e.world._physics_functions,{})
            e.providers[0].close.assert_called_once(); e.providers[0].camera.close.assert_called_once()
        self.run_case(fn)

    def test_repeat_build_after_stop_replaces_only_owned_callback(self):
        async def fn(m,e):
            a=await m.build_runtime('a.toml'); e.world.stop()
            b=await m.build_runtime('b.toml'); e.world.tick()
            self.assertEqual(e.world.errors,[]); self.assertIs(e.world._robot_skill_stack_bundle,b)
            self.assertEqual(len(e.world._physics_functions),1); self.assertTrue(a.world_updater._closed)
            e.providers[0].close.assert_called_once(); b.close()
        self.run_case(fn)

    def test_repeated_close_is_idempotent(self):
        async def fn(m,e):
            b=await m.build_runtime('a.toml'); b.close(); b.close()
            e.providers[0].camera.close.assert_called_once(); self.assertIsNone(e.world._robot_skill_stack_bundle)
        self.run_case(fn)

    def test_close_clears_bundle_owner_despite_camera_error(self):
        async def fn(m,e):
            b=await m.build_runtime('a.toml'); b.state_provider.camera.close.side_effect=ValueError('camera')
            with self.assertRaises(ValueError): b.close()
            self.assertEqual(e.world._physics_functions,{}); self.assertIsNone(e.world._robot_skill_stack_bundle)
        self.run_case(fn)

    def test_concurrent_build_rejected(self):
        async def fn(m,e):
            task=asyncio.create_task(m.build_runtime('a.toml')); await asyncio.sleep(0)
            with self.assertRaisesRegex(RuntimeError,'in progress'): await m.build_runtime('b.toml')
            b=await task; b.close()
        self.run_case(fn)

    def test_retry_after_partial_context_initialization(self):
        async def fn(m,e):
            e.world=m.World(); self.assertIsNone(e.world.context)
            b=await m.build_runtime('a.toml'); self.assertEqual(e.init_calls,1); b.close()
        self.run_case(fn)

    def test_cancellation_releases_build_guard(self):
        async def fn(m,e):
            task=asyncio.create_task(m.build_runtime('a.toml')); await asyncio.sleep(0); task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            self.assertFalse(m._BUILDING); b=await m.build_runtime('b.toml'); b.close()
        self.run_case(fn)


class CameraCleanupTests(unittest.TestCase):
    def adapter(self):
        name='camera_cleanup_test'; spec=importlib.util.spec_from_file_location(name,ISAAC/'sensors/rgbd_camera.py')
        m=importlib.util.module_from_spec(spec)
        omni=ModuleType('omni'); omni.usd=ModuleType('omni.usd')
        stub=ModuleType('isaacsim.sensors.camera'); stub.Camera=Mock()
        with patch.dict(sys.modules,{'omni':omni,'omni.usd':omni.usd,'isaacsim.sensors.camera':stub}):
            spec.loader.exec_module(m)
        obj=m.IsaacRgbdCamera.__new__(m.IsaacRgbdCamera); obj._closed=False; obj.camera=Mock()
        obj.camera.get_render_product_path.return_value='/Render/camera'
        return obj

    def test_detaches_reference_and_events_before_render_product(self):
        obj=self.adapter(); ref=obj.camera._fabric_time_annotator; obj.close()
        obj.camera.pause.assert_called_once(); ref.detach.assert_called_once_with(['/Render/camera'])
        self.assertIsNone(obj.camera._stage_open_callback); self.assertIsNone(obj.camera._timer_reset_callback)
        obj.camera.destroy.assert_called_once()

    def test_partial_camera_without_reference_and_idempotent_close(self):
        obj=self.adapter(); obj.camera._fabric_time_annotator=None; obj.close(); obj.close()
        obj.camera.destroy.assert_called_once()

    def test_render_product_destroy_attempted_after_detach_failure(self):
        obj=self.adapter(); obj.camera._fabric_time_annotator.detach.side_effect=ValueError('detaching')
        with self.assertRaisesRegex(RuntimeError,'detaching'): obj.close()
        obj.camera.destroy.assert_called_once()


class SnapshotDiagnosticsTests(unittest.TestCase):
    def test_capture_call_can_follow_camera_callback_without_latest_event(self):
        with fake_runtime() as (m,e):
            s=m.IsaacRobotSnapshotSource(e.world,e.camera,'/Robot',(),.002)
            self.assertEqual(s.snapshot('A').token,'A'); self.assertEqual(s.diagnostics_snapshot()['matched_frames'],1)
            s.close()

    def test_timing_diagnostics_show_unmatched_frame(self):
        with fake_runtime() as (m,e):
            s=m.IsaacRobotSnapshotSource(e.world,e.camera,'/Robot',(),.002)
            e.image_time=.2
            with self.assertRaisesRegex(RuntimeError,'No same-frame'): s.snapshot('A')
            d=s.diagnostics_snapshot(); self.assertEqual(d['rendering_time'],.2); self.assertEqual(d['nearest_pose_time'],1.)
            self.assertEqual(d['matched_frames'],0); s.close()

    def test_no_use_of_new_pose_when_old_frame_time_does_not_match(self):
        with fake_runtime() as (m,e):
            s=m.IsaacRobotSnapshotSource(e.world,e.camera,'/Robot',(),.002); s.snapshot('A')
            e.world.current_time=2.; e.image_time=1.5; e.token='B'
            with self.assertRaises(RuntimeError): s.snapshot('B')
            self.assertEqual(s.latest.token,'A'); s.close()


class V2DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        # Reuse fixture construction, not the inherited existing test methods.
        self.fx=v2_tests.V2ProviderTests(); self.fx.setUp(); self.addCleanup(self.fx.doCleanups)

    def test_stale_capture_is_saved_but_never_published(self):
        f=self.fx; f.p.observe(); f.now+=3.; f.executor.finish(); self.assertFalse(f.p.observe())
        with tempfile.TemporaryDirectory() as d:
            path=f.p.save_capture(Path(d)/'stale.npz')
            with np.load(path,allow_pickle=False) as z: data=json.loads(str(z['settings']))
            self.assertFalse(data['observation_accepted']); self.assertEqual(data['provider_status']['stale_results'],1)
        self.assertFalse(f.p.tracker.tracks()); self.assertIn('Robot filter',f.p.last_error)

    def test_empty_initialization_has_json_serializable_diagnostics(self):
        d=self.fx.p.diagnostics_snapshot(); json.dumps(d,allow_nan=False)
        self.assertEqual(d['accepted_frames'],0); self.assertIsNone(d['candidate_count'])

    def test_in_flight_age_and_submissions(self):
        f=self.fx; f.p.observe(); f.now+=.6; d=f.p.diagnostics_snapshot()
        self.assertTrue(d['in_flight']); self.assertAlmostEqual(d['request_age_s'],.6)
        self.assertEqual(d['submitted_frames'],1)

    def test_accepted_capture_flag_and_counts(self):
        f=self.fx; f.confirm()
        with tempfile.TemporaryDirectory() as d:
            path=f.p.save_capture(Path(d)/'fresh.npz')
            with np.load(path,allow_pickle=False) as z: meta=json.loads(str(z['settings']))
            self.assertTrue(meta['observation_accepted']); self.assertEqual(meta['provider_status']['accepted_frames'],3)
            self.assertEqual(meta['provider_status']['confirmed_tracks'],1)

    def test_capture_failure_contains_actual_reason(self):
        self.fx.p.last_error='No matching robot pose'
        with self.assertRaisesRegex(ValueError,'No matching robot pose'):
            self.fx.p.save_capture('does_not_exist.npz')

    def test_ui_capture_can_save_status_when_no_image_completed(self):
        with extension_environment() as (_,c,module),tempfile.TemporaryDirectory() as d:
            c.bundle=bundle_(); c.bundle.state_provider=self.fx.p; c.bundle.world_updater.last_error=None
            with patch.object(module,'ROOT',Path(d)):
                c._capture_perception_clicked()
            files=list(Path(d).glob('outputs/perception/status_*.json'))
            self.assertEqual(len(files),1); meta=json.loads(files[0].read_text())
            self.assertEqual(meta['accepted_frames'],0)


if __name__=='__main__': unittest.main(verbosity=2)
