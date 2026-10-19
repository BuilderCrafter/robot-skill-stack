"""Execute production bootstrap with recording SDK boundaries; no simulator emulation."""
import time
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.provider import ObservationBatch
from robot_skill_stack.world.model.observations import ObjectObservation
from tests.unit.test_runtime_lifecycle import StopPlayWorld

ROOT=Path(__file__).resolve().parents[2]
ISAAC=ROOT/'robot_skill_stack/integrations/isaac'


@contextmanager
def bootstrap_fixture():
    env=NS(world=None, providers=[], fail_backend=False, init_calls=0, backend_calls=[])
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
        env.backend_calls.append(kw)
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
        factory_name='robot_skill_stack.integrations.isaac.manipulation.factory'
        factory_spec=importlib.util.spec_from_file_location(factory_name,ISAAC/'manipulation/factory.py')
        factory=importlib.util.module_from_spec(factory_spec)
        sys.modules[factory_name]=factory
        factory_spec.loader.exec_module(factory)
        source_name='robot_skill_stack.integrations.isaac.world.factory'
        source_spec=importlib.util.spec_from_file_location(source_name,ISAAC/'world/factory.py')
        source_factory=importlib.util.module_from_spec(source_spec)
        sys.modules[source_name]=source_factory
        source_spec.loader.exec_module(source_factory)
        spec.loader.exec_module(m)
        cfg=load_scene_config(ROOT/'config/scenes/playground.toml')
        cfg=replace(cfg,world=replace(cfg.world,provider='ground_truth'))
        async def make_provider(config):
            camera=NS(close=Mock())
            p=NS(observe=Mock(side_effect=lambda context: ObservationBatch([ObjectObservation('cube',pose=Pose([.45,0,.025]),
                                  class_name='cube',size=[.05]*3,visible=True,source='ground_truth')], time.monotonic())),
                 forget=Mock(), close=Mock(side_effect=camera.close), camera=camera)
            env.providers.append(p); return p
        with patch.object(m,'load_scene_config',return_value=cfg), patch.object(m,'create_source',side_effect=make_provider):
            yield m,env


class BootstrapTests(unittest.TestCase):
    def run_case(self,fn):
        async def run():
            with bootstrap_fixture() as (m,e): await fn(m,e)
        asyncio.run(run())

    def test_explicit_franka_selection_preserves_backend_and_gripper_values(self):
        async def fn(m,e):
            cfg=m.load_scene_config('x.toml')
            factory=sys.modules['robot_skill_stack.integrations.isaac.manipulation.factory']
            robot=object()
            world=NS(scene=NS(get_object=Mock(return_value=None),add=Mock(return_value=robot)))
            self.assertIs(m.create_robot(world,cfg.robot),robot)
            factory.Franka.assert_called_once_with(prim_path='/Franka',name='franka')
            backend,gripper=m.create_backend(world,robot,cfg.robot)
            self.assertEqual(e.backend_calls[-1],dict(world=world,robot=robot,
                             position_tolerance=.01,orientation_tolerance=.05,max_motion_steps=1000))
            self.assertEqual((gripper.min_width,gripper.max_width,gripper.aperture),(.005,.075,.08))
            self.assertEqual((gripper.tcp_to_palm,gripper.fingertip_offset,gripper.pad_above_tcp,
                              gripper.pad_below_tcp,gripper.finger_depth,gripper.finger_thickness,
                              gripper.palm_depth,gripper.palm_width,gripper.palm_height),
                             (.03,.012,.01,.01,.024,.025,.07,.17,.075))
        self.run_case(fn)

    def test_unknown_robot_backend_is_rejected_by_construction(self):
        async def fn(m,e):
            cfg=m.load_scene_config('x.toml')
            unknown=replace(cfg.robot,backend='unsupported')
            for operation,args in ((m.create_robot,(None,unknown)),
                                   (m.create_backend,(None,None,unknown))):
                with self.assertRaisesRegex(ValueError,'Unsupported robot backend'):
                    operation(*args)
            self.assertEqual(e.backend_calls,[])
        self.run_case(fn)

    def test_source_factories_select_only_the_requested_adapter(self):
        async def fn(m,e):
            cfg=m.load_scene_config('x.toml')
            factory=sys.modules['robot_skill_stack.integrations.isaac.world.factory']
            truth=await factory.create_source(cfg)
            factory.IsaacGroundTruthProvider.assert_called_once_with(
                cfg.ground_truth.objects_root,cfg.ground_truth.objects)
            factory.IsaacRgbdCamera.assert_not_called()
            cfg=replace(cfg,world=replace(cfg.world,provider='perception'))
            with patch.object(factory,'build_perception_provider',return_value=object()) as build:
                source=await factory.create_source(cfg)
            camera=factory.IsaacRgbdCamera.return_value
            factory.IsaacRgbdCamera.assert_called_once_with('/World/PerceptionCamera',resolution=(640,480))
            camera.initialize.assert_called_once_with(semantic_segmentation=False)
            build.assert_called_once_with(camera,cfg)
            camera.close.assert_not_called()
            with patch.object(factory,'build_perception_provider',side_effect=ValueError('discovery init')):
                with self.assertRaisesRegex(ValueError,'discovery init'):
                    await factory.create_source(cfg)
            camera.close.assert_called_once()
        self.run_case(fn)

    def test_rgbd_startup_needs_no_ground_truth_prims_or_objects(self):
        async def fn(m,e):
            cfg=m.load_scene_config('x.toml')
            cfg=replace(cfg,world=replace(cfg.world,provider='perception'),
                        ground_truth=replace(cfg.ground_truth,objects_root=None,objects={}))
            valid={'/Franka','/World/PerceptionCamera'}
            stage=NS(GetPrimAtPath=Mock(side_effect=lambda p:NS(IsValid=lambda:p in valid)))
            with patch.object(m,'load_scene_config',return_value=cfg), \
                 patch.object(m.omni.usd,'get_context',return_value=NS(get_stage=lambda:stage)):
                b=await m.build_runtime('rgbd.toml')
            self.assertEqual({c.args[0] for c in stage.GetPrimAtPath.call_args_list},valid)
            self.assertFalse(hasattr(b,'objects'))
            b.close()
        self.run_case(fn)

    def test_ground_truth_startup_does_not_require_camera_prim(self):
        async def fn(m,e):
            valid={'/Franka','/World/Objects','/World/Objects/Cube'}
            stage=NS(GetPrimAtPath=Mock(side_effect=lambda p:NS(IsValid=lambda:p in valid)))
            with patch.object(m.omni.usd,'get_context',return_value=NS(get_stage=lambda:stage)):
                b=await m.build_runtime('truth.toml')
            self.assertEqual({c.args[0] for c in stage.GetPrimAtPath.call_args_list},valid)
            b.close()
        self.run_case(fn)

    def test_required_prims_are_validated_before_runtime_construction(self):
        async def fn(m,e):
            cfg=m.load_scene_config('x.toml')
            for provider,paths in (('perception',('/Franka','/World/PerceptionCamera')),
                                   ('ground_truth',('/Franka','/World/Objects','/World/Objects/Cube'))):
                selected=replace(cfg,world=replace(cfg.world,provider=provider))
                for missing in paths:
                    stage=NS(GetPrimAtPath=lambda p:NS(IsValid=lambda:p!=missing))
                    with patch.object(m,'load_scene_config',return_value=selected), \
                         patch.object(m.omni.usd,'get_context',return_value=NS(get_stage=lambda:stage)):
                        with self.assertRaisesRegex(RuntimeError,missing):
                            await m.build_runtime('missing.toml')
            self.assertIsNone(e.world)
            self.assertFalse(e.providers)
        self.run_case(fn)

    def test_unknown_source_and_missing_selected_source_reference_fail_clearly(self):
        async def fn(m,e):
            cfg=m.load_scene_config('x.toml')
            factory=sys.modules['robot_skill_stack.integrations.isaac.world.factory']
            unknown=replace(cfg,world=replace(cfg.world,provider='unsupported'))
            with self.assertRaisesRegex(ValueError,'Unsupported observation source'):
                await factory.create_source(unknown)
            rgbd=replace(cfg,world=replace(cfg.world,provider='perception'),
                         perception=replace(cfg.perception,camera_prim_path=None))
            with self.assertRaisesRegex(ValueError,'camera_prim_path'):
                factory.required_source_prims(rgbd)
            factory.IsaacRgbdCamera.assert_not_called()
            factory.IsaacGroundTruthProvider.assert_not_called()
        self.run_case(fn)

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

    def test_invalid_updater_configuration_closes_selected_source(self):
        async def fn(m,e):
            with patch.object(m,'WorldModelUpdater',side_effect=ValueError('injected updater failure')):
                with self.assertRaisesRegex(ValueError,'updater failure'):
                    await m.build_runtime('x.toml')
            e.providers[0].close.assert_called_once()
            self.assertEqual(e.world._physics_functions,{})
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


if __name__=='__main__': unittest.main(verbosity=2)
