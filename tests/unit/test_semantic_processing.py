"""Semantic acquisition/compute/association boundary and publication contracts.

Planned public seam: provider.acquire(), provider.processor.compute(input),
provider.associate(result, context), updater.accepts(timestamp, context_revision)
and updater.publish(batch, context_revision=...). The runtime checks accepts()
BEFORE association. Computation owns no tracker or simulator handle; tracking,
diagnostic publication, forgetting and WorldModel mutation remain on the owner.

These tests deliberately retain the different V1/MV held association policies.
Frozen V4 provider code provides a numerical oracle for the extraction boundary;
it imports the unchanged geometry algorithms, not benchmark truth at runtime.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.provider import ObservationBatch
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.perception.diagnostics import save_capture
from robot_skill_stack.world.perception.factory import build_perception_provider
from robot_skill_stack.world.perception_multiview.acquisition import FrameSynchronizer
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from tests.support.multiview_research import analytic_frames
from tests.support.primitive_scene import FrameCamera, Solid, render
from tests.unit.test_multiview_provider import Camera


ROOT=Path(__file__).resolve().parents[2]
BASELINE='74169203d2a937c2195089c6f4f2efdd8afd508d'


def frozen_provider(kind):
    package='perception' if kind=='v1' else 'perception_multiview'
    path=f'robot_skill_stack/world/{package}/provider.py'
    source=subprocess.check_output(['git','show',f'{BASELINE}:{path}'],cwd=ROOT)
    module=types.ModuleType(f'frozen_v4_semantic_{kind}')
    sys.modules[module.__name__]=module
    exec(compile(source,f'{BASELINE}:{path}','exec'),module.__dict__)
    return module.PerceptionStateProvider if kind=='v1' else module.MultiViewProvider


def canonical(value):
    """Compare all observation geometry/metadata, including exact numeric arrays."""
    if isinstance(value,np.ndarray):return canonical(value.tolist())
    if isinstance(value,np.generic):return canonical(value.item())
    if isinstance(value,Enum):return value.value
    if is_dataclass(value):return {field.name:canonical(getattr(value,field.name)) for field in fields(value)}
    if isinstance(value,dict):return {key:canonical(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [canonical(item) for item in value]
    if isinstance(value,float) and np.isnan(value):return 'NaN'
    return value


class SemanticProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sphere=render([Solid('sphere')])
        cls.empty=render([])
        solids=[Solid('cube',center=(.32,-.18,.025),yaw=.37),Solid('sphere'),
            Solid('cylinder',center=(.58,.18,.05),radius=.02),
            Solid('ellipsoid',center=(.62,-.16,.025),size=(.08,.05,.05))]
        cls.multiple=render(solids)
        cls.multiview=analytic_frames(solids)
        cls.old={kind:frozen_provider(kind) for kind in ('v1','mv')}

    def setUp(self):
        self.owner=threading.get_ident()
        self.clock=patch('time.monotonic',return_value=100.)
        self.now=self.clock.start()
        self.addCleanup(self.clock.stop)

    def provider(self,kind='v1',*,frozen=False,multiple=False):
        if kind=='v1':
            camera=FrameCamera(self.multiple if multiple else self.sphere)
            value=build_perception_provider(camera,load_scene_config('config/scenes/playground.toml'))
            if frozen:
                value=self.old[kind](camera,value.localizer,value.discoverer,value.tracker,
                    primitive_estimator=value.primitive_estimator,source=value.source)
            advance=lambda:camera.push(camera.frame)
            cameras=[camera]
        else:
            frames=self.multiview if multiple else analytic_frames([Solid('sphere')])
            cameras=[Camera(frame) for frame in frames]
            rig=FrameSynchronizer(dict(zip(('A','B'),cameras)))
            scene=load_scene_config('config/scenes/multiview_playground.toml').scene
            value=(self.old[kind] if frozen else MultiViewProvider)(rig,scene)
            def advance():
                for camera in cameras:camera.number+=1
        self.addCleanup(value.close)
        return value,advance,cameras

    def confirm(self):
        provider,advance,_=self.provider()
        model=WorldModel()
        updater=WorldModelUpdater(model,provider,stale_object_ttl=2.)
        for _ in range(3):
            self.now.return_value+=.2;advance();updater.update()
        self.assertEqual(len(model.objects()),1)
        return provider,advance,model,updater,model.objects()[0].object_id

    def complete(self,provider,updater,model,result,revision):
        # The coordinator's ordering is part of the safety boundary: reject
        # before tracker association, then revalidate publication without awaits.
        if not updater.accepts(result.timestamp,revision):return []
        batch=provider.associate(result,context=model.observation_context())
        return updater.publish(batch,context_revision=revision)

    def assert_equivalent(self,kind):
        old,advance_old,_=self.provider(kind,frozen=True,multiple=True)
        new,advance_new,_=self.provider(kind,multiple=True)
        observed=False
        for _ in range(4):
            self.now.return_value+=.2;advance_old();advance_new()
            expected=old.observe()
            acquired=new.acquire()
            computed=new.processor.compute(acquired)
            actual=new.associate(computed)
            self.assertEqual(canonical(actual),canonical(expected))
            observed|=bool(actual.observations)
        self.assertTrue(observed)
        for observation in actual.observations:
            np.testing.assert_array_equal(new.get_point_cloud(observation.object_id),
                                          old.get_point_cloud(observation.object_id))

    def test_v1_split_preserves_frozen_observations_geometry_ids_and_clouds(self):
        self.assert_equivalent('v1')

    def test_multiview_split_preserves_frozen_observations_geometry_ids_and_clouds(self):
        self.assert_equivalent('mv')

    def test_compute_uses_no_camera_tracker_or_world_model_and_publishes_only_on_owner(self):
        for kind in ('v1','mv'):
            with self.subTest(kind=kind):
                provider,advance,cameras=self.provider(kind)
                advance();self.now.return_value+=.2
                acquired=provider.acquire()
                diagnostics=deepcopy(provider.diagnostics)
                tracker_calls=[]
                original=provider.tracker.update
                def track_on_owner(*args,**kwargs):
                    tracker_calls.append(threading.get_ident())
                    self.assertEqual(threading.get_ident(),self.owner)
                    return original(*args,**kwargs)
                with patch.object(provider.tracker,'update',side_effect=track_on_owner):
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        forbidden=[]
                        for camera in cameras:
                            for method in ('get_depth','get_rgb','get_intrinsics','get_world_from_camera_transform','get_frame_token'):
                                forbidden.append(patch.object(camera,method,side_effect=AssertionError('SDK read in compute')))
                        for guard in forbidden:guard.start()
                        try:result=executor.submit(provider.processor.compute,acquired).result(timeout=10.)
                        finally:
                            for guard in reversed(forbidden):guard.stop()
                    self.assertFalse(tracker_calls)
                    self.assertEqual(canonical(provider.diagnostics),canonical(diagnostics))
                    provider.associate(result)
                self.assertEqual(tracker_calls,[self.owner])
                self.assertFalse(any(value is provider or value is provider.tracker or
                    any(value is camera for camera in cameras) for value in vars(provider.processor).values()))

    def test_acquisition_copies_writable_buffers_and_reuses_immutable_shared_arrays(self):
        provider,advance,_=self.provider('mv')
        advance();prepared=provider.acquire()
        shared=prepared.frames
        for frame in shared:
            for array in (frame.depth,frame.intrinsics,frame.world_from_camera,frame.rgb):
                if array is not None:
                    self.assertFalse(array.flags.writeable)
                    with self.assertRaises(ValueError):array.setflags(write=True)
        provider._last_prepared_tokens=None
        reused=provider.acquire(frames=shared)
        for before,after in zip(shared,reused.frames):
            self.assertIs(before.depth,after.depth)
            self.assertIs(before.intrinsics,after.intrinsics)
            self.assertIs(before.world_from_camera,after.world_from_camera)
        original=provider.rig.cameras['A'].frame.depth
        self.assertFalse(np.shares_memory(original,shared[0].depth))

    def test_held_and_hint_aba_transitions_change_context_revision(self):
        model=WorldModel();model.register(WorldObject('a'));model.register(WorldObject('b'))
        model.set_held('a');first=model.observation_context()
        model.set_held('b');middle=model.observation_context()
        model.set_held('a');last=model.observation_context()
        self.assertEqual(first.held_object_id,last.held_object_id)
        self.assertLess(first.revision,middle.revision);self.assertLess(middle.revision,last.revision)
        model.set_held(None)
        model.expect_object_at('a',[.4,0,.025],ttl=5.);first=model.observation_context()
        model.expect_object_at('a',[.4,.2,.025],ttl=5.);middle=model.observation_context()
        model.expect_object_at('a',[.4,0,.025],ttl=5.);last=model.observation_context()
        np.testing.assert_array_equal(first.association_hints[0].expected_position,last.association_hints[0].expected_position)
        self.assertEqual(first.association_hints[0].expires_at,last.association_hints[0].expires_at)
        self.assertLess(first.revision,middle.revision);self.assertLess(middle.revision,last.revision)

    def test_direct_uncertainty_aba_changes_revision_without_changing_attachment(self):
        model=WorldModel();model.register(WorldObject('a'));model.set_held('a')
        revision=model.observation_context().revision
        model.held_object_uncertain=True
        middle=model.observation_context().revision
        model.held_object_uncertain=False
        self.assertLess(revision,middle)
        self.assertLess(middle,model.observation_context().revision)
        self.assertEqual(model.held_object_id,'a')

    def test_hint_context_owns_immutable_position_arrays(self):
        model=WorldModel();model.register(WorldObject('a'))
        position=np.array([.4,.2,.025])
        model.expect_object_at('a',position,ttl=5.)
        context=model.observation_context();position[:]=0
        stored=context.association_hints[0].expected_position
        np.testing.assert_array_equal(stored,[.4,.2,.025])
        with self.assertRaises(ValueError):stored[0]=99
        with self.assertRaises(ValueError):stored.setflags(write=True)

    def test_held_aba_rejects_old_geometry_before_tracker_or_world_publication(self):
        provider,advance,model,updater,key=self.confirm()
        self.now.return_value+=.2;advance();revision=model.observation_context().revision
        result=provider.processor.compute(provider.acquire())
        before=canonical(model.require(key))
        model.set_held(key);model.set_held(None)
        with patch.object(provider.tracker,'update',side_effect=AssertionError('outdated tracker update')):
            self.assertEqual(self.complete(provider,updater,model,result,revision),[])
        self.assertEqual(canonical(model.require(key)),before)

    def test_frame_acquired_before_place_hint_cannot_consume_or_overwrite_it(self):
        provider,advance,model,updater,key=self.confirm()
        self.now.return_value+=.2;advance();revision=model.observation_context().revision
        result=provider.processor.compute(provider.acquire())
        hint=model.expect_object_at(key,[.45,.25,.025],ttl=5.,reason='place')
        seen=model.require(key).last_seen
        with patch.object(provider.tracker,'update',side_effect=AssertionError('pre-place frame associated')):
            self.assertEqual(self.complete(provider,updater,model,result,revision),[])
        self.assertEqual(canonical(model.association_hints()),canonical((hint,)))
        self.assertEqual(model.require(key).last_seen,seen)

    def test_clear_lost_and_ttl_forget_invalidate_pending_geometry(self):
        for method in ('clear_lost','cleanup'):
            with self.subTest(method=method):
                provider,advance,model,updater,key=self.confirm()
                self.now.return_value+=.2;advance();revision=model.observation_context().revision
                result=provider.processor.compute(provider.acquire())
                model.require(key).visible=False
                if method=='cleanup':self.now.return_value+=2.1
                getattr(updater,method)()
                self.assertFalse(model.exists(key))
                self.assertIsNone(provider.tracker.get(key))
                self.assertIsNone(provider.get_point_cloud(key))
                self.assertGreater(model.observation_context().revision,revision)
                with patch.object(provider.tracker,'update',side_effect=AssertionError('forgotten track resurrected')):
                    self.assertEqual(self.complete(provider,updater,model,result,revision),[])

    def test_publication_keeps_original_time_and_rejects_outdated_revision_or_timestamp(self):
        provider,_,_=self.provider()
        model=WorldModel();updater=WorldModelUpdater(model,provider)
        revision=model.observation_context().revision
        batch=ObservationBatch([ObjectObservation('a',pose=Pose([.4,0,.025]))],99.8)
        self.assertTrue(updater.accepts(99.8,revision))
        updater.publish(batch,context_revision=revision)
        self.assertEqual(model.require('a').last_seen,99.8)
        self.assertEqual(updater.health.last_frame_at,99.8)
        self.assertFalse(updater.accepts(99.8,model.observation_context().revision))
        model.set_held('a');model.set_held(None)
        unseen=ObservationBatch([ObjectObservation('a',pose=Pose([.7,0,.025]))],99.9)
        self.assertEqual(updater.publish(unseen,context_revision=revision),[])
        np.testing.assert_array_equal(model.require('a').pose.position,[.4,0,.025])
        self.assertEqual(updater.health.last_frame_at,99.8)
        self.now.return_value=100.9
        self.assertFalse(updater.accepts(99.9,model.observation_context().revision))

    def test_capture_diagnostics_follow_last_published_frame_not_unpublished_computation(self):
        provider,advance,cameras=self.provider()
        advance();first=provider.processor.compute(provider.acquire());provider.associate(first)
        published=canonical(provider.diagnostics)
        with tempfile.TemporaryDirectory() as temporary:
            first_path=save_capture(provider,Path(temporary)/'published.npz')
            cameras[0].push(self.empty);self.now.return_value+=.2
            provider.processor.compute(provider.acquire())
            second_path=save_capture(provider,Path(temporary)/'still_published.npz')
            self.assertEqual(canonical(provider.diagnostics),published)
            with np.load(first_path,allow_pickle=False) as old,np.load(second_path,allow_pickle=False) as new:
                for key in ('frame_id','timestamp','depth','rgb','intrinsics','world_from_camera'):
                    np.testing.assert_array_equal(old[key],new[key])
                self.assertEqual(json.loads(str(old['settings'])),json.loads(str(new['settings'])))

    def test_completion_after_close_does_not_associate_or_publish(self):
        provider,advance,_=self.provider()
        advance();prepared=provider.acquire()
        started,released=threading.Event(),threading.Event()
        compute=provider.processor.compute
        def blocked():
            started.set()
            if not released.wait(5.):raise AssertionError('test worker was not released')
            return compute(prepared)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future=executor.submit(blocked)
            self.assertTrue(started.wait(1.))
            provider.close();released.set();result=future.result(timeout=10.)
        with patch.object(provider.tracker,'update',side_effect=AssertionError('late association')):
            self.assertIsNone(provider.associate(result).timestamp)


if __name__=='__main__':
    unittest.main()
