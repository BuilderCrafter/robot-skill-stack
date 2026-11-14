"""Autonomous same-worker geometry and conservative source-bound assembly."""
import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pickle
import hashlib
import tomllib
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.world.spatial_mapping import reconstruction as rec
from robot_skill_stack.world.spatial_mapping.box_geometry import reference, visibility
from robot_skill_stack.world.spatial_mapping.runtime import MappingInput, MappingProcessor
from robot_skill_stack.integrations.isaac.config import MotionConfig


def model(center=(.105,.005,.185), dimensions=(.04,.04,.04)):
    return dict(status='qualified',geometry=dict(center=center, dimensions=dimensions,
                rotation=np.eye(3), yaw_rad=0.))


def fixture(ids=('0:1:0:1:1:2','10:11:0:1:20:21'), models=None):
    records=tuple(dict(camera_id=str(i),token=(1.,.1),acquired_at=10.,depth_shape=(480,640),
        intrinsics=np.eye(3),world_from_camera=np.eye(4)) for i in range(2))
    product=rec.InferredGeometry(2,3,10.,'qualified','ok',tuple(models or [model()]),records,{})
    boxes=[]
    for key in ids:
        a,b,c,d,e,f=map(int,key.split(':'))
        boxes.append(NS(id=key,center=(.005*(a+b),.005*(c+d),-.02+.005*(e+f)),
                        dimensions=(.01,.01,.01),orientation_wxyz=(1.,0.,0.,0.)))
    frozen=NS(generation=2,revision=3,timestamp=10.,view=NS(inferred=product),boxes=tuple(boxes),
        spec=NS(voxel_size=.01),metadata={'source_frames':records,'extraction':{'support_plane_z':0.}})
    return frozen, product


class AssemblyTests(unittest.TestCase):
    def test_complete_union_floor_and_eight_corners(self):
        frozen,p=fixture(models=[model(dimensions=(1,1,1))]);original=frozen.boxes
        descriptors,audit,returned=rec.combine(frozen)
        self.assertIs(returned,p);self.assertIs(frozen.boxes,original)
        self.assertEqual(audit['combined_camera_proxies'],2)
        self.assertEqual(audit['protected_floor_ids'],['0:1:0:1:1:2'])
        self.assertEqual(sum(audit['subsumed'].values(),[]),['10:11:0:1:20:21'])
        tiny,_=fixture(models=[model(dimensions=(.005,.005,.005))])
        self.assertEqual(rec.combine(tiny)[1]['combined_camera_proxies'],3)

    def test_budget_and_invalid_numeric_geometry(self):
        ids=tuple(f'{i}:{i+1}:0:1:20:21' for i in range(20,84))
        with self.assertRaisesRegex(ValueError,'64'):rec.combine(fixture(ids)[0])
        for change in ({'dimensions':(-1,.1,.1)},{'center':(np.nan,0,0)},
                       {'rotation':np.zeros((3,3))},{'yaw_rad':np.inf}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                rec.validate_geometry(model()['geometry']|change)

    def test_generation_revision_time_and_each_exposure_field(self):
        for field,value in [('generation',4),('revision',5),('timestamp',9.)]:
            frozen,p=fixture();frozen.view.inferred=replace(p,**{field:value})
            with self.assertRaisesRegex(ValueError,'identity'):rec.combine(frozen)
        for key in ('camera_id','token','acquired_at','depth_shape','intrinsics','world_from_camera'):
            frozen,p=fixture();records=[dict(v) for v in frozen.metadata['source_frames']]
            records[0][key]='wrong' if key=='camera_id' else np.zeros_like(records[0][key])
            frozen.metadata['source_frames']=records
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'exposure'):rec.combine(frozen)
        frozen,p=fixture();frozen.metadata['source_frames']=()
        with self.assertRaisesRegex(ValueError,'cameras'):rec.combine(frozen)

    def test_failed_inference_never_removes_evidence_and_empty_keeps_all_cells(self):
        for status in ('rejected','error'):
            frozen,p=fixture();original=frozen.boxes;frozen.view.inferred=replace(p,status=status,models=())
            with self.assertRaisesRegex(ValueError,'Reconstruction'):rec.combine(frozen)
            self.assertIs(frozen.boxes,original)
        frozen,p=fixture();frozen.view.inferred=replace(p,status='empty',models=())
        descriptors,audit,_=rec.combine(frozen)
        self.assertEqual(len(descriptors),len(frozen.boxes))
        self.assertEqual(audit['inference_status'],'empty')
        self.assertFalse(any(v['source_kind']=='rgbd_inferred_box' for v in descriptors))

    def test_product_deeply_immutable(self):
        frozen,p=fixture()
        with self.assertRaises(TypeError):p.models[0]['geometry']['center']=np.zeros(3)
        with self.assertRaises(ValueError):p.models[0]['geometry']['rotation'].setflags(write=True)


class WorkerInferenceTests(unittest.TestCase):
    def payload(self):
        from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
        frames=tuple(DepthFrame(str(i),(1.,.1),np.ones((480,640)),np.eye(3),np.eye(4),10.) for i in range(2))
        return MappingInput(2,frames,{'0':object(),'1':object()})

    def test_empty_robot_only_mixed_and_unqualified_components(self):
        cases=[([], 'empty'),([dict(component=0,reason='possible_robot_component',points=20,robot_envelope_points=20)],'empty'),
               ([dict(component=0,reason='possible_robot_component',points=20,robot_envelope_points=19)],'rejected'),
               ([dict(component=0,reason='box',points=20,robot_envelope_points=0)],'rejected')]
        for components,expected in cases:
            with patch.object(reference,'reconstruct',return_value=({'components':components},None,None)), \
                 patch.object(visibility,'qualify',return_value={'status':'unqualified','reason':'missing bound'}):
                p=rec.infer(self.payload(),NS(),NS(revision=3,timestamp=10.))
                self.assertEqual(p.status,expected)
                self.assertFalse(p.models)

    def test_multiple_qualified_boxes_and_gt_missing_camera_invalid_depth_refusal(self):
        components=[dict(component=i,reason='box',points=100,robot_envelope_points=0) for i in range(2)]
        payload=self.payload()
        with patch.object(reference,'reconstruct',return_value=({'components':components},None,None)) as infer, \
             patch.object(visibility,'qualify',return_value=model()):
            result=rec.infer(payload,NS(),NS(revision=3,timestamp=10.))
            self.assertEqual(len(result.models),2)
            for bad in (NS(generation=2,frames=payload.frames,ground_truth=object(),robot_snapshots={}),
                        replace(payload,frames=payload.frames[:1]),
                        replace(payload,frames=tuple(replace(f,depth=np.full_like(f.depth,np.nan)) for f in payload.frames))):
                self.assertEqual(rec.infer(bad,NS(),NS(revision=3,timestamp=10.)).status,'error')
            infer.assert_called_once()

    def test_mapping_default_has_zero_reconstruction_work(self):
        from tests.unit.test_spatial_mapping_runtime import settings,gt
        with patch.object(rec,'infer',side_effect=AssertionError('default inference')):
            result=MappingProcessor(settings(),clock=lambda:100.2).compute(MappingInput(ground_truth=gt()))
        self.assertIsNone(result.inferred)


class NumericalContractTests(unittest.TestCase):
    def test_extracted_functions_are_identical_to_frozen_research(self):
        for name in ('reference.py','visibility.py'):
            old=ast.parse((Path('research/rgbd_box_geometry')/name).read_text())
            new=ast.parse((Path('robot_skill_stack/world/spatial_mapping/box_geometry')/name).read_text())
            self.assertEqual(ast.dump(ast.Module(body=old.body[1:],type_ignores=[])),
                             ast.dump(ast.Module(body=new.body[1:],type_ignores=[])))

    def test_missing_rear_lower_top_bounds_and_invalid_or_unobservable_patches(self):
        old=json.loads(Path('research/rgbd_box_geometry/evidence/reference/blind-results.json').read_text())
        original=next(c for c in old['v43']['components'] if c.get('hypothesis'))
        for face in ('y-1','z-1','z+1'):
            c=deepcopy(original);c['faces'][face]['qualified']=False
            self.assertEqual(visibility.qualify(c,())['status'],'unqualified')
        for state in ('invalid_depth','outside_fov','occluded','contradictory_free'):
            samples=[dict(pixel_xy=[i,j],grid_bin=[i,j],state=state) for i in range(3) for j in range(3)]
            self.assertFalse(visibility.patch_decision(samples)['qualified'])
        samples=[dict(pixel_xy=[i,j],grid_bin=[i,j],state='supported' if j<2 else 'outside_fov') for i in range(3) for j in range(3)]
        self.assertTrue(visibility.patch_decision(samples)['qualified'])

    def test_opt_in_configuration_is_explicit_and_bounded(self):
        self.assertEqual(MotionConfig().global_rgbd_representation,'occupied')
        args=dict(strategy='global',global_obstacle_source='frozen_rgbd',global_collision_model='enhanced',global_rgbd_representation='reconstructed')
        self.assertEqual(MotionConfig(**args).max_obstacles,64)
        for changed in ({'max_obstacles':128},{'strategy':'direct'},{'global_obstacle_source':'static'},{'global_rgbd_representation':'unknown'}):
            with self.assertRaises(ValueError):MotionConfig(**(args|changed))


class PublicationTests(unittest.TestCase):
    def source(self):
        from tests.unit.test_frozen_rgbd_world import FrozenRgbdWorldTests
        fixture=FrozenRgbdWorldTests();source=fixture.prepared(support=0.)
        self.addCleanup(fixture.doCleanups)
        view=source.snapshot()
        source._published_inferred=rec.InferredGeometry(view.generation,view.revision,view.timestamp,
            'qualified','ok',(model(center=(0,0,1),dimensions=(.5,.5,.5)),),
            tuple(dict(v) for v in source.diagnostics['source_frames']),{})
        return source,fixture

    def test_real_source_adapter_is_opt_in_and_support_preserved(self):
        from robot_skill_stack.integrations.isaac.manipulation.motion.frozen_rgbd import capture_frozen_rgbd_world
        from tests.unit.test_global_frozen_rgbd_v5 import support
        source,fixture=self.source()
        occupied,old=capture_frozen_rgbd_world(None,lambda:source,support_capture=lambda *a,**k:support())
        inferred,new=capture_frozen_rgbd_world(None,lambda:source,support_capture=lambda *a,**k:support(),representation='reconstructed')
        self.assertEqual(occupied.obstacles[0],inferred.obstacles[0])
        self.assertTrue(any(o.source_kind=='rgbd_inferred_box' for o in inferred.obstacles))
        self.assertTrue(all(o.source_kind!='rgbd_inferred_box' for o in occupied.obstacles))
        self.assertIs(old.view._snapshot,new.view._snapshot)
        np.testing.assert_array_equal(old.view.states,new.view.states)
        self.assertTrue(inferred.source_metadata['occupied_coverage']['occupied_union_preserved'])

    def test_stale_generation_reset_and_late_completion_cannot_reuse_geometry(self):
        from robot_skill_stack.world.spatial_mapping.frozen import capture_frozen_rgbd,FrozenRgbdSourceError
        source,fixture=self.source();saved=source.snapshot().inferred
        fixture.clock[0]=102.
        with self.assertRaises(FrozenRgbdSourceError):capture_frozen_rgbd(source)
        fixture.clock[0]=100.2
        source.submit(fixture.payload(source,3,100.15))
        source.reset();source._worker.finish();source.poll()
        self.assertIsNone(source.snapshot().inferred)
        self.assertIsNot(saved,source.snapshot().inferred)
        self.assertEqual(source.health['published'],2)
        with self.assertRaises(FrozenRgbdSourceError):capture_frozen_rgbd(source)

    def test_source_failure_and_rejected_completion_never_enable_old_geometry(self):
        from robot_skill_stack.world.spatial_mapping.frozen import capture_frozen_rgbd,FrozenRgbdSourceError
        source,fixture=self.source()
        source.note_source_failure('robot model changed')
        with self.assertRaises(FrozenRgbdSourceError):capture_frozen_rgbd(source)
        self.assertEqual(source.health['source_failures'],1)

    def test_worker_attaches_inference_to_same_completion_and_serializes(self):
        from tests.unit.test_frozen_rgbd_world import FrozenRgbdWorldTests
        from robot_skill_stack.runtime.process_worker import _TransportPickler
        import io
        fixture=FrozenRgbdWorldTests();source=fixture.service(support=0.)
        self.addCleanup(fixture.doCleanups)
        source._processor._reconstruction_scene=NS(workspace_min=(-1,-1,0),workspace_max=(1,1,2),support_plane_z=0.)
        result=fixture.publish(source)
        # Small test cameras are not the authored 640x480 pair: geometry fails,
        # while the original accepted occupancy publication remains intact.
        self.assertEqual(result.inferred.status,'error')
        self.assertIs(source.snapshot().inferred,result.inferred)
        self.assertEqual((result.inferred.generation,result.inferred.revision,result.inferred.timestamp),
                         (result.snapshot.generation,result.snapshot.revision,result.snapshot.timestamp))
        stream=io.BytesIO();_TransportPickler(stream,protocol=5).dump(result)
        copied=pickle.loads(stream.getvalue())
        self.assertEqual(copied.inferred.status,'error')


class ArchivedNumericalTests(unittest.TestCase):
    def test_successful_partial_replay_matches_frozen_numerics_without_gt(self):
        folder=Path('outputs/motion-planning-v5/reconstructed-native-proof-postfix-20261114/attempt-1')
        if not folder.exists():self.skipTest('Trusted native RGB-D archive not on this workstation')
        seal=json.loads((folder.parent/'sensor-result-seal.json').read_text())
        raw=(folder/'accepted_mapping_input.pickle').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),seal['files']['accepted_mapping_input.pickle'])
        payload=pickle.loads(raw);cfg=tomllib.loads((folder/'runtime_profile.toml').read_text())['scene']
        scene=NS(**{k:cfg[k] for k in ('workspace_min','workspace_max','support_plane_z')})
        expected=json.loads((folder/'research.json').read_text())
        product=rec.infer(payload,scene,NS(revision=32,timestamp=expected['publication'][2]))
        self.assertEqual(product.status,'qualified');self.assertEqual(len(product.models),2)
        for got,want in zip(product.models,[m for m in expected['models'] if m['status']=='qualified']):
            for key in ('center','dimensions','rotation'):
                np.testing.assert_allclose(got['geometry'][key],want['geometry'][key],atol=1e-12,rtol=0.)

    def test_measured_missing_boundary_robot_merged_and_contradictory_controls(self):
        from research.rgbd_box_geometry.inputs import load_capture
        from research.rgbd_box_geometry.controls import make
        root=Path('outputs/motion-planning-v5/rgbd-base-clearance-20261113')
        if not root.exists():self.skipTest('Trusted original sensor archives unavailable')
        for case in ('static','v43'):
            payload,scene,_=load_capture(root,case)
            result,clusters,envelopes=reference.reconstruct(payload.frames,scene,payload.robot_snapshots)
            component=next(c for c in result['components'] if visibility.qualify(c,payload.frames)['status']=='qualified')
            points=sorted(clusters,key=len,reverse=True)[component['component']]
            for name,(cloud,frames) in make(payload.frames,points,component['hypothesis'],payload.robot_snapshots).items():
                with self.subTest(case=case,control=name):
                    control=reference.qualify(cloud,frames,envelopes)
                    self.assertNotEqual(visibility.qualify(control,frames)['status'],'qualified')
            theta=np.linspace(0,2*np.pi,40);z=np.linspace(-1,1,20)
            sphere=np.array([[np.sqrt(max(0,1-h*h))*np.cos(t),np.sqrt(max(0,1-h*h))*np.sin(t),h] for h in z for t in theta])*.05
            sphere+=component['hypothesis']['center']
            self.assertNotEqual(visibility.qualify(reference.qualify(sphere,payload.frames,envelopes),payload.frames)['status'],'qualified')


if __name__=='__main__':unittest.main()
