"""Focused source/geometry/cleanup invariants, with no SDK or live cameras."""
from copy import deepcopy
from types import SimpleNamespace as NS,ModuleType
import sys
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock,patch,AsyncMock

import numpy as np

from . import rrt_bridge as b
from .native_trial import Comparison
from . import native_trial
from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import StaticObstacle,StaticWorldSnapshot
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure
from scripts.benchmark_global_rgbd_v5 import restore_strategy_hooks


def model(center=(.105,.005,.185),dimensions=(.04,.04,.04)):
    return dict(status='qualified',geometry=dict(center=center,dimensions=dimensions,yaw_rad=0.,rotation=np.eye(3)))


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.frames=tuple(NS(camera_id=str(i),token=(5.,1.),acquired_at=10.,depth=np.zeros((480,640)),
                        intrinsics=np.eye(3),world_from_camera=np.eye(4)) for i in range(2))
        self.records=[dict(camera_id=f.camera_id,token=f.token,acquired_at=f.acquired_at,
                          depth_shape=f.depth.shape,intrinsics=f.intrinsics.copy(),world_from_camera=f.world_from_camera.copy()) for f in self.frames]
        self.view=NS(generation=2,revision=3,timestamp=10.)
        self.payload=NS(generation=2,frames=self.frames,ground_truth=None,robot_snapshots={'known_robot':'only'})
        self.source=NS(generation=2,snapshot=lambda:self.view,diagnostics={'source_frames':self.records},clock=lambda:10.5)
        self.floor='0:1:0:1:1:2';self.hazard='10:11:0:1:20:21'

    def worlds(self,ids=None):
        boxes=[];obstacles=[StaticObstacle('/support','cuboid',(0,0,-25),dimensions=(50,50,50),source_kind='Plane')]
        for key in ids or (self.floor,self.hazard):
            x0,x1,y0,y1,z0,z1=map(int,key.split(':'))
            center=(.005*(x0+x1),.005*(y0+y1),-.02+.005*(z0+z1));dims=(.01,.01,.01)
            boxes.append(NS(id=key,center=center,dimensions=dims))
            obstacles.append(StaticObstacle('/__rgbd_occupied__/'+key,'cuboid',center,dimensions=dims,source_kind='rgbd_occupied_cells'))
        world=StaticWorldSnapshot(tuple(obstacles),10.,1.,'/Franka',source_metadata={'source_type':'rgbd'})
        frozen=NS(generation=2,revision=3,timestamp=10.,boxes=tuple(boxes),
                  spec=NS(bounds_min=(0,0,-.02),bounds_max=(1,1,1),voxel_size=.01),
                  metadata={'source_frames':self.records},age_seconds=.5,evaluated_at=10.5,view=object())
        return world,frozen

    def test_exact_exposure_association_and_mismatch(self):
        self.assertIs(b.associate(self.source,[self.payload],(2,3,10.))[1],self.payload)
        for change in ('token','time','intrinsics','extrinsics','generation','revision','gt'):
            with self.subTest(change=change):
                records=deepcopy(self.records);source=NS(**vars(self.source));source.diagnostics={'source_frames':records}
                payload=NS(**vars(self.payload));expected=(2,3,10.)
                if change=='token':records[0]['token']=(6.,1.)
                elif change=='time':records[0]['acquired_at']=9.
                elif change=='intrinsics':records[0]['intrinsics'][0,0]=2.
                elif change=='extrinsics':records[0]['world_from_camera'][0,3]=1.
                elif change=='generation':source.generation=4
                elif change=='revision':expected=(2,4,10.)
                else:payload.ground_truth=object()
                with self.assertRaises(PlanningFailure):b.associate(source,[payload],expected)

    def test_floor_and_original_union_preserved(self):
        world,frozen=self.worlds();original=world.obstacles;view=frozen.view
        combined,audit=b.combine(world,frozen,[model(dimensions=(1,1,1))])
        self.assertEqual(audit['combined_camera_proxies'],2)
        self.assertEqual(audit['protected_floor_ids'],[self.floor])
        self.assertIn(self.floor,audit['retained_selected_ids'])
        self.assertEqual(sum(audit['subsumed'].values(),[]),[self.hazard])
        self.assertEqual(world.obstacles,original);self.assertIs(frozen.view,view)
        self.assertEqual(combined.source_metadata['camera_obstacle_count'],2)

    def test_center_only_containment_is_insufficient(self):
        world,frozen=self.worlds();_,audit=b.combine(world,frozen,[model(dimensions=(.005,.005,.005))])
        self.assertEqual(audit['combined_camera_proxies'],3)
        self.assertEqual(sum(audit['subsumed'].values(),[]),[])

    def test_invalid_unqualified_and_overbudget_geometry(self):
        world,frozen=self.worlds()
        for models in ([],[dict(status='unqualified',geometry=None)],[model()|{'geometry':model()['geometry']|{'rotation':np.zeros((3,3))}}]):
            with self.assertRaises(PlanningFailure):b.combine(world,frozen,models)
        ids=[f'{i}:{i+1}:0:1:20:21' for i in range(20,84)]
        world,frozen=self.worlds(ids)
        with self.assertRaises(PlanningFailure) as e:b.combine(world,frozen,[model()])
        self.assertEqual(e.exception.code,'research_proxy_budget')

    def test_gt_obstacles_cannot_enter_combined_camera_world(self):
        world,frozen=self.worlds()
        bad=StaticObstacle('/fixture','cuboid',(1,1,1),dimensions=(.1,.1,.1),source_kind='Cube')
        world=StaticWorldSnapshot((*world.obstacles,bad),10.,1.,'/Franka',source_metadata={})
        with self.assertRaises(PlanningFailure):b.combine(world,frozen,[model()])

    def test_final_source_capture_is_after_inference_and_stale_propagates(self):
        calls=[];record={};scene=NS(support_plane_z=0.)
        result=dict(components=[dict(component=0,points=30,robot_envelope_points=0,reason='box')])
        def reconstruct(*args):calls.append('infer');return result,(),()
        def stale(*args,**kwargs):calls.append('capture');raise PlanningFailure('planning_source_stale','stale')
        with patch.object(b.ref,'reconstruct',side_effect=reconstruct) as infer,patch.object(b.visibility,'qualify',return_value=model()):
            with self.assertRaises(PlanningFailure):b.capture(None,self.source,[self.payload],(2,3,10.),scene,record,stale)
            infer.assert_called_once_with(self.frames,scene,self.payload.robot_snapshots)
        self.assertEqual(calls,['infer','capture']);self.assertEqual(record['failure']['code'],'planning_source_stale')

    def test_unqualified_or_robot_connected_stops_before_capture(self):
        for component in (dict(component=0,points=30,robot_envelope_points=0,reason='box'),
                          dict(component=0,points=30,robot_envelope_points=1,reason='possible_robot_component')):
            with patch.object(b.ref,'reconstruct',return_value=({'components':[component]},(),())),patch.object(b.visibility,'qualify',return_value={'status':'unqualified','reason':'bound'}):
                capture=Mock()
                with self.assertRaises(PlanningFailure):b.capture(None,self.source,[self.payload],(2,3,10.),NS(),{},capture)
                capture.assert_not_called()

    def test_success_keeps_read_view_and_final_publication_mismatch_rejects(self):
        world,frozen=self.worlds();view=frozen.view;record={}
        result=dict(components=[dict(component=0,points=30,robot_envelope_points=0,reason='box')])
        with patch.object(b.ref,'reconstruct',return_value=(result,(),())),patch.object(b.visibility,'qualify',return_value=model()):
            combined,returned=b.capture(None,self.source,[self.payload],(2,3,10.),NS(support_plane_z=0.),record,Mock(return_value=(world,frozen)))
            self.assertIs(returned,frozen);self.assertIs(returned.view,view)
            self.assertEqual(len(combined.obstacles),3)
            frozen.revision=4
            with self.assertRaises(PlanningFailure):b.capture(None,self.source,[self.payload],(2,3,10.),NS(),{},Mock(return_value=(world,frozen)))

    def test_delegate_veto_or_missing_common_goal_cannot_reach_counterplan(self):
        wrapper=Comparison.__new__(Comparison);wrapper.backend=NS(world=NS(current_time=1,current_time_step_index=2));wrapper.state={};wrapper.empty=None
        for failure in (PlanningFailure('research_physical_veto','unsafe'),None):
            wrapper.delegate=NS(plan=Mock(side_effect=failure,return_value=NS()))
            with patch('scripts.global_detour_fixed_target_v4_3.FixedJointTargetPlanner') as counter:
                with self.assertRaises(PlanningFailure):wrapper.plan(None,np.zeros(7),None)
                counter.assert_not_called()

    def test_failed_capture_restores_harness_seams_and_closes_controls(self):
        strategy=NS(_rgbd_world_capture=object(),_planner_factory=object());old=dict(vars(strategy));restore=restore_strategy_hooks(strategy)
        try:
            strategy._rgbd_world_capture=Mock(side_effect=RuntimeError('capture'));strategy._planner_factory=object()
            with self.assertRaises(RuntimeError):strategy._rgbd_world_capture()
        finally:restore()
        self.assertEqual(vars(strategy),old)
        wrapper=Comparison.__new__(Comparison);wrapper.delegate=NS(close=Mock(side_effect=RuntimeError('close')));wrapper.empty=NS(close=Mock())
        with self.assertRaises(RuntimeError):wrapper.close()
        wrapper.empty.close.assert_called_once()

    def test_receipt_precedes_nonreturning_sdk_close(self):
        app=NS(close=Mock(side_effect=SystemExit(0)))
        sdk=ModuleType('isaacsim');sdk.SimulationApp=Mock(return_value=app)
        with TemporaryDirectory() as tmp:
            output=Path(tmp)/'attempt'
            with patch.dict(sys.modules,{'isaacsim':sdk}),patch.object(sys,'argv',['native_trial','--output',str(output)]), \
                 patch.object(native_trial.bench,'execute',return_value=False) as execute:
                with self.assertRaises(SystemExit):native_trial.main()
                execute.assert_called_once()
            receipt=json.loads((output/'research.json').read_text())
            self.assertFalse(receipt['measurement_pass']);self.assertTrue(receipt['research_sources_unchanged'])
            app.close.assert_called_once()


class ReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_accepted_publication_only_and_strict_deadline(self):
        fixture=BridgeTests();fixture.setUp();source=fixture.source
        source.health={'published':7,'fresh':True}
        omni,kit,app_module=(ModuleType(x) for x in ('omni','omni.kit','omni.kit.app'))
        omni.kit=kit;kit.app=app_module
        update=AsyncMock();app_module.get_app=lambda:NS(next_update_async=update)
        for published,clock,allowed in ((8,[0,0,1,1],True),(9,[0,0,1],False),(8,[0,0,5.1],False)):
            state=dict(mapping_inputs=[fixture.payload],research_record={})
            observation=NS(diagnostics={'state':'running'},robot_filter=NS(source_error=None if allowed else 'captured source failure'))
            bundle=NS(spatial_mapping=source,observation_runtime=observation)
            with patch.dict(sys.modules,{'omni':omni,'omni.kit':kit,'omni.kit.app':app_module}), \
                 patch.object(native_trial,'time',NS(monotonic=Mock(side_effect=clock))), \
                 patch.object(native_trial.bench,'retain_mapping_publication',return_value={'published':published,'fresh':True}):
                if allowed:
                    await native_trial.prepare(bundle,state)
                    self.assertEqual(state['research_ready'],(2,3,10.))
                else:
                    with self.assertRaises(PlanningFailure):await native_trial.prepare(bundle,state)
                    self.assertNotIn('research_ready',state)
                self.assertEqual(state['research_record']['readiness_before']['publication'],(2,3,10.))
                diagnostic=state['research_record']['readiness_after']
                self.assertEqual(diagnostic['robot_filter_error'],observation.robot_filter.source_error)
                self.assertEqual(diagnostic['observation'],{'state':'running'})
                self.assertEqual(diagnostic['source_frames'][0]['camera_id'],'0')


if __name__=='__main__':unittest.main()
