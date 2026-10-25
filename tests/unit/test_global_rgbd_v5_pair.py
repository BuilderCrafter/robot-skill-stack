from collections import deque
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure
from scripts import benchmark_global_rgbd_v5 as harness
from scripts import global_rgbd_v5_pair as pair


ARM = tuple(f'panda_joint{i}' for i in range(1,8))
LULA = 'robot_skill_stack.integrations.isaac.manipulation.motion.lula_global.LulaGlobalPlanner'
FIXED = 'scripts.global_detour_fixed_target_v4_3.FixedJointTargetPlanner'
EVALUATOR = 'scripts.evaluate_global_rgbd_v5.PhysicalMotionEvaluator'


def plan():
    q=np.zeros((2,7))
    return NS(joint_names=ARM,route=q,duration=.01,velocity_limits=np.ones(7),
        lower_limits=-np.ones(7),upper_limits=np.ones(7),acceleration_limits=np.ones(7),jerk_limits=np.ones(7),
        trajectory=NS(get_joint_targets=lambda t:(np.zeros(7),np.zeros(7))),
        diagnostics=dict(selected_ik_index=0,ik_attempts=[dict(ik_q=list(np.zeros(7)))],marker='original'))


class MatchedPairTests(unittest.TestCase):
    def state(self):
        return dict(paired_plans={},motion_payloads={},physical_evaluation={},robot_parts=(),physical_obstacles=())

    def planner(self,state,source,capture):
        backend=NS(world=NS(current_time=1.,current_time_step_index=60),end_effector_frame='right_gripper')
        return pair.MatchedSourcePlanner(backend,state=state,selected_source=source,
            capture_rgbd=capture,gt_snapshot='GT',physical_context={})

    def fake(self,result):
        return NS(joint_names=ARM,_config={'urdf_path':'actual.urdf'},plan=Mock(return_value=result),close=Mock())

    def test_same_start_uses_distinct_worlds_and_only_validated_joint_goal_crosses_boundary(self):
        gt_plan,rgbd_plan=plan(),plan()
        gt,rgbd=self.fake(gt_plan),self.fake(rgbd_plan)
        state=self.state()
        capture=Mock(return_value='CAMERA')
        with patch(LULA,return_value=gt),patch(FIXED,return_value=rgbd) as factory, \
                patch(EVALUATOR),patch.object(pair,'bounded_evaluate',return_value={'execution_allowed':True}):
            selected=self.planner(state,'static',capture)
            actual=selected.plan(Pose([.45,0.,.35],[0.,0.,1.,0.]),np.zeros(7),'SELECTED_GT',
                position_tolerance=.025,orientation_tolerance=.10)
        self.assertIs(actual,gt_plan)
        capture.assert_called_once_with()
        self.assertEqual(gt.plan.call_args.args[2],'GT')
        self.assertEqual(rgbd.plan.call_args.args[2],'CAMERA')
        self.assertEqual(factory.call_args.kwargs['fixed_target'].q,(0.,)*7)
        self.assertTrue(state['matched_joint_goal'])
        self.assertTrue(state['matched_start_and_physics_instant'])
        self.assertEqual(state['urdf_path'],'actual.urdf')

    def test_rgbd_selected_world_never_recaptures_and_physical_veto_retains_planner_success(self):
        state=self.state()
        capture=Mock(side_effect=AssertionError('Second capture forbidden'))
        gt,rgbd=self.fake(plan()),self.fake(plan())
        with patch(LULA,return_value=gt),patch(FIXED,return_value=rgbd),patch(EVALUATOR), \
                patch.object(pair,'bounded_evaluate',side_effect=[{'execution_allowed':True},
                    {'execution_allowed':False,'status':'VETO_UNRESOLVED_CLEARANCE'}]):
            selected=self.planner(state,'frozen_rgbd',capture)
            with self.assertRaises(PlanningFailure) as caught:
                selected.plan(Pose([.45,0.,.35],[0.,0.,1.,0.]),np.zeros(7),'CAMERA',
                    position_tolerance=.025,orientation_tolerance=.10)
        self.assertEqual(caught.exception.code,'research_physical_veto')
        self.assertTrue(state['paired_plans']['frozen_rgbd']['planner_success'])
        capture.assert_not_called()
        self.assertEqual(rgbd.plan.call_args.args[2],'CAMERA')

    def test_source_rejection_remains_a_source_error_without_substituting_gt_into_camera_plan(self):
        state=self.state()
        gt=self.fake(plan())
        with patch(LULA,return_value=gt) as factory,patch(EVALUATOR), \
                patch.object(pair,'bounded_evaluate',return_value={'execution_allowed':True}):
            selected=self.planner(state,'static',Mock(side_effect=PlanningFailure('planning_source_stale','Stale')))
            selected.plan(Pose([.45,0.,.35],[0.,0.,1.,0.]),np.zeros(7),'GT',
                position_tolerance=.025,orientation_tolerance=.10)
        self.assertEqual(factory.call_count,1)
        self.assertEqual(state['paired_plans']['frozen_rgbd']['stage'],'SOURCE')
        self.assertEqual(state['paired_plans']['frozen_rgbd']['failure_code'],'planning_source_stale')
        self.assertNotIn('frozen_rgbd',state['motion_payloads'])

    def test_sampling_rejects_unbounded_duration_before_allocating_times(self):
        value=plan()
        value.duration=1e12
        with patch.object(pair.np,'linspace',side_effect=AssertionError('Unbounded allocation')):
            with self.assertRaises(PlanningFailure):pair.sample_trajectory(value)

    def test_executor_diagnostics_protocol_retains_selected_success_and_failure_metadata(self):
        for failure in (False,True):
            with self.subTest(failure=failure):
                state=self.state()
                gt=self.fake(plan())
                if failure:
                    gt.plan.side_effect=PlanningFailure('invalid_goal','Blocked goal',{'ik_stage':'collision'})
                with patch(LULA,return_value=gt),patch(EVALUATOR), \
                        patch.object(pair,'bounded_evaluate',return_value={'execution_allowed':True}):
                    selected=self.planner(state,'static',Mock(side_effect=PlanningFailure('planning_source_stale','Stale')))
                    self.assertEqual(selected.diagnostics,{})
                    request=lambda:selected.plan(Pose([.45,0.,.35],[0.,0.,1.,0.]),np.zeros(7),'GT',
                        position_tolerance=.025,orientation_tolerance=.10)
                    if failure:
                        with self.assertRaises(PlanningFailure):request()
                        self.assertEqual(selected.diagnostics['ik_stage'],'collision')
                    else:
                        request()
                        self.assertEqual(selected.diagnostics['marker'],'original')
                    result=selected.diagnostics
                    result['caller_only']=True
                    self.assertNotIn('caller_only',selected.diagnostics)
                    selected.close()
                    gt.close.assert_called_once_with()


class HarnessHelperTests(unittest.TestCase):
    def test_camera_timing_observes_existing_calls_tokens_and_restores_without_sensor_reads(self):
        frames=(NS(camera_id='/A',token=(1,.1)),NS(camera_id='/B',token=(1,.1)))
        original=Mock(side_effect=[frames,None,RuntimeError('Unavailable')])
        pool=NS(batch=original)
        state={}
        restore=harness.install_camera_timing(pool,state)
        self.assertIs(pool.batch('semantic',('/A','/B'),max_skew=.025),frames)
        self.assertIsNone(pool.batch('mapping',('/A','/B')))
        with self.assertRaises(RuntimeError):pool.batch('mapping',('/A','/B'))
        restore()
        self.assertIs(pool.batch,original)
        self.assertEqual(original.call_count,3)
        self.assertTrue(state['camera_acquisition'][0]['tokens_advanced'])
        self.assertFalse(state['camera_acquisition'][1]['tokens_advanced'])
        self.assertEqual(state['camera_acquisition'][2]['error'],'RuntimeError')
        summary=harness.camera_timing_summary(state['camera_acquisition'])
        self.assertEqual(summary['semantic']['successful_batches'],1)
        self.assertEqual(summary['mapping']['successful_batches'],0)

    def test_transport_retains_immutable_mapping_and_readonly_array_without_dumps_api(self):
        data=np.frombuffer(np.arange(3.).tobytes(),dtype=float)
        restored=pickle.loads(harness.serialize_mapping_input(MappingProxyType({'depth':data})))
        np.testing.assert_array_equal(restored['depth'],data)
        with self.assertRaises(TypeError):restored['other']=1
        with self.assertRaises(ValueError):restored['depth'].setflags(write=True)

    def test_publication_telemetry_uses_published_count_and_retains_actual_revision(self):
        state={'mapping_publications':[]}
        service=NS(health={'generation':0,'published':1},diagnostics={'accepted':True},
            snapshot=Mock(return_value=NS(generation=0,revision=5,timestamp=10.)))
        harness.retain_mapping_publication(state,service)
        harness.retain_mapping_publication(state,service)
        self.assertEqual(service.snapshot.call_count,1)
        self.assertEqual(state['mapping_publications'][0]['revision'],5)
        service.health['published']=2
        service.snapshot.return_value.revision=6
        harness.retain_mapping_publication(state,service)
        self.assertEqual([row['revision'] for row in state['mapping_publications']],[5,6])

    def test_source_failure_archive_error_is_retained_without_raising_save_failure(self):
        with TemporaryDirectory() as temporary:
            state=dict(mapping_inputs=deque(),mapping_publications=[],mapping_submissions=[],motion_payloads={},
                physical_evaluation={},paired_plans={'frozen_rgbd':{'stage':'SOURCE'}},
                capture_view=NS(generation=0,timestamp=1.))
            with patch('scripts.evaluate_global_rgbd_v5.write_read_view_archive',side_effect=ValueError('No usable packet')):
                operations,archive=harness.save_capture(NS(artifacts_dir=temporary),state)
            self.assertEqual(operations,[])
            self.assertFalse(archive['source_accepted_for_planning'])
            self.assertIn('No usable packet',archive['archive_error'])
            self.assertTrue((Path(temporary)/'mapping_publications.json').exists())


if __name__=='__main__':unittest.main()
