"""Selected normal IK precedes all test controls; independent veto is mandatory."""
import asyncio
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure
from research.rgbd_box_geometry.autonomous_trial import AutonomousPlanner,classify,fixed_request_delay
from tests.unit.test_global_frozen_rgbd_v5 import support


class AutonomousTrialTests(unittest.TestCase):
    def test_classifier_preserves_first_stage_and_specific_backend_code(self):
        for code,category in [('base_geometry_rejected','EXPECTED_SAFE_REJECTION'),
            ('planning_source_stale','SOURCE_FAILURE'),('planning_source_reconstruction','RECONSTRUCTION_FAILURE'),
            ('research_physical_veto','PHYSICAL_VETO'),('tracking_failure','EXECUTION_FAILURE'),
            ('rrt_failed','PLANNING_FAILURE')]:
            result={'metrics':{'selected_result':{'status':'failure','failure_code':'backend_failure',
                     'details':{'failure_code':code},'message':'qualification failed'}}}
            self.assertEqual(classify(result),category)

    def fixture(self):
        primary=NS(joint_names=tuple(f'panda_joint{i}' for i in range(1,8)),diagnostics={},
                   _config={'urdf_path':'unused'},close=Mock())
        factory=Mock(return_value=primary)
        backend=NS(world=NS(current_time=1.,current_time_step_index=2),end_effector_frame='right_gripper',
                   motion_strategy=NS(last_plan_trace={'frozen_rgbd':NS(generation=0,timestamp=1.,view=NS(inferred=None))}))
        state=dict(mapping_inputs=[],paired_plans={},motion_payloads={},physical_evaluation={},
                   research_record={},robot_parts=[],physical_obstacles=[])
        wrapper=AutonomousPlanner(backend,state=state,selected_source='frozen_rgbd',capture_rgbd=Mock(),
            gt_snapshot=support(),physical_context={},production_factory=factory,seed=123456)
        plan=NS(joint_names=primary.joint_names,route=np.array([[0.]*7,[-.2]*7,[.3]*7]),diagnostics={})
        return wrapper,state,plan,factory

    def test_ordinary_selected_goal_flows_out_to_controls_and_veto_stops_return(self):
        for allowed in (True,False):
            wrapper,state,plan,factory=self.fixture();labels=[]
            def run(label,planner,target,start,world,tolerances):
                labels.append(label)
                route=plan.route if label!='unobstructed' else plan.route[[0,-1]]
                state['paired_plans'][label]={'planner_success':True,'route':route.tolist()}
                state['motion_payloads'][label]={'label':label}
                return plan,None
            wrapper._run=run
            def evaluate(evaluator,payload):
                return dict(execution_allowed=allowed if payload['label']=='reconstructed' else False,
                    status='VETO_PREDICTED_PHYSICAL_COLLISION' if payload['label']=='unobstructed' else 'ALLOW')
            with patch('scripts.global_detour_fixed_target_v4_3.FixedJointTargetPlanner',return_value=NS(close=Mock())) as controls, \
                 patch('scripts.evaluate_global_rgbd_v5.PhysicalMotionEvaluator'), \
                 patch('robot_skill_stack.integrations.isaac.manipulation.motion.frozen_rgbd.planning_world_from_frozen',return_value=support()), \
                 patch('research.rgbd_box_geometry.autonomous_trial.bounded_evaluate',side_effect=evaluate):
                if allowed:self.assertIs(wrapper.plan(Pose([.45,.25,.35],[0.,0.,1.,0.]),np.zeros(7),support()),plan)
                else:
                    with self.assertRaisesRegex(PlanningFailure,'independent physical'):wrapper.plan(Pose([.45,.25,.35],[0.,0.,1.,0.]),np.zeros(7),support())
                self.assertEqual(labels,['reconstructed','static','occupied','unobstructed'])
                for call in controls.call_args_list:self.assertEqual(call.kwargs['fixed_target'].q,tuple(plan.route[-1]))
                self.assertFalse(state['research_record']['selected_uses_fixed_joint_target'])
                self.assertEqual(factory.call_args.kwargs,{'seed':123456})
            wrapper.close()

    def test_selected_failure_does_not_request_gt_goal_or_evaluate_controls(self):
        wrapper,state,plan,_=self.fixture()
        wrapper._run=Mock(return_value=(None,PlanningFailure('base_geometry_rejected','floor')))
        with patch('scripts.global_detour_fixed_target_v4_3.FixedJointTargetPlanner') as controls:
            with self.assertRaises(PlanningFailure):wrapper.plan(Pose([.45,.25,.35],[0.,0.,1.,0.]),np.zeros(7),support())
            controls.assert_not_called()
        self.assertNotIn('fixed_joint_target',state)
        wrapper.close()

    def test_ordinary_request_resamples_physical_start_before_planning(self):
        wrapper,state,plan,_=self.fixture()
        state.update(ordinary_request_timing=True,observer=object(),capture=object())
        fresh={'actual_start_q':[.2]*9}
        def run(*a,**k):
            self.assertIs(wrapper.context,fresh)
            return None,PlanningFailure('rrt_failed','test')
        wrapper._run=run
        with patch('research.rgbd_box_geometry.autonomous_trial.bench.physical_context_from_observer',return_value=fresh) as refresh:
            with self.assertRaises(PlanningFailure):wrapper.plan(Pose([.45,.25,.35],[0.,0.,1.,0.]),np.zeros(7),support())
            refresh.assert_called_once_with(wrapper.backend,state['observer'],state['capture'])
        wrapper.close()


class OrdinaryDelayTests(unittest.IsolatedAsyncioTestCase):
    async def test_declared_delays_do_not_consult_mapping_or_wait_for_publication(self):
        for delay in (0.,.3,.6):
            now=[10.];calls=[]
            async def update():
                calls.append(True);now[0]+=.1
            result=await fixed_request_delay(delay,update,lambda:now[0])
            self.assertFalse(result['publication_wait'])
            self.assertEqual(result['requested_delay_s'],delay)
            self.assertGreaterEqual(result['actual_delay_s'],delay)
            self.assertLess(result['actual_delay_s'],delay+.101)
            if delay==0.:self.assertFalse(calls)

    async def test_fixed_delay_propagates_cancellation(self):
        async def update():raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await fixed_request_delay(.3,update,lambda:0.)


if __name__=='__main__':unittest.main()
