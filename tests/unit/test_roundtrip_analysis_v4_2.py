"""Exact recorded-name/time mapping for offline measured figures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import analyze_roundtrip_v4_2 as analysis


class RoundtripAnalysisTests(unittest.TestCase):
    def trace(self):
        names=list(reversed(analysis.ARM))+['panda_finger_joint2','panda_finger_joint1']
        plan_names=list(analysis.ARM[2:]+analysis.ARM[:2])
        samples=[];commands=[];actual=[];actions=[]
        for index in range(3):
            values={name:(number+1)*.1+index*.005 for number,name in enumerate(analysis.ARM)}
            q=[values.get(name,.0023) for name in names]
            plan_q=[values[name]+.01 for name in plan_names]
            samples.append(dict(physics_step=32+index,simulation_time=index*.1,
                joint_feedback=dict(positions=q,velocities=[0.]*9),
                tcp_pose=dict(position=[.45,-.25,.35],orientation=[0.,0.,1.,0.]),
                dynamics=dict(get_solver_position_iteration_counts=dict(available=True,value=[255 if index==1 else 32]))))
            commands.append(dict(sim_time=index*.1,q=plan_q,qd=[0.]*7))
            actual.append(dict(sim_time=index*.1,q=[values[name] for name in plan_names],tracking_error_rad=.01))
            actions.append(dict(simulation_time=index*.1,actual_physx_targets=dict(available=True,
                value=dict(joint_positions=q,joint_indices=None))))
        operation=dict(label='outward',first_sample=0,last_sample=3,first_action=0,last_action=3,
            returned_success=True,target=dict(position=[.45,-.25,.35],orientation=[0.,0.,1.,0.]),
            strategy_diagnostics=dict(trajectory_duration_seconds=0.,terminal_settle_seconds=.25,
                terminal_velocity_limit_rad_s=.01,position_tolerance_m=.025,orientation_tolerance_rad=.10),
            plan_trace=dict(joint_names=plan_names,commanded=commands,actual=actual))
        return dict(metadata=dict(robot_dof_names=names),samples=samples,actions=actions,
            operations=[operation],clearance_curves=dict(outward=[dict(physics_step=32+index,
                simulation_time=index*.1,minimum_m=.04+index*.001) for index in range(3)]))

    def test_native_and_plan_joint_orders_are_both_mapped_by_explicit_names(self):
        trace=self.trace();op=analysis.extract_operation(trace,trace['operations'][0])
        np.testing.assert_allclose(op['q'][0],[.1,.2,.3,.4,.5,.6,.7])
        np.testing.assert_allclose(op['command_q'][0],[.11,.21,.31,.41,.51,.61,.71])
        np.testing.assert_allclose(op['applied_q'][0],[.1,.2,.3,.4,.5,.6,.7])
        self.assertAlmostEqual(op['summary']['exact_same_time_actual_command_error_max_rad'],.01)
        self.assertEqual(op['summary']['exact_same_time_pairs'],3)
        self.assertTrue(op['summary']['solver_position_iterations_restored'])
        self.assertAlmostEqual(op['summary']['observed_terminal_settling_sim_seconds'],.2)

    def test_failed_plan_has_actual_samples_but_no_invented_commands_or_guard_metrics(self):
        trace=self.trace();trace['operations'][0]['plan_trace']={}
        op=analysis.extract_operation(trace,trace['operations'][0])
        self.assertEqual(op['command_q'].shape,(0,7))
        self.assertEqual(len(op['q']),3)
        self.assertIsNone(op['summary']['runtime_guard_error_max_rad'])
        self.assertIsNone(op['summary']['observed_terminal_settling_sim_seconds'])

    def test_duplicate_boundaries_use_the_last_actual_sample_and_missing_directions_stay_missing(self):
        trace=self.trace()
        trace['samples'].insert(1,copy.deepcopy(trace['samples'][0]))
        trace['operations'][0]['last_sample']=4
        self.assertEqual(len(analysis.extract_operation(trace,trace['operations'][0])['times']),3)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder/'roundtrip_trace.json').write_text(json.dumps(trace))
            result=folder/'result.json'
            result.write_text(json.dumps(dict(status='PASS',metrics=dict(mode='roundtrip',
                collision_model='enhanced',seed=123456,operations=[]))))
            case=analysis.load_case(result)
            self.assertEqual(case['summary']['missing_roundtrip_directions'],['home_return'])

    def test_missing_failed_capture_is_not_reported_as_a_motion_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            result=Path(directory)/'result.json'
            result.write_text(json.dumps(dict(status='FAIL',error='preflight failed',metrics={})))
            case=analysis.load_case(result)
            self.assertEqual(case['summary']['actual_trace_status'],'NOT_CAPTURED')
            self.assertEqual(case['operations'],[])

    def test_positive_ball_witness_distinguishes_volume_overlap_from_separation(self):
        vertices=np.array([[x,y,z] for x in (-.1,.1) for y in (-.1,.1) for z in (-.1,.1)])
        measured=analysis.intersection_ball(vertices,[.15,0.,0.],[.1,.1,.1])
        self.assertEqual(measured['status'],'POSITIVE_VOLUME_INTERSECTION')
        self.assertAlmostEqual(measured['positive_ball_radius_lower_bound_m'],.025-1e-9)
        self.assertLessEqual(measured['maximum_halfspace_feasibility_residual_m'],1e-12)
        separated=analysis.intersection_ball(vertices,[.5,0.,0.],[.1,.1,.1])
        self.assertEqual(separated['status'],'NO_POSITIVE_BALL_FOUND')

    def test_path_comparison_keeps_identical_two_waypoint_routes_visible(self):
        trace=self.trace();trace['operations'][0]['plan_trace']['route']=[[0.]*7,[.1]*7]
        op=analysis.extract_operation(trace,trace['operations'][0])
        cases=[dict(summary=dict(mode='roundtrip',model=model,seed=seed,result=model),operations=[op])
            for model,seed in (('enhanced',123456),('stock',234567))]
        measured=analysis.path_comparison(cases)
        self.assertEqual([row['route_waypoints'] for row in measured['rows']],[2,2])
        self.assertEqual(measured['rows'][1]['maximum_joint_difference_to_first_route_rad'],0.)
        self.assertEqual(measured['rows'][0]['route_joint_polyline_sha256'],
            measured['rows'][1]['route_joint_polyline_sha256'])


if __name__=='__main__':unittest.main()
