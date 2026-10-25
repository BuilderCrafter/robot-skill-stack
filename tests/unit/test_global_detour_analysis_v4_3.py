"""Causal route comparison must not confuse IK changes or timing with detours."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts import analyze_global_detour_v4_3 as analysis


class GlobalDetourAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.start = [0.]*7
        self.goal = [1., 0., 0., 0., 0., 0., 0.]
        self.direct = [self.start, self.goal]
        self.detour = [self.start, [.5, .2, 0., 0., 0., 0., 0.], self.goal]
        self.pose = dict(position=[.45, -.25, .35], orientation=[0., 0., 1., 0.], frame='world')
        self.fixture = dict(prim_path='/FrozenBox', position_m=[.5, 0., .3], dimensions_m=[.04, .04, .12], orientation_wxyz=[1., 0., 0., 0.])
        self.world_box = dict(path='/FrozenBox', kind='cuboid', position=[.5, 0., .3], dimensions=[.04, .04, .12], orientation=[1., 0., 0., 0.])
        self.frozen = dict(fixture=self.fixture, baseline=dict(result='baseline.json', start_q=self.start,
            goal_q=self.goal, target_pose=self.pose, source_sha256={'baseline.json': 'baseline-hash'}),
            obstruction_proof=dict(status='VERIFIED_POSITIVE_INTERSECTION_BALL', reference='independentFKplannedtrajectory',
                positive_ball_radius_lower_bound_m=.005),
            witness=dict(independent_of_candidate_rrt=True, numericcontinuousexternal_lower_bound_m=.02, self_lower_bound_m=.004))

    def operation(self, label, route):
        return dict(route=np.asarray(route, float), summary=dict(label=label, returned_success=True,
            maximum_actual_arm_displacement_rad=.5, sampled_all_scene_minimum_m=.02,
            observed_physical_contact_steps=0, final_position_error_m=.001, final_orientation_error_rad=.002,
            final_measured_arm_velocity_max_rad_s=.005, solver_position_iterations_restored=True,
            observed_terminal_settling_sim_seconds=.25, sampled_fixture_minimum_m=.03))

    def case(self, condition, route=None):
        route = self.direct if route is None else route
        obstacles = [dict(path='/Floor')]
        if condition == 'obstacle':
            obstacles.append(self.world_box)
        controls = dict(start_q=self.start.copy(), measured_fingers=[.0023, .0023],
            native_finger_targets=dict(zip(analysis.FINGERS, [.0023, .0023])), fixed_q=self.goal.copy(),
            selected_ik_q=self.goal.copy(), full_pose=copy.deepcopy(self.pose), model='enhanced',
            model_sha256=analysis.ENHANCED_MODEL_SHA256, max_iterations=10000, rrt_step_size_rad=.05,
            planning_controls=dict(limits=[1.]*7), execution_controls={key:.1 for key in analysis.EXECUTION_KEYS},
            robot_base_pose=dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.]),
            world_snapshot=dict(timestamp=1. if condition == 'free' else 2., obstacles=obstacles, meters_per_unit=1.),
            selected_fk_errors=dict(fk_position_error=.0001, fk_orientation_error=.001),
            planning_validation=dict(route_samples=10, trajectory=dict(collision_validation='PASS_SAMPLED_SDK_MODEL')),
            fixed_adapter=dict(production_validation_preserved=True))
        return dict(summary=dict(result=condition+'.json', condition=condition, seed=123456, repeat='r1',
            source_manifest_sha256='shared-source', fixture_sha256='fixture-sha', measurement_valid=True,
            outward_physical_acceptance=True, home_physical_acceptance=condition == 'obstacle'),
            controls=controls, operations=[self.operation('outward', route), self.operation('home_return', list(reversed(route)))])

    def baseline(self):
        case = self.case('baseline_free')
        case['summary'].update(result='baseline.json', result_sha256='baseline-hash')
        return case

    def test_geometry_measures_all_seven_joints_and_actual_polyline_length(self):
        route = copy.deepcopy(self.detour)
        route[1][1], route[1][6] = 0., .2
        measured = analysis.route_geometry(route)
        self.assertAlmostEqual(measured['maximum_deviation_from_direct_rad'], .2)
        self.assertAlmostEqual(measured['joint_space_length_rad'], 2*np.sqrt(.5**2+.2**2))
        self.assertEqual(measured['waypoints'], 3)

    def test_route_coordinates_use_explicit_joint_names(self):
        names = list(reversed(analysis.ARM))
        route = [[.7, .6, .5, .4, .3, .2, .1], [.8, .7, .6, .5, .4, .3, .2]]
        np.testing.assert_allclose(analysis.named_route(route, names)[0], [.1, .2, .3, .4, .5, .6, .7])
        with self.assertRaises(ValueError):
            analysis.named_route(route, [analysis.ARM[0]]*7)

    def test_extra_collinear_waypoint_and_duration_changes_are_not_a_detour(self):
        collinear = [self.start, [.5, 0., 0., 0., 0., 0., 0.], self.goal]
        compared = analysis.compare_routes(self.direct, collinear)
        self.assertFalse(compared['exact_waypoint_array_equal'])
        self.assertEqual(compared['sampled_symmetric_route_distance_rad'], 0.)
        self.assertFalse(compared['meaningful_multisegment_detour'])

    def test_duplicate_route_vertices_and_zero_length_connections_remain_finite(self):
        route = [self.start, self.start, self.goal]
        self.assertEqual(analysis.route_geometry(route)['joint_space_length_rad'], 1.)
        self.assertEqual(analysis.distances_to_polyline([self.goal], [self.start, self.start])[0], 1.)
        self.assertEqual(analysis.sample_polyline([self.start]).shape, (1, 7))
        with self.assertRaises(ValueError):
            analysis.route_geometry([[0.]*6])
        with self.assertRaises(ValueError):
            analysis.sample_polyline(self.direct, step_rad=float('nan'))

    def test_changed_ik_target_cannot_be_credited_to_an_obstacle(self):
        free, obstacle = self.case('free'), self.case('obstacle', self.detour)
        obstacle['controls']['fixed_q'] = [.9, 0., 0., 0., 0., 0., 0.]
        obstacle['controls']['selected_ik_q'] = obstacle['controls']['fixed_q']
        checks = analysis.match_controls(free, obstacle, fixture_path='/FrozenBox')
        self.assertFalse(checks['fixed_q'])
        self.assertFalse(checks['unchanged_fixed_goal'])

    def test_only_frozen_box_may_change_between_planning_worlds(self):
        free, obstacle = self.case('free'), self.case('obstacle', self.detour)
        checks = analysis.match_controls(free, obstacle, fixture_path='/FrozenBox')
        self.assertTrue(all(checks.values()), checks)
        obstacle['controls']['world_snapshot']['obstacles'][0] = dict(path='/DifferentFloor')
        checks = analysis.match_controls(free, obstacle, fixture_path='/FrozenBox')
        self.assertFalse(checks['only_frozen_obstacle_added_to_world'])

    def test_missing_controls_and_unsupported_finger_targets_cannot_pass(self):
        free, obstacle = self.case('free'), self.case('obstacle', self.detour)
        free['controls']['planning_controls'] = None
        obstacle['controls']['native_finger_targets'][analysis.FINGERS[0]] = .04
        checks = analysis.match_controls(free, obstacle, fixture_path='/FrozenBox')
        self.assertFalse(checks['planning_controls'])
        self.assertFalse(checks['native_finger_targets_supported'])

    def test_valid_numeric_detour_requires_independent_proof_and_observed_home(self):
        cases = [self.baseline(), self.case('free'), self.case('obstacle', self.detour)]
        proof = analysis.design_checks(self.frozen, cases)
        self.assertTrue(all(proof.values()), proof)
        paired = analysis.paired_results(cases, proof, self.frozen, 'fixture-sha')
        self.assertTrue(paired[0]['strong_detour_accepted'], paired)
        cases[-1]['summary']['home_physical_acceptance'] = False
        paired = analysis.paired_results(cases, proof, self.frozen, 'fixture-sha')
        self.assertFalse(paired[0]['strong_detour_accepted'])
        self.assertFalse(paired[0]['controls_and_acceptance']['obstacle_home_physical_acceptance'])

    def test_actual_tcp_or_surface_tangency_does_not_prove_original_global_route_obstruction(self):
        cases = [self.baseline(), self.case('free'), self.case('obstacle', self.detour)]
        frozen = copy.deepcopy(self.frozen)
        frozen['obstruction_proof']['reference'] = 'actualTCPcurve'
        self.assertFalse(analysis.design_checks(frozen, cases)['original_global_route_positive_volume_obstruction'])
        frozen['obstruction_proof']['reference'] = 'independentFKplannedtrajectory'
        frozen['obstruction_proof']['positive_ball_radius_lower_bound_m'] = 0.
        self.assertFalse(analysis.design_checks(frozen, cases)['original_global_route_positive_volume_obstruction'])

    def test_a_changed_free_reference_cannot_reuse_the_original_obstruction_proof(self):
        cases = [self.baseline(), self.case('free', self.detour), self.case('obstacle',
            [self.start, [.5, .3, 0., 0., 0., 0., 0.], self.goal])]
        paired = analysis.paired_results(cases, analysis.design_checks(self.frozen, cases), self.frozen, 'fixture-sha')
        self.assertFalse(paired[0]['controls_and_acceptance']['free_restores_original_global_route'])
        self.assertFalse(paired[0]['strong_detour_accepted'])

    def test_missing_or_failed_pair_remains_visible(self):
        cases = [self.baseline(), self.case('free')]
        summary = analysis.summarize(cases, self.frozen, 'fixture-sha')
        self.assertEqual(summary['matched_pairs'][0]['status'], 'MISSING_OR_AMBIGUOUS_PAIR')
        self.assertFalse(summary['detour_demonstrated'])
        self.assertNotEqual(summary['status'], 'PASS_DECLARED_STATIC_DETOUR')

    def test_clean_native_rejection_alone_does_not_prove_a_physically_blocked_goal(self):
        blocked = self.case('goal_blocked')
        blocked['operations'] = [self.operation('outward', self.direct)]
        blocked['operations'][0]['raw_operation'] = dict(returned_success=False,
            strategy_diagnostics=dict(motion_strategy='global', failure_code='invalid_goal'))
        blocked['raw'] = dict(metrics=dict(operations=[dict(label='outward', action_count=0)]))
        frozen = copy.deepcopy(self.frozen)
        self.assertFalse(analysis.negative_controls([blocked], frozen)['goal_blocked_passed'])
        frozen['control_proofs'] = dict(goal_blocked=dict(passed=True,
            goal_positive_volume_intersection=dict(status='VERIFIED_POSITIVE_INTERSECTION_BALL', positive_ball_radius_lower_bound_m=.01),
            endpoints=dict(actual_start_home=dict(minimum_m=.04))))
        measured = analysis.negative_controls([blocked], frozen)
        self.assertTrue(measured['goal_blocked_passed'], measured)
        frozen['control_proofs']['goal_blocked']['goal_positive_volume_intersection']['positive_ball_radius_lower_bound_m'] = 0.
        self.assertFalse(analysis.negative_controls([blocked], frozen)['goal_blocked_passed'])

    def test_no_result_and_malformed_result_are_not_successful_measurements(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'result.json'
            case = analysis.read_case(path, dict(condition='obstacle', seed=123456))
            self.assertEqual(case['summary']['status'], 'NOT_RUN')
            self.assertFalse(case['summary']['measurement_valid'])
            path.write_text('{bad')
            case = analysis.read_case(path)
            self.assertEqual(case['summary']['status'], 'INVALID_CAPTURE')

    def test_nonfinite_actual_error_cannot_pass_physical_acceptance(self):
        op = self.operation('outward', self.direct)
        self.assertTrue(analysis._operation_pass(op))
        op['summary']['final_position_error_m'] = float('nan')
        self.assertFalse(analysis._operation_pass(op))

    def test_job_inventory_includes_missing_declared_result(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'jobs.json'
            jobs = [dict(result='outputs/test/obstacle/r1/result.json', label='blocked r1',
                artifacts_dir='outputs/test/obstacle/r1', arguments=['script.py', '--condition', 'obstacle', '--seed', '234567'])]
            path.write_text(json.dumps(jobs))
            record = analysis.declarations([path])['outputs/test/obstacle/r1/result.json']
            self.assertEqual(record['seed'], 234567)
            self.assertEqual(record['condition'], 'obstacle')


if __name__ == '__main__':
    unittest.main()
