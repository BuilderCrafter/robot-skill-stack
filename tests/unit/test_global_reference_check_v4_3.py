"""No missing endpoint or clear-model obstruction can become a PASS receipt."""
from copy import deepcopy
import unittest

import numpy as np

from scripts.check_global_reference_v4_3 import evaluate_reference, selected_proof, verify_control_override


class GlobalReferenceCheckTests(unittest.TestCase):
    def setUp(self):
        self.route = np.zeros((3, 7))
        self.route[:, 0] = [0., .5, 1.]
        self.endpoints = dict(start=self.route[0].copy(), goal=self.route[-1].copy())

    def query(self, condition, states):
        return [dict(external_collision=bool(condition == 'obstacle' and .4 < q[0] < .6
            or condition == 'goal_blocked' and q[0] > .9), unmasked_self_pairs=[], domain_error=None)
            for q in states]

    def evaluate(self, query=None, endpoints=None, index=1):
        return evaluate_reference(self.route, self.endpoints if endpoints is None else endpoints,
            index, query or self.query)

    def test_only_candidate_invalidates_original_reference_and_removal_restores_it(self):
        summary, records = self.evaluate()
        self.assertEqual(summary['status'], 'PASS')
        self.assertTrue(all(summary['checks'].values()))
        self.assertEqual(summary['conditions']['obstacle']['trajectory_external_collision_samples'], [1])
        self.assertEqual(summary['conditions']['free']['trajectory_external_collision_samples'], [])
        self.assertEqual(records['free'], records['removed'])

    def test_both_explicit_endpoints_are_mandatory(self):
        calls = []
        with self.assertRaisesRegex(ValueError, 'both named endpoints'):
            self.evaluate(lambda *args:calls.append(args), endpoints={'start':self.route[0]})
        self.assertEqual(calls, [])

    def test_collision_at_goal_cannot_be_called_detour_feasibility(self):
        def query(condition, states):
            rows = self.query(condition, states)
            if condition == 'obstacle':
                rows[-1]['external_collision'] = True
            return rows
        summary, _ = self.evaluate(query)
        self.assertEqual(summary['status'], 'FAIL')
        self.assertFalse(summary['checks']['candidate_goal_clear'])

    def test_physical_obstruction_not_seen_by_model_is_explicit_failure(self):
        def query(condition, states):
            rows = self.query(condition, states)
            if condition == 'obstacle':
                for row in rows:row['external_collision'] = False
            return rows
        summary, _ = self.evaluate(query)
        self.assertEqual(summary['status'], 'FAIL')
        self.assertFalse(summary['checks']['candidate_obstructs_declared_physical_intersection_state'])

    def test_original_world_or_irrelevant_control_collision_cannot_pass(self):
        for failing_condition in ('free', 'near', 'removed'):
            with self.subTest(condition=failing_condition):
                def query(condition, states):
                    rows = self.query(condition, states)
                    if condition == failing_condition:rows[1]['external_collision'] = True
                    return rows
                summary, _ = self.evaluate(query)
                self.assertEqual(summary['status'], 'FAIL')

    def test_self_or_domain_failures_at_endpoint_are_not_ignored(self):
        for key, value in (('unmasked_self_pairs', [['finger', 'forearm']]), ('domain_error', 'outside')):
            with self.subTest(key=key):
                def query(condition, states):
                    rows = self.query(condition, states)
                    if condition == 'obstacle':rows[-2][key] = value
                    return rows
                summary, _ = self.evaluate(query)
                self.assertEqual(summary['status'], 'FAIL')
                self.assertFalse(summary['checks']['candidate_start_clear'])

    def test_missing_query_rows_or_unbound_reference_index_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete native'):
            self.evaluate(lambda condition, states:self.query(condition, states)[:-1])
        for index in (-1, 3, True, .5):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'obstruction index'):
                self.evaluate(index=index)

    def test_original_reference_must_connect_the_fixed_endpoints(self):
        self.endpoints['goal'][0] = .9
        with self.assertRaisesRegex(ValueError, 'endpoints differ'):
            self.evaluate()

    def test_original_and_exact_proof_formats_bind_the_same_frozen_box_and_state(self):
        proof = dict(baseline_sample=1, planned_time=.25, positive_ball_radius_lower_bound_m=.005)
        fixture = dict(fixture=dict(position_m=[.5, 0., .3], dimensions_m=[.04, .04, .08]),
            obstruction_proof=proof, producer_sha256='frozen')
        original = dict(selected_candidate=dict(center_m=[.5, 0., .3], dimensions_m=[.04, .04, .08]),
            planned_obstruction=proof)
        selected_proof(fixture, original)
        fixture['algorithm_revision'] = 2
        exact = dict(selected_candidate=dict(center=[.5, 0., .3], dimensions=[.04, .04, .08], planned=proof),
            producer_sha256='frozen', candidate_RRT_calls=0)
        selected_proof(fixture, exact)

    def test_exact_proof_cannot_rebind_box_producer_or_candidate_search_history(self):
        proof = dict(baseline_sample=1, planned_time=.25)
        fixture = dict(algorithm_revision=2, producer_sha256='frozen', obstruction_proof=proof,
            fixture=dict(position_m=[.5, 0., .3], dimensions_m=[.04, .04, .08]))
        def raw():
            return dict(selected_candidate=dict(center=[.5, 0., .3], dimensions=[.04, .04, .08], planned=proof.copy()),
                producer_sha256='frozen', candidate_RRT_calls=0)
        for mutate in (lambda value:value.update(producer_sha256='changed'),
                       lambda value:value.update(candidate_RRT_calls=1),
                       lambda value:value['selected_candidate'].update(center=[.6, 0., .3]),
                       lambda value:value['selected_candidate']['planned'].update(baseline_sample=2)):
            value = raw()
            mutate(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                selected_proof(fixture, value)

    def control_chain(self):
        key = 'negative_controls.goal_blocked.dimensions_m'
        geometry = dict(position_m=[.55, -.225, .3], dimensions_m=[.06, .06, .12], orientation_wxyz=[1., 0., 0., 0.])
        world_target = [.45, -.25, .35]
        artifact = dict(path='numeric_witness.npz', sha256='source-bound')
        lock = dict(locked_main_fixture=deepcopy(geometry), planned_obstruction={'state':35},
            actual_obstruction={'step':57}, raw_witness=artifact.copy())
        design = dict(negative_controls={
            'near':dict(position_m=[.72, -.25, .38], dimensions_m=[.09, .14, .25], orientation_wxyz=[1., 0., 0., 0.]),
            'goal_blocked':dict(position_m='Fixed target', dimensions_m=[.18]*3, orientation_wxyz=[1., 0., 0., 0.])})
        fixture = dict(fixture=geometry, obstruction_proof={'state':35}, actual_obstruction_proof={'step':57},
            witness=dict(raw_artifact=artifact.copy()), baseline=dict(target_pose=dict(position=world_target)),
            controls={'near':deepcopy(design['negative_controls']['near']),
                'goal_blocked':dict(position_m=world_target.copy(), dimensions_m=[.08]*3, orientation_wxyz=[1., 0., 0., 0.])},
            protocol_overrides={key:dict(original=[.18]*3, active=[.08]*3)}, finalization_producer_sha256='control-source')
        declaration = dict(permitted_protocol_overrides=[key], candidate_RRT_calls=0,
            main_selection_rerun_allowed=False, main_geometry_must_remain_identical=True,
            goal_blocked_control=dict(candidate_cube_edge_lengths_m=[.08, .06, .04]),
            unchanged_criteria=dict(minimum_actual_home_clearance_m=.01, minimum_goal_intersection_ball_radius_m=.003))
        proof = dict(producer_sha256='control-source', candidate_RRT_calls=0, selected_cube_edge_length_m=.08,
            records=[dict(cube_edge_length_m=.08, actual_home_clearance_m=.07, enhanced_home_clear=True,
                enhanced_goal_blocked=True, goal_intersection_radius_m=.01)])
        return fixture, design, declaration, lock, proof

    def test_control_only_finalization_preserves_main_box_and_physical_witness(self):
        verify_control_override(*self.control_chain())

    def test_finalization_cannot_change_main_near_or_predeclared_selection_rules(self):
        for mutate in (
                lambda fixture, design, declaration, lock, proof:fixture['fixture'].update(position_m=[.6, -.225, .3]),
                lambda fixture, design, declaration, lock, proof:fixture['controls']['near'].update(dimensions_m=[.2]*3),
                lambda fixture, design, declaration, lock, proof:fixture['protocol_overrides'].update(extra='forbidden'),
                lambda fixture, design, declaration, lock, proof:proof.update(candidate_RRT_calls=1),
                lambda fixture, design, declaration, lock, proof:proof['records'][0].update(actual_home_clearance_m=.001)):
            chain = self.control_chain()
            mutate(*chain)
            with self.subTest(chain=chain), self.assertRaises(ValueError):
                verify_control_override(*chain)


if __name__ == '__main__':
    unittest.main()
