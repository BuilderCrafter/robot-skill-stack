import copy
from types import SimpleNamespace as NS
import unittest

import numpy as np

from scripts.benchmark_global_detour_v4_3 import load_inputs, match_initial, observe_planner


class DetourBenchmarkTests(unittest.TestCase):
    def test_original_keyword_tolerances_and_plan_limits_are_observed_without_changing_call(self):
        calls = []
        values = {name:np.arange(7.) for name in ('route', 'lower_limits', 'upper_limits',
            'velocity_limits', 'acceleration_limits', 'jerk_limits')}
        result = NS(**values, joint_names=('a',)*7, duration=1., diagnostics={'untouched':True})
        def plan(*args, **kwargs):
            calls.append((args, kwargs))
            return result
        planner = NS(plan=plan, _last_diagnostics={'ordinary':True})
        state = dict(planning_records=[])
        wrapped = observe_planner(lambda backend, **kw:planner, state)(None, seed=123456)
        got = wrapped.plan('pose', np.zeros(7), 'world', position_tolerance=.025, orientation_tolerance=.10)
        self.assertIs(got, result)
        self.assertEqual(calls[0][1], dict(position_tolerance=.025, orientation_tolerance=.10))
        record = state['planning_records'][0]
        self.assertEqual(record['position_tolerance_m'], .025)
        self.assertEqual(record['limits']['jerk_limits'], list(range(7)))
        self.assertEqual(record['diagnostics'], result.diagnostics)

    def test_failed_planning_is_retained_and_never_substituted(self):
        error = RuntimeError('RRT failure')
        def plan(*args, **kwargs):raise error
        planner = NS(plan=plan, _last_diagnostics={'state':'failed'})
        state = dict(planning_records=[])
        wrapped = observe_planner(lambda backend, **kw:planner, state)(None)
        with self.assertRaises(RuntimeError) as caught:
            wrapped.plan('pose', np.zeros(7), 'world', position_tolerance=.025, orientation_tolerance=.10)
        self.assertIs(caught.exception, error)
        self.assertEqual(state['planning_records'][0]['diagnostics']['state'], 'failed')

    def initial(self):
        return dict(feedback=dict(positions=list(np.arange(9.))),
            live_drive_feedback=dict(robot_applied_action={'available':True,'value':{'joint_positions':[.002]*9}}))

    def test_pair_start_requires_exact_actual_joints_names_and_latched_targets(self):
        actual = self.initial()
        target = dict(initial_state=copy.deepcopy(actual), robot_dof_names=list('abcdefghi'), named_home={'position':[1,2,3]})
        self.assertTrue(match_initial(actual, target, list('abcdefghi'), target['named_home'])['matched'])
        actual['feedback']['positions'][0] += 1e-12
        self.assertFalse(match_initial(actual, target, list('abcdefghi'), target['named_home'])['matched'])

    def test_partial_or_changed_native_target_feedback_cannot_be_an_exact_pair(self):
        actual = self.initial()
        target = dict(initial_state=copy.deepcopy(actual), robot_dof_names=list('abcdefghi'), named_home={})
        actual['live_drive_feedback']['robot_applied_action']['value']['joint_positions'][8] = .004
        self.assertFalse(match_initial(actual, target, list('abcdefghi'), {})['matched'])

    def test_iteration_and_seed_limits_cannot_be_relaxed(self):
        for seed, iterations in ((1,10000),(123456,20000)):
            args = NS(fixed_target=None,fixture=None,condition='baseline_free',seed=seed,max_iterations=iterations)
            with self.assertRaises(ValueError):load_inputs(args)

    def test_paired_free_case_cannot_silently_use_normal_ik(self):
        args = NS(fixed_target=None,fixture=None,condition='free',seed=123456,max_iterations=10000)
        with self.assertRaises(ValueError):load_inputs(args)


if __name__ == '__main__':unittest.main()
