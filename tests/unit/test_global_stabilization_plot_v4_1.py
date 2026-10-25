import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analyze_global_stabilization_v4_1 import ARM, digest
from scripts.plot_global_stabilization_v4_1 import extract_case, extract_hold_case, quaternion_distance


class RecordedPlotTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        samples = []
        for time, q1 in ((10., .2), (10.1, .1), (10.2, -.3)):
            samples.append(dict(simulation_time=time, feedback=dict(positions=[q1]+[0.]*8),
                tcp_pose=dict(position=[.1, 0., 0.], orientation=[1., 0., 0., 0.])))
        actions = [dict(simulation_time=10., requested_positions=[.3]+[0.]*6,
            requested_indices=list(range(7)), actual_physx_targets=dict(available=True,
                value=dict(joint_positions=[.3]+[0.]*8, joint_indices=None)))]
        trace = dict(metadata=dict(robot_dof_names=[*ARM, 'finger1', 'finger2']),
            samples=samples, actions=actions, operations=[dict(label='home', first_sample=0,
                last_sample=3, first_action=0, last_action=1, returned_success=False,
                target=dict(position=[0., 0., 0.], orientation=[-1., 0., 0., 0.]),
                strategy_diagnostics=dict(failure_code='tracking_failure', tracking_error_rad=.42,
                    tracking_limit_rad=.15, tracking_abort_rad=.4),
                plan_trace=dict(joint_names=list(ARM), trajectory=[
                    dict(time=0., q=[.3]+[0.]*6), dict(time=.3, q=[.6]+[0.]*6)],
                    actual=[dict(sim_time=10., q=[.2]+[0.]*6, tracking_error_rad=.1),
                            dict(sim_time=10.1, q=[.1]+[0.]*6, tracking_error_rad=.3)],
                    # Missing second same-time command deliberately: never
                    # synthesize an exact desired error from the plan grid.
                    commanded=[dict(sim_time=10., q=[.3]+[0.]*6)]))])
        trace_path = root/'controlled_trace.json'
        trace_path.write_text(json.dumps(trace))
        result_path = root/'result.json'
        result_path.write_text(json.dumps(dict(controlled=dict(mode='C', strategy='global',
            raw_trace=dict(path=str(trace_path), sha256=digest(trace_path))))))
        return result_path, trace_path

    def test_extract_preserves_simulation_time_exact_error_and_terminal_abort(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = self.fixture(folder)
            case = extract_case(path)
        self.assertTrue(np.allclose(case['sample_time'], [0., .1, .2]))
        self.assertTrue(np.array_equal(case['measured_q1'], [.2, .1, -.3]))
        self.assertTrue(np.allclose(case['exact_time'], [0.]))
        self.assertTrue(np.allclose(case['exact_q1_error'], [.1]))
        self.assertTrue(np.allclose(case['guard_error'], [.1, .3]))
        self.assertEqual(case['abort_error'], .42)
        self.assertTrue(np.allclose(case['tcp_position'], [.1]*3))
        self.assertTrue(np.allclose(case['tcp_orientation'], [0.]*3))
        self.assertTrue(np.allclose(case['target_gap'], [0.]))
        self.assertFalse(case['returned_success'])

    def test_modified_trace_is_rejected_before_plotting(self):
        with tempfile.TemporaryDirectory() as folder:
            path, trace = self.fixture(folder)
            trace.write_text(trace.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                extract_case(path)

    def test_unknown_operation_is_explicitly_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = self.fixture(folder)
            with self.assertRaisesRegex(ValueError, 'Exactly one'):
                extract_case(path, operation='invented_after')

    def test_quaternion_angle_is_sign_invariant_and_rejects_bad_input(self):
        self.assertAlmostEqual(quaternion_distance([0., 1., 0., 0.], [0., -2., 0., 0.]), 0.)
        self.assertAlmostEqual(quaternion_distance([1., 0., 0., 0.], [0., 1., 0., 0.]), np.pi)
        with self.assertRaises(ValueError):
            quaternion_distance([0.]*4, [1., 0., 0., 0.])

    def test_hold_windows_preserve_measured_count_switch_and_do_not_invent_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            path, trace_path = self.fixture(folder)
            trace = json.loads(trace_path.read_text())
            home = trace['operations'][0]
            home['after'] = dict(simulation_time=10.2)
            for time, count in ((10.2, 255), (11.2, 255), (11.2, 32), (12.2, 32)):
                trace['samples'].append(dict(simulation_time=time,
                    feedback=dict(velocities=[.02]*9), actual_minus_applied_target_arm_rad=[.03]*7,
                    tcp_pose=dict(position=[.001, 0., 0.], orientation=[1., 0., 0., 0.]),
                    native_actor_solver_properties=dict(available=True, position=count, velocity=1)))
            for label, first, last in (('retained_home_hold', 3, 5), ('post_home_hold', 5, 7)):
                trace['operations'].append(dict(label=label, kind='post_hold', first_sample=first,
                    last_sample=last, first_action=1, last_action=1,
                    before=dict(simulation_time=trace['samples'][first]['simulation_time']),
                    after=dict(simulation_time=trace['samples'][last-1]['simulation_time'])))
            trace_path.write_text(json.dumps(trace))
            result = json.loads(path.read_text());result['controlled']['raw_trace']['sha256'] = digest(trace_path)
            path.write_text(json.dumps(result))
            case = extract_hold_case(path)
        self.assertTrue(np.allclose(case['time'], [0., 1., 1., 2.]))
        self.assertEqual(case['native_position_count'].tolist(), [255, 255, 32, 32])
        self.assertTrue(all(row['additional_actions'] == 0 for row in case['windows']))
        self.assertTrue(np.allclose(case['position'], [.001]*4))
        self.assertTrue(np.allclose(case['speed'], [.02]*4))


if __name__ == '__main__':
    unittest.main()
