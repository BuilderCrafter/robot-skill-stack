"""Fixed posture declarations and observational scope stay separate from goals."""
import unittest

from scripts import capture_collision_postures_v4_2 as capture


class PostureCaptureTests(unittest.TestCase):
    def test_canonical_goals_are_predeclared_and_include_actual_low_twenty_centimeters(self):
        self.assertEqual(capture.POSTURES, (
            ('Extended', (.65, 0., .35)), ('Raised', (.45, -.1, .65)), ('Low', (.45, -.1, .20))))
        self.assertEqual(capture.DOWN, (0., 0., 1., 0.))

    def sample(self, measured=(.0023,.0023), target=(.0023,.0023)):
        # Deliberately reordered names establish named rather than positional checking.
        return dict(joint_feedback=dict(positions=[measured[1],0.,measured[0]]),
            live_drive_feedback=dict(robot_applied_action=dict(available=True,
                value=dict(joint_positions=[target[1],0.,target[0]], joint_indices=None))))

    def names(self):
        return ['panda_finger_joint2', 'panda_joint1', 'panda_finger_joint1']

    def test_actual_and_latched_named_fingers_must_both_fit_enhanced_scope(self):
        self.assertEqual(capture.gripper_scope([self.sample()], self.names())['classification'],
            'WITHIN_ENHANCED_GRIPPER_CONTRACT')
        for sample in (self.sample(measured=(.006,.002)), self.sample(target=(.002,.006))):
            measured = capture.gripper_scope([sample], self.names())
            self.assertEqual(measured['classification'], 'OUTSIDE_ENHANCED_GRIPPER_CONTRACT')
            self.assertIn('retained', measured['interpretation'])

    def test_missing_targets_and_invalid_encoders_are_not_full_scope_passes(self):
        row = self.sample()
        row['live_drive_feedback'] = {}
        self.assertEqual(capture.gripper_scope([row], self.names())['classification'],
            'MEASURED_WITHIN_CONTRACT_TARGETS_UNAVAILABLE')
        row['joint_feedback']['positions'][0] = float('nan')
        self.assertEqual(capture.gripper_scope([row], self.names())['classification'],
            'INVALID_GRIPPER_FEEDBACK')

    def test_partial_arm_targets_cannot_masquerade_as_measured_finger_targets(self):
        row = self.sample()
        row['live_drive_feedback']['robot_applied_action']['value'] = dict(
            joint_positions=[1.], joint_indices=[1])
        scope = capture.gripper_scope([row], self.names())
        self.assertEqual(scope['classification'], 'MEASURED_WITHIN_CONTRACT_TARGETS_UNAVAILABLE')
        self.assertIsNone(scope['latched_target_minimum_m'])

    def test_failed_goal_retains_valid_measurement_but_overall_outcome_is_fail(self):
        row = dict(returned_success=True, measured_full_pose_within_tolerances=True,
            clearance=dict(minimum_m=.01), observed_physical_contact_steps=0,
            maximum_actual_joint_displacement_rad=.3)
        operations = [dict(row, label=label) for label in ('Extended','Raised','Low','Home_return')]
        checks = capture.goal_checks(operations)
        self.assertEqual(capture.outcome(True,checks)['status'], 'PASS')
        operations[2]['returned_success'] = False
        checks = capture.goal_checks(operations)
        self.assertFalse(checks['Low'])
        outcome = capture.outcome(True,checks)
        self.assertEqual(outcome['measurement_status'], 'PASS')
        self.assertEqual(outcome['status'], 'FAIL')
        self.assertFalse(outcome['posture_acceptance'])
        operations[2]['returned_success'] = True
        operations[2]['observed_physical_contact_steps'] = 1
        self.assertFalse(capture.goal_checks(operations)['Low'])


if __name__ == '__main__':
    unittest.main()
