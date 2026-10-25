"""Static compaction preserves nominal evidence and discloses missing offsets."""
import unittest

from scripts.plot_obstacle_representation_v3_response import compact


def measured_rows():
    rows = []
    for index, name in enumerate(('nominal', 'x-', 'x+', 'y-', 'y+', 'z-', 'z+')):
        rows.append(dict(case_id='actual', layout_id='A_current', scenario='static', representation='baseline',
            posture='home', snapshot=dict(revision=3), perturbation=dict(id=name, translation_m=[index*.001, 0., 0.]),
            natural_policy_force=[1., 2.], natural_policy_metric=[[2., 0.], [0., 1.]],
            canonical_acceleration_rad_s2=[0., index*.1],
            collision_force_norm=3., collision_metric_norm=4., canonical_acceleration_norm=index*.1,
            joint_acceleration_direction_cosine_to_baseline=1., collision_force_direction_cosine_to_baseline=1.))
    return rows


class ObstacleRepresentationV3ResponseTests(unittest.TestCase):
    def test_original_nominal_matrix_and_largest_offset_change_retained(self):
        rows = measured_rows()
        report = compact(dict(status='PASS', rows=rows))
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['counts'], dict(raw_queries=7, cases=1, nominal_conditions=1, sensitivity_queries=6))
        self.assertEqual(report['responses'][0]['nominal']['natural_policy_metric'], [[2., 0.], [0., 1.]])
        sensitivity = report['responses'][0]['offset_sensitivity']['canonical_acceleration_rad_s2']
        self.assertAlmostEqual(sensitivity['maximum_delta_norm'], .6)
        self.assertEqual(sensitivity['attaining_displacement']['id'], 'z+')

    def test_incomplete_sensitivity_never_becomes_complete_pass(self):
        report = compact(dict(status='PASS', rows=measured_rows()[:-1]))
        self.assertEqual(report['status'], 'PARTIAL_FAILURE')
        self.assertEqual(report['incomplete_conditions'][0]['missing_offsets'], ['z+'])

    def test_duplicate_nominal_does_not_select_a_favorable_attempt(self):
        rows = measured_rows()
        with self.assertRaises(ValueError):
            compact(dict(status='PASS', rows=rows+[rows[0]]))

    def test_failed_identical_geometry_control_is_retained(self):
        report = compact(dict(status='PASS', rows=measured_rows(),
            identical_geometry_negative_control=dict(status='FAIL', checks=[dict(case_id='actual',
                representation='axis_best', posture='home', perturbation='nominal', matched=False,
                maximum_absolute_differences=dict(natural_policy_force=.1))])))
        self.assertEqual(report['status'], 'PARTIAL_FAILURE')
        self.assertEqual(len(report['identical_geometry_negative_control']['failed_comparisons']), 1)


if __name__ == '__main__':
    unittest.main()
