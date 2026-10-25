"""Static-query inputs retain actual geometry provenance and matched conditions."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.probe_obstacle_representation_v3 import (
    choose_cases, export_postures, negative_controls, perturbations, recorded_base,
    selected_cell_union, validate_packet)


class ObstacleRepresentationV3ProbeTests(unittest.TestCase):
    def test_recorded_postures_export_only_declared_q_and_static_velocity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'capture.json'
            path.write_text(json.dumps(dict(snapshots=[dict(posture=name, joints=[float(i) for i in range(9)])
                for name in ('home', 'raised', 'low')])))
            result = export_postures(path)
        self.assertEqual([row['id'] for row in result['postures']], ['home', 'raised', 'low'])
        self.assertEqual(result['postures'][0]['q'], [float(i) for i in range(7)])
        self.assertEqual(result['postures'][0]['qd'], [0.]*7)
        self.assertIn('not asserted exposure-matched', result['scope'])

    def test_missing_explicit_packet_case_is_not_silently_replaced(self):
        bundle = dict(cases=[dict(case_id='first'), dict(case_id='second')])
        self.assertEqual(choose_cases(bundle, maximum=1), [dict(case_id='first')])
        with self.assertRaises(ValueError):
            choose_cases(bundle, case_ids=['missing'])

    def test_expired_or_invalid_numeric_geometry_is_rejected(self):
        box = dict(id='cell', center=[0., 0., 0.], dimensions=[.01]*3, expires_at=11.)
        self.assertEqual(validate_packet(dict(boxes=[box]), evaluated_at=10.), [box])
        with self.assertRaises(ValueError):
            validate_packet(dict(boxes=[box]), evaluated_at=11.)
        with self.assertRaises(ValueError):
            validate_packet(dict(boxes=[box, box]))

    def test_geometry_sensitivity_offsets_are_small_symmetric_and_bounded(self):
        values = perturbations(1.)
        self.assertEqual(len(values), 7)
        self.assertEqual(values[0]['translation_m'], [0., 0., 0.])
        self.assertEqual(values[1]['translation_m'][0], -.001)
        self.assertEqual(values[2]['translation_m'][0], .001)
        with self.assertRaises(ValueError):
            perturbations(10.)

    def test_standalone_policy_uses_recorded_base_and_rejects_motion(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            result = folder/'result.json'
            result.write_text(json.dumps(dict(metrics=dict(capture=dict(body_paths=['/Franka/panda_link0'])))))
            trace = folder/'trace.json'
            first = [1., 2., 3., 0., 0., 0., 1.]
            trace.write_text(json.dumps(dict(samples=[dict(poses=[first]), dict(poses=[first])])) )
            base = recorded_base(result)
            self.assertEqual(base['translation_world_m'], [1., 2., 3.])
            self.assertEqual(base['world_from_base_rotation'], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            changed = [1.1, 2., 3., 0., 0., 0., 1.]
            trace.write_text(json.dumps(dict(samples=[dict(poses=[first]), dict(poses=[changed])])) )
            with self.assertRaises(ValueError):
                recorded_base(result)

    def test_equal_selected_union_detected_across_different_cuboid_partitions(self):
        snapshot = dict(voxel_size=.01, bounds_min=[0., 0., 0.])
        merged = dict(boxes=[dict(center=[.01, .005, .005], dimensions=[.02, .01, .01])])
        split = dict(boxes=[dict(center=[x, .005, .005], dimensions=[.01, .01, .01]) for x in (.005, .015)])
        self.assertEqual(selected_cell_union(merged, snapshot), selected_cell_union(split, snapshot))
        self.assertEqual(len(selected_cell_union(merged, snapshot)), 2)
        split['boxes'][0]['center'][0] += .001
        with self.assertRaises(ValueError):
            selected_cell_union(split, snapshot)

    def test_identical_geometry_negative_control_checks_actual_policy_outputs(self):
        common = dict(case_id='actual', posture='home', perturbation=dict(id='nominal'),
            natural_policy_force=[1., 2.], natural_policy_metric=[[1., 0.], [0., 1.]],
            canonical_acceleration_rad_s2=[.2, .3])
        rows = [dict(common, representation='baseline'), dict(common, representation='axis_best')]
        unions = [dict(case_id='actual', representation='axis_best', identical_numeric_cuboids_as_baseline=True)]
        self.assertEqual(negative_controls(rows, unions)['status'], 'PASS')
        rows[1]['canonical_acceleration_rad_s2'] = [.2, .4]
        report = negative_controls(rows, unions)
        self.assertEqual(report['status'], 'FAIL')
        self.assertAlmostEqual(report['checks'][0]['maximum_absolute_differences']['canonical_acceleration_rad_s2'], .1)


if __name__ == '__main__':
    unittest.main()
