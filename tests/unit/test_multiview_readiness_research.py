"""Harness correctness, with the real unchanged grasp planner and no simulator."""
import unittest
import json

import numpy as np

from robot_skill_stack.world.model.world_model import WorldModel
from tests.support import multiview_iteration2 as previous
from tests.support.multiview_readiness_research import (
    EYES, SEEDS, QueryOnlyBackend, evaluate_grasp, grasp_width_scenes, invalid_geometry_queries,
    make_planner, pick_start_guard, readiness_holdout_scenes, sphere, truth_world_object,
)


class ReadinessResearchTests(unittest.TestCase):
    def test_all_fixture_planner_rows_and_numpy_nested_evidence_serialize(self):
        from scripts.benchmark_multiview_grasp_readiness import plain_values
        for scene in grasp_width_scenes():
            model = WorldModel()
            for truth in scene.objects:
                model.register(truth_world_object(truth))
            for truth in scene.objects:
                row = evaluate_grasp(truth, model.require(truth.name), model, scene.objects)
                row['nested_numerical_evidence'] = dict(valid=np.bool_(True), count=np.int64(4),
                                                        bounds=np.array([.01, .02]))
                encoded = json.dumps(plain_values(row))
                self.assertTrue(json.loads(encoded)['nested_numerical_evidence']['valid'])

    def test_final_holdout_is_deterministic_and_independent(self):
        first, second = readiness_holdout_scenes(), readiness_holdout_scenes()
        self.assertEqual(len(first), 27)
        self.assertEqual([scene.seed for scene in first], [scene.seed for scene in second])
        self.assertTrue(set(SEEDS.values()).isdisjoint(previous.SEEDS.values()))
        self.assertTrue(set(EYES.values()).isdisjoint(previous.EYES.values()))
        names = [obj.name for scene in first if scene.gap_mm is not None for obj in scene.objects]
        self.assertEqual(len(names), len(set(names)))
        for scene in first:
            if scene.name.endswith('sphere_sphere'):
                a, b = scene.objects
                gap = np.linalg.norm(a.center-b.center)-a.radius-b.radius
                self.assertAlmostEqual(gap*1000, scene.gap_mm, places=8)

    def test_ground_truth_uses_actual_planner_limits_instead_of_backend_width_only(self):
        planner = make_planner(WorldModel())
        scenes = grasp_width_scenes()
        for kind, width, feasible in (('box', '69', True), ('box', '70', True), ('box', '71', False),
                                       ('sphere', '60', True), ('sphere', '62', True), ('sphere', '64', False)):
            scene = next(scene for scene in scenes if scene.name == f'readiness_selection_{kind}_{width}')
            self.assertEqual(planner.plan(truth_world_object(scene.objects[0])).ok, feasible, scene.name)

    def test_underestimated_sphere_is_counted_as_false_feasible(self):
        truth = sphere('real_64mm', (.45, 0.), .032)
        observed = truth_world_object(truth, 'measured')
        observed.geometry.radius = .030
        observed.size = np.full(3, .060)
        model = WorldModel()
        model.register(observed)
        row = evaluate_grasp(truth, observed, model, (truth,))
        self.assertTrue(row['false_feasible'])
        self.assertTrue(row['proposed_truth_checks']['clearance_violation'])
        self.assertFalse(row['proposed_truth_checks']['physical_overlap'])

    def test_unknown_and_nonfinite_faults_are_rejected_without_mutating_measured_object(self):
        truth = sphere('ordinary', (.45, 0.), .025)
        observed = truth_world_object(truth)
        model = WorldModel()
        model.register(observed)
        results = invalid_geometry_queries(observed, model)
        self.assertTrue(all(not row['feasible'] for row in results.values()))
        self.assertEqual(observed.geometry.shape.value, 'sphere')
        self.assertTrue(np.isfinite(observed.pose.position).all())

    def test_pick_precondition_query_protects_held_and_uncertain_without_execution(self):
        observed = truth_world_object(sphere('ordinary', (.45, 0.), .025))
        model = WorldModel()
        model.register(observed)
        backend = QueryOnlyBackend()
        planner = make_planner(model)
        self.assertTrue(pick_start_guard(model, observed.object_id, planner, backend)['allowed'])
        for uncertain in (False, True):
            model.set_held(observed.object_id, uncertain=uncertain)
            self.assertFalse(pick_start_guard(model, observed.object_id, planner, backend)['allowed'])
        self.assertFalse(backend.commands)
        with self.assertRaises(RuntimeError):
            backend.move_to_pose_async(None)
        self.assertEqual(backend.commands, ['move_to_pose_async'])


if __name__ == '__main__':
    unittest.main()
