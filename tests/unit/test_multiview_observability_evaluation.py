"""Evaluation correctness; the native test separately exercises real depth."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from robot_skill_stack.world.model.provider import ObservationBatch
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from scripts.benchmark_multiview_observability import package_hashes, result_gate, verify_freeze
from tests.support.multiview_iteration2 import box
from tests.support.multiview_observability_evaluation import EvaluationSession, method_summary
from tests.support.multiview_observability_evaluation import score_planning
from tests.support.multiview_readiness_research import QueryOnlyBackend, truth_world_object
from tests.support.multiview_research import ResearchObject
from robot_skill_stack.world.model.world_model import WorldModel
from scripts.replay_multiview_observability_robot import replay_robot
from scripts.benchmark_multiview_robustness import save_frames
from tests.support.multiview_iteration2 import object_payload


class ProtocolSource:
    """Controlled batch source for verifying scorer/WorldModel transitions only."""
    def __init__(self, rig, scene):
        self.rig, self.hits = rig, 0
        self.tracker = SimpleNamespace(tracks={})
        self.diagnostics = {}

    def observe(self, context=None):
        frames = self.rig.capture()
        if frames is None:
            return ObservationBatch()
        self.hits += 1
        now = frames[0].acquired_at
        if self.hits < 3:
            return ObservationBatch([], now)
        complete = frames[0].depth[0, 0] == 1
        geometry = PrimitiveGeometry('cube', confidence=.99, box_size=np.full(3, .05), yaw=0.,
                                     metadata={'parameter_observable': bool(complete)})
        obs = ObjectObservation(object_id='stable_1', class_name='cube', pose=Pose(np.array([.45, 0., .025])),
            size=np.full(3, .05), geometry=geometry, graspable=bool(complete), timestamp=now)
        self.tracker.tracks['stable_1'] = True
        return ObservationBatch([obs], now)

    def get_point_cloud(self, object_id):
        return None

    def forget(self, object_id):
        self.tracker.tracks.pop(object_id, None)

    def close(self):
        self.tracker.tracks.clear()


class ObservabilityEvaluationTests(unittest.TestCase):
    def test_saved_robot_adapter_keeps_source_across_phases_and_native_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir, output = root/'capture', root/'replay'
            source_dir.mkdir()
            target = box('target', (.45, 0.), (.05, .05, .05))
            phases, samples, counts = [], [], []
            for phase, indices in [('native_phase_A', (0, 1)), ('native_phase_B', (2,))]:
                snapshots = []
                for local_index, index in enumerate(indices):
                    frame = DepthFrame('A', (index, index), np.ones((1, 1)), np.eye(3), np.eye(4),
                                       100.+index*.1, None)
                    snapshots.append(dict(index=local_index, files=save_frames(source_dir, phase, local_index, (frame,)),
                        objects=[object_payload(target)], visible_surface_points={'0': {'target': 12}}))
                    samples.append(dict(phase=phase, frame_index=local_index, fresh_capture=True,
                                        delivered_at=frame.acquired_at+.05))
                    counts.append(dict(trial=phase, frame_index=local_index, confirmation_frame=index < 2))
                phases.append(dict(key=phase, frames=snapshots))
            manifest = dict(scenes=phases, workspace_min=[.2, -.4, 0.], workspace_max=[.7, .4, .22])
            for name, value in [('capture_manifest', manifest), ('samples', samples), ('detections', counts)]:
                (source_dir/f'{name}.json').write_text(json.dumps(value))
            result = replay_robot(source_dir, output, ProtocolSource, current_class=ProtocolSource)
            observations = json.loads((output/'observations.json').read_text())
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['metrics']['summary']['improved']['correct'], 1)
        self.assertEqual(result['metrics']['summary']['improved']['consumed_native_samples'], 3)
        self.assertTrue(all(row['timestamp'] == 100.2 for row in observations))
        self.assertTrue(all(row['phase'] == 'native_phase_B' for row in observations))

    def test_physical_cover_collision_is_retained_separately_from_target_dimensions(self):
        target = box('target', (.45, 0.), (.05, .05, .05))
        roof = ResearchObject('physical_roof', 'cube', np.array([.45, 0., .105]), np.array([.1, .1, .01]))
        observed = truth_world_object(target)
        model = WorldModel()
        model.register(observed)
        row = score_planning(target, observed, model, (target, roof), QueryOnlyBackend())
        self.assertTrue(row['truth_feasible'])
        self.assertFalse(row['false_feasible'])
        self.assertTrue(row['full_scene_false_feasible'])
        self.assertTrue(row['full_scene_proposed_obstacle_violation'])
        self.assertEqual(row['full_scene_proposed_obstacle_evidence']['blocker'], 'physical_roof')

    def test_held_pick_rejection_is_separate_from_geometry_false_infeasibility(self):
        target = box('target', (.45, 0.), (.05, .05, .05))
        observed = truth_world_object(target)
        model = WorldModel()
        model.register(observed)
        model.set_held(observed.object_id, uncertain=True)
        row = score_planning(target, observed, model, (target,), QueryOnlyBackend())
        self.assertTrue(row['estimated_feasible'])
        self.assertFalse(row['read_only_pick_precondition_feasible'])
        self.assertFalse(row['false_infeasible'])
        self.assertTrue(model.held_object_uncertain)
        roof = ResearchObject('physical_roof', 'cube', np.array([.45, 0., .105]), np.array([.1, .1, .01]))
        row = score_planning(target, observed, model, (target, roof), QueryOnlyBackend())
        self.assertFalse(row['read_only_pick_precondition_feasible'])
        self.assertTrue(row['full_scene_false_feasible'])
        self.assertTrue(row['full_scene_proposed_obstacle_violation'])

    def test_strict_current_gate_retains_bad_baseline_and_fails_invalid_or_unsafe_current(self):
        summary = dict(baseline=dict(contracts_passed=False, safe_valid_geometry=False, full_scene_safety_passed=False),
                       improved=dict(contracts_passed=True, safe_valid_geometry=True, full_scene_safety_passed=True))
        self.assertTrue(result_gate(summary, [dict(valid=True)], [], strict=True))
        self.assertFalse(result_gate(summary, [dict(valid=False)], [], strict=True))
        self.assertFalse(result_gate(summary, [dict(valid=True)], ['runtime error'], strict=True))
        summary['improved']['safe_valid_geometry'] = False
        self.assertFalse(result_gate(summary, [dict(valid=True)], [], strict=True))
        summary['improved']['safe_valid_geometry'] = True
        summary['improved']['full_scene_safety_passed'] = False
        self.assertFalse(result_gate(summary, [dict(valid=True)], [], strict=True))

    def test_full_protocol_revokes_freezes_restores_and_preserves_manipulation_state(self):
        target = box('target', (.45, 0.), (.05, .05, .05))
        session = EvaluationSession(ProtocolSource, None, method='controlled_test', scene_key='temporal',
                                    replay=True, temporal=True)
        for index in range(9):
            phase = 'complete' if index < 4 else 'incomplete' if index < 6 else 'restored'
            depth = np.full((1, 1), 2. if phase == 'incomplete' else 1.)
            frame = DepthFrame('A', (index, index), depth, np.eye(3), np.eye(4), 100.+index*.1, None)
            session.consume((frame,), (target,), (), index=index, phase=phase)
            if index == 5:
                session.frozen()
        contracts = session.finish()
        self.assertTrue(all(row['passed'] for row in contracts), contracts)
        self.assertEqual([row['uncertain'] for row in contracts if row['kind'] ==
                          'held_state_preserved_on_fresh_measurement'], [False, True])
        self.assertEqual([row['graspable'] for row in session.rows], [True, True, False, False, True, True, True])
        self.assertTrue(all(row['object_id'] == 'stable_1' for row in session.rows))
        self.assertEqual(session.samples[-1]['batch_timestamp'], 100.8)

    def test_provider_failure_stays_in_denominator_and_invalid_is_separate(self):
        target = box('target', (.45, 0.), (.05, .05, .05))
        session = EvaluationSession(ProtocolSource, None, method='test', scene_key='failure', replay=True)
        session.retain_failure((target,), (), index=3, phase='static', error='Deliberate provider exception')
        session.rows[0]['valid_fixture'] = True
        session.detections[0]['valid_fixture'] = True
        values = method_summary(session.rows, session.samples, [], session.detections)
        self.assertEqual(values['expected'], 1)
        self.assertEqual(values['missing'], 1)
        self.assertEqual(values['false_infeasible'], 1)
        self.assertFalse(values['contracts_passed'])
        session.rows[0]['valid_fixture'] = False
        values = method_summary(session.rows, session.samples, [], session.detections)
        self.assertEqual(values['expected'], 0)
        self.assertEqual(values['invalid_fixture_rows'], 1)

    def test_holdout_freeze_guard_requires_every_package_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'freeze.json'
            hashes = package_hashes()
            path.write_text(json.dumps({'sha256': hashes}))
            self.assertEqual(verify_freeze(path)['verified_package_sha256'], hashes)
            hashes.pop(next(iter(hashes)))
            path.write_text(json.dumps({'sha256': hashes}))
            with self.assertRaisesRegex(ValueError, 'omits package files'):
                verify_freeze(path)
        with self.assertRaisesRegex(ValueError, 'before generating'):
            verify_freeze(None)


if __name__ == '__main__':
    unittest.main()
