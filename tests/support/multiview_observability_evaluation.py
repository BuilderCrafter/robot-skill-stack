"""Read-only full-path observability scoring; simulator-free research support.

The primary truth plan isolates target dimensional feasibility (including other
targets), while full_scene fields retain physical covers and clutter. Existing
planner obstacle checks also score the actual proposed pose against every true
solid. Both scopes must be safe for strict acceptance; a dimensionally correct
grasp may still collide with an unobserved cover. Real measured cover surfaces
are reported as clutter, with unmatched-target counts retained separately from
off-geometry false positives. No oracle geometry enters the source or updater.
"""
import copy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.manipulation.grasping.clearance import check_obstacles
from tests.support.multiview_iteration2 import SnapshotRig, distribution, object_payload, score_observations
from tests.support.multiview_readiness_research import (
    QueryOnlyBackend, evaluate_grasp, invalid_geometry_queries, make_planner, pick_start_guard,
    plan_payload, truth_world_object,
)
from tests.support.multiview_research import metric_row


class RecordedSource:
    """Record actual batches while forwarding the unchanged provider interface."""
    def __init__(self, provider):
        self.provider, self.batch = provider, None

    def observe(self, context=None):
        self.batch = None
        self.batch = self.provider.observe(context=context)
        return self.batch

    def forget(self, object_id):
        self.provider.forget(object_id)

    def close(self):
        self.provider.close()


def match(objects, observations):
    assigned, selected = set(), {}
    for distance, oi, di in sorted((float(np.linalg.norm(obj.center-obs.pose.position)), oi, di)
                                  for oi, obj in enumerate(objects) for di, obs in enumerate(observations)):
        if distance < .04 and oi not in selected and di not in assigned:
            selected[oi] = di
            assigned.add(di)
    return {objects[oi].name: observations[di] for oi, di in selected.items()}


def score_planning(truth, observed, model, solids, backend, *, targets=None):
    targets = (truth,) if targets is None else targets
    row = evaluate_grasp(truth, observed, model, targets, backend=backend)
    scene_model = WorldModel()
    for obj in solids:
        scene_model.register(truth_world_object(obj))
    scene_planner = make_planner(scene_model)
    full_scene = scene_planner.plan(scene_model.require(truth.name))
    row.update(full_scene_truth_plan=plan_payload(full_scene),
        full_scene_false_feasible=bool(row['estimated_feasible'] and not full_scene.ok),
        full_scene_proposed_obstacle_violation=False, full_scene_proposed_obstacle_evidence=None)
    if observed is not None and row['estimated_feasible']:
        proposed = make_planner(model).plan(observed)
        yaw = proposed.details['wrist_yaw_rad']
        width = row['proposed_truth_checks']['true_required_width_m']
        blocker, evidence = check_obstacles(scene_model.require(truth.name), proposed.plan, yaw,
            width if width is not None else proposed.details['required_width_m'],
            scene_planner.gripper, scene_planner.safety, scene_model)
        row['full_scene_proposed_obstacle_violation'] = blocker is not None
        row['full_scene_proposed_obstacle_evidence'] = dict(blocker=blocker, details=evidence)
    row.update(truth_geometry=object_payload(truth),
        classification_correct=observed is not None and observed.class_name == (
            None if truth.shape == 'ellipsoid' else truth.shape),
        unknown=observed is not None and observed.class_name is None,
        graspable=None if observed is None else observed.graspable,
        geometry_evidence=None, dimension_error_mm=None, axis_or_yaw_error_deg=None)
    if observed is not None:
        g = observed.geometry
        row['geometry_evidence'] = dict(metadata=copy.deepcopy(g.metadata), scores=dict(g.scores),
            confidence=g.confidence, graspable=observed.graspable, estimated_center=observed.pose.position.copy(),
            estimated_size=observed.size.copy(), box_size=g.box_size, yaw=g.yaw,
            radius=g.radius, length=g.length, axis=g.axis)
        if truth.shape != 'ellipsoid':
            geometric = metric_row(truth, SimpleNamespace(geometry=g, center=observed.pose.position, size=observed.size))
            row.update(dimension_error_mm=geometric['dimension_error_mm'],
                       axis_or_yaw_error_deg=geometric['axis_error_deg'])
    return row


def held_signature(model):
    return dict(object_id=model.held_object_id, uncertain=model.held_object_uncertain,
        offset=None if model.held_object_offset_in_ee is None else model.held_object_offset_in_ee.tolist(),
        gripper_state=model.gripper_state.value)


class EvaluationSession:
    """One persistent model/source per scene; each saved snapshot is consumed once."""
    def __init__(self, provider_class, scene_config, *, method, scene_key, replay=False, temporal=False):
        self.rig = SnapshotRig()
        self.provider = provider_class(self.rig, scene_config)
        self.source = RecordedSource(self.provider)
        self.model = WorldModel()
        self.updater = WorldModelUpdater(self.model, self.source, stale_object_ttl=2.5)
        self.backend = QueryOnlyBackend()
        self.method, self.scene_key, self.replay, self.temporal = method, scene_key, replay, temporal
        self.rows, self.samples, self.detections, self.contracts = [], [], [], []
        self.previous_ids, self.initial_id, self.last_now = {}, None, None
        self.last_timestamp, self.last_phase = None, None
        self.phase_states = {}
        self.sentinel_id = None

    def clock(self, now):
        # Replaying does not replace native acquisition stamps. The test clock
        # follows their original timeline so the unchanged updater can consume
        # historical frames; this is not current wall-clock freshness evidence.
        return patch('time.monotonic', return_value=now) if self.replay else nullcontext()

    def _sentinel(self, uncertain):
        if self.sentinel_id is None:
            obj = next(iter(self.model.visible_objects()), None)
            if obj is None:
                return None
            obj = copy.deepcopy(obj)
            obj.object_id = 'research_held_sentinel'
            obj.pose.position += [.4, .4, .3]
            self.model.register(obj)
            self.sentinel_id = obj.object_id
        self.model.set_held(self.sentinel_id, uncertain=uncertain, object_offset_in_ee=[.01, .02, .03])
        return held_signature(self.model)

    def consume(self, frames, objects, clutter, *, index, phase='static'):
        now = min(frame.acquired_at for frame in frames)+1e-6
        self.last_now = now
        with self.clock(now):
            held_before = None
            if self.temporal and phase == 'incomplete':
                held_before = self._sentinel(uncertain=self.last_phase == 'incomplete')
            elif self.temporal and phase == 'restored':
                self.model.set_held(None)
            self.rig.push(frames)
            before_updates = self.updater.update_count
            observations = self.updater.update()
            batch = self.source.batch
            fresh = self.updater.update_count > before_updates
            health = self.updater.health.status
            matches = match(objects, observations)
            ids = {name: obj.object_id for name, obj in matches.items()}
            switches = sum(name in self.previous_ids and self.previous_ids[name] != identity for name, identity in ids.items())
            self.previous_ids.update(ids)
            self.samples.append(dict(scene=self.scene_key, method=self.method, frame_index=index, phase=phase,
                source_health=health, source_error=self.updater.health.error, fresh=fresh,
                original_acquired_at=min(frame.acquired_at for frame in frames),
                batch_timestamp=None if batch is None else batch.timestamp,
                observation_timestamps=[obs.timestamp for obs in observations],
                observed=len(observations), identity_switches=switches,
                early_confirmations=len(observations) if index < 2 else 0,
                diagnostics=copy.deepcopy(self.provider.diagnostics),
                held_before=held_before, held_after=None if held_before is None else held_signature(self.model)))
            if held_before is not None:
                self.contracts.append(dict(kind='held_state_preserved_on_fresh_measurement', phase=phase,
                    frame_index=index, uncertain=held_before['uncertain'],
                    passed=held_before == held_signature(self.model)))
            if index >= 2:
                scored, counts, _ = score_observations(objects+clutter, observations,
                    [self.provider.get_point_cloud(obs.object_id) for obs in observations], method=self.method,
                    trial=self.scene_key, frame_index=index)
                # Real physical covers can themselves be detected, especially
                # in small fixtures. Retain unmatched-target count separately
                # from clouds unsupported by any actual scene surface.
                unmatched = [cloud for cloud in counts['clouds'] if cloud['object_id'] not in ids.values()]
                real_clutter, target_fragments = 0, 0
                for cloud in unmatched:
                    fractions, number = cloud['truth_surface_fractions'], cloud['points']
                    if number >= 8 and sum(fractions[len(objects):]) >= .5:
                        real_clutter += 1
                    elif number >= 8 and sum(fractions[:len(objects)]) >= .5:
                        target_fragments += 1
                counts.update(phase=phase, expected=len(objects), matched=len(matches), missing=len(objects)-len(matches),
                    raw_non_target_observations=len(unmatched), real_clutter_observations=real_clutter,
                    unmatched_target_surface_fragments=target_fragments,
                    false_positives=len(unmatched)-real_clutter-target_fragments)
                self.detections.append(counts)
                for truth in objects:
                    measured = matches.get(truth.name)
                    observed = None if measured is None else self.model.get(measured.object_id)
                    row = score_planning(truth, observed, self.model, objects+clutter, self.backend, targets=objects)
                    row.update(scene=self.scene_key, method=self.method, frame_index=index, phase=phase)
                    self.rows.append(row)
                if self.temporal and objects:
                    obj = matches.get(objects[0].name)
                    state = dict(object_id=None if obj is None else obj.object_id,
                        graspable=None if obj is None else bool(obj.graspable),
                        planner_accepted=False if obj is None else make_planner(self.model).plan(self.model.require(obj.object_id)).ok)
                    self.phase_states[phase] = state
                    if phase == 'complete' and obj is not None:
                        self.initial_id = obj.object_id
            self.last_timestamp, self.last_phase = self.updater.health.last_frame_at, phase

    def frozen(self):
        with self.clock(self.last_now+1e-5):
            before = self.updater.health.last_frame_at
            eligible = {obj.object_id: obj.graspable for obj in self.model.objects()}
            self.updater.update()
            batch = self.source.batch
            self.contracts.append(dict(kind='frozen_frame_no_restore_or_freshness_refresh',
                passed=bool(self.updater.health.last_frame_at == before and batch is not None and
                    batch.timestamp is None and not batch.observations and
                    eligible == {obj.object_id: obj.graspable for obj in self.model.objects()}),
                original_timestamp=before, after_timestamp=self.updater.health.last_frame_at))

    def finish(self):
        with self.clock(self.last_now+1e-4):
            if self.temporal:
                states = self.phase_states
                complete, incomplete, restored = (states.get(phase, {}) for phase in ('complete', 'incomplete', 'restored'))
                self.contracts.append(dict(kind='complete_incomplete_complete_stable_identity', states=states,
                    passed=bool(complete.get('graspable') is True and incomplete.get('graspable') is False and
                        complete.get('planner_accepted') is True and not incomplete.get('planner_accepted') and
                        restored.get('graspable') is True and restored.get('planner_accepted') is True and
                        complete.get('object_id') == incomplete.get('object_id') == restored.get('object_id') == self.initial_id)))
            obj = next((obj for obj in self.model.visible_objects() if obj.object_id != self.sentinel_id), None)
            if obj is not None:
                invalid = invalid_geometry_queries(obj, self.model, self.backend)
                self.contracts.append(dict(kind='unknown_and_invalid_geometry_rejected', details=invalid,
                    passed=all(not value['feasible'] for value in invalid.values())))
                key = obj.object_id
                now = self.last_now+self.updater.health.stale_after+.01
                with patch('time.monotonic', return_value=now):
                    self.updater.cleanup()
                    guard = pick_start_guard(self.model, key, make_planner(self.model), self.backend)
                    self.contracts.append(dict(kind='stale_geometry_rejected', guard=guard,
                                              passed=not guard['allowed'] and not self.model.require(key).visible))
                self.updater.cleanup(now=self.last_now+self.updater.stale_object_ttl+.1)
                self.contracts.append(dict(kind='expired_unprotected_identity_forgotten',
                    passed=not self.model.exists(key) and key not in self.provider.tracker.tracks))
            self.contracts.append(dict(kind='no_backend_commands', passed=not self.backend.commands,
                                       commands=list(self.backend.commands)))
        self.updater.stop()
        self.updater.stop()
        self.contracts.append(dict(kind='source_cleanup', passed=not self.provider.tracker.tracks))
        return [dict(scene=self.scene_key, method=self.method, **value) for value in self.contracts]

    def retain_failure(self, objects, clutter, *, index, phase, error):
        """A failed provider still contributes every expected missing-object row."""
        for truth in objects:
            if any(row['frame_index'] == index and row['object'] == truth.name for row in self.rows):
                continue
            if index >= 2:
                row = score_planning(truth, None, WorldModel(), objects+clutter, self.backend, targets=objects)
                row.update(scene=self.scene_key, method=self.method, frame_index=index, phase=phase,
                           provider_error=error)
                self.rows.append(row)
        if not any(row['frame_index'] == index for row in self.samples):
            self.samples.append(dict(scene=self.scene_key, method=self.method, frame_index=index, phase=phase,
                source_health='failed', source_error=error, fresh=False, observed=0,
                identity_switches=0, early_confirmations=0, diagnostics={}))
        if index >= 2 and not any(row['frame_index'] == index for row in self.detections):
            _, counts, _ = score_observations(objects, [], [], method=self.method,
                trial=self.scene_key, frame_index=index)
            counts.update(phase=phase, provider_error=error)
            self.detections.append(counts)


def method_summary(rows, samples, contracts, detections):
    valid = [row for row in rows if row.get('valid_fixture')]
    invalid = [row for row in rows if not row.get('valid_fixture')]
    values = dict(expected=len(valid), expected_all_rows=len(rows), invalid_fixture_rows=len(invalid),
        invalid_fixture_false_feasible=sum(row['false_feasible'] for row in invalid),
        detected=sum(row['detected'] for row in valid), missing=sum(not row['detected'] for row in valid),
        correct=sum(row['classification_correct'] for row in valid), unknown=sum(row['unknown'] for row in valid),
        graspable=sum(row['graspable'] is True for row in valid),
        true_feasible=sum(row['truth_feasible'] for row in valid),
        estimated_feasible=sum(row['estimated_feasible'] for row in valid),
        correctly_feasible=sum(row['truth_feasible'] and row['estimated_feasible'] for row in valid),
        correctly_rejected=sum(not row['truth_feasible'] and not row['estimated_feasible'] for row in valid),
        false_feasible=sum(row['false_feasible'] for row in valid),
        false_infeasible=sum(row['false_infeasible'] for row in valid),
        full_scene_false_feasible=sum(row['full_scene_false_feasible'] for row in valid),
        full_scene_proposed_obstacle_violations=sum(row['full_scene_proposed_obstacle_violation'] for row in valid),
        **{label: sum(bool(row.get('proposed_truth_checks') and row['proposed_truth_checks'][flag]) for row in valid)
           for label, flag in [('width_violations', 'width_violation'), ('clearance_violations', 'clearance_violation'),
                               ('physical_overlaps', 'physical_overlap')]},
        center_error_mm=distribution([row['center_error_mm'] for row in valid]),
        dimension_error_mm=distribution([row['dimension_error_mm'] for row in valid]),
        identity_switches=sum(row['identity_switches'] for row in samples),
        source_health_failures=sum(row['source_health'] != 'fresh' for row in samples),
        unconsumed_fresh_samples=sum(not row['fresh'] for row in samples),
        early_confirmations=sum(row['early_confirmations'] for row in samples),
        failed_contracts=[row['kind'] for row in contracts if not row['passed']],
        **{name: sum(row.get(name, 0) for row in detections if row.get('valid_fixture')) for name in
           ('false_positives', 'false_splits', 'mixed_accepted_clouds', 'duplicate_ids',
            'raw_non_target_observations', 'real_clutter_observations', 'unmatched_target_surface_fragments')})
    values['safe_valid_geometry'] = not any(values[key] for key in
        ('false_feasible', 'width_violations', 'clearance_violations', 'physical_overlaps'))
    values['contracts_passed'] = not (values['failed_contracts'] or values['identity_switches'] or
        values['source_health_failures'] or values['unconsumed_fresh_samples'] or values['early_confirmations'])
    values['full_scene_safety_passed'] = values['safe_valid_geometry'] and not (
        values['full_scene_false_feasible'] or values['full_scene_proposed_obstacle_violations'])
    return values
