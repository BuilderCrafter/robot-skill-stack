#!/usr/bin/env python3
"""One isolated Home-only TGS solver diagnostic; no production patch.

The frozen controlled runner supplies every action and observation. This wrapper
changes velocity iterations from 1 to 0 by default, or position iterations from
32 to 64 or 255 with --position-iterations. The other count stays nominal. It
restores the original session opinion. Its SDK
getter is USD-backed, not proof of an effective native island iteration count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from scripts import diagnose_global_stabilization_v4_1 as controlled


VELOCITY = 'physxArticulation:solverVelocityIterationCount'
POSITION = 'physxArticulation:solverPositionIterationCount'


def attribute_opinions(session, paths, name):
    rows = []
    for path in paths:
        spec = session.GetAttributeAtPath(str(path)+'.'+name)
        present = spec is not None and spec.HasInfo('default')
        rows.append(dict(path=str(path), existed=spec is not None,
            default_present=present, default=spec.GetInfo('default') if present else None))
    return rows


def missing_session_ancestors(session, paths):
    """Only these previously absent specs can belong to our count setter."""
    candidates = set()
    for path in paths:
        parts = str(path).strip('/').split('/')
        for index in range(1, len(parts)+1):
            ancestor = '/'+ '/'.join(parts[:index])
            if session.GetPrimAtPath(ancestor) is None:
                candidates.add(ancestor)
    return sorted(candidates, key=lambda path: (path.count('/'), path), reverse=True)


def remove_owned_empty_overs(stage, session, candidates):
    """Preserve pre-existing specs and all newly authored unrelated opinions."""
    from pxr import Sdf
    removed = []
    for path in candidates:
        spec = session.GetPrimAtPath(path)
        if (spec is not None and spec.specifier == Sdf.SpecifierOver
                and not list(spec.properties) and not list(spec.nameChildren)
                and set(spec.ListInfoKeys()).issubset({'specifier'})):
            if stage.RemovePrim(path):
                removed.append(path)
    return removed


def native_properties(robot):
    try:
        from omni.isaac.dynamic_control import _dynamic_control
        api = _dynamic_control.acquire_dynamic_control_interface()
        handle = api.get_articulation(robot.prim_path)
        properties = api.get_articulation_properties(handle)
        return dict(available=True, position=int(properties.solver_position_iteration_count),
            velocity=int(properties.solver_velocity_iteration_count),
            scope='Native actor requested counts; not effective island counts')
    except Exception as error:
        return dict(available=False, error=f'{type(error).__name__}: {error}',
            scope='No extension is enabled or installed by this diagnostic')


def scene_configuration(stage):
    from pxr import UsdPhysics
    scenes, actors = [], []
    for prim in stage.Traverse():
        scene = prim.IsA(UsdPhysics.Scene)
        rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        articulation = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        if not (scene or rigid or articulation):
            continue
        attributes = {str(a.GetName()): controlled.fixture.serial(a.Get()) for a in prim.GetAttributes()
            if any(word in str(a.GetName()).lower() for word in ('iteration', 'solvertype', 'enablegpudynamics'))}
        row = dict(path=str(prim.GetPath()), attributes=attributes,
            articulation=articulation, rigid_body=rigid)
        if scene:
            scenes.append(row)
        else:
            actors.append(row)
    minimums = [row['attributes'].get('physxScene:minVelocityIterationCount') for row in scenes]
    return dict(scenes=scenes, relevant_actor_requests=actors,
        scene_minimum_allows_zero=bool(minimums) and all(value == 0 for value in minimums),
        scope='USD scene and actor requests; an island may use the highest connected actor count. No native effective-island getter is assumed.')


def maximum_difference(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != b.shape or not a.size:
        return None
    return float(np.max(np.abs(a-b)))


def compare_home(baseline, current):
    old = next((row for row in baseline.get('operations', []) if row['label'] == 'home'), None)
    if old is None:
        return dict(available=False, reason='Reference has no observed Home')
    before, after = old['before'], current['before']
    old_plan, plan = old.get('plan_trace') or {}, current.get('plan_trace') or {}
    old_commands, commands = old_plan.get('commanded', []), plan.get('commanded', [])
    count = min(len(old_commands), len(commands))
    elapsed = lambda rows: [row['sim_time']-rows[0]['sim_time'] for row in rows[:count]] if count else []
    return dict(available=True,
        before_home_joint_position_max_difference_rad=maximum_difference(before['feedback']['positions'], after['feedback']['positions']),
        before_home_joint_velocity_max_difference_rad_s=maximum_difference(before['feedback']['velocities'], after['feedback']['velocities']),
        before_home_simulation_time_equal=before['simulation_time'] == after['simulation_time'],
        before_home_physics_step_equal=before['physics_step'] == after['physics_step'],
        route_max_difference_rad=maximum_difference(old_plan.get('route', []), plan.get('route', [])),
        trajectory_samples_equal=old_plan.get('trajectory') == plan.get('trajectory'),
        common_command_prefix_count=count,
        command_prefix_q_max_difference_rad=maximum_difference([r['q'] for r in old_commands[:count]], [r['q'] for r in commands[:count]]),
        command_prefix_qd_max_difference_rad_s=maximum_difference([r['qd'] for r in old_commands[:count]], [r['qd'] for r in commands[:count]]),
        command_prefix_elapsed_max_difference_s=maximum_difference(elapsed(old_commands), elapsed(commands)),
        command_counts=[len(old_commands), len(commands)],
        scope='Common pre-completion command prefix; successful execution may end earlier than the failed reference.')


def zero_arm_velocity(action, indices):
    if action is None:
        return False
    velocity = action.get('joint_velocities')
    if velocity is None:
        return False
    count = max(max(indices)+1, len(velocity) if action.get('joint_indices') is None else 0)
    error = controlled.arm_target_error(np.zeros(count),
        dict(joint_positions=action.get('joint_velocities'), joint_indices=action.get('joint_indices')), indices)
    return error is not None and np.max(np.abs(error)) <= 1e-12


async def observe_post_home_hold(observer, seconds, *, label='post_home_hold', expected_position_count=32):
    """Observe existing targets at the specified count; never command the arm."""
    home = next((row for row in reversed(observer.operations) if row['label'] == 'home'), None)
    if home is None:
        observer.metadata[label] = dict(available=False, reason='No Home operation observed')
        return
    row = observer.begin_operation(label, home['target'], kind='post_hold')
    started = row['before']['simulation_time']
    try:
        for _ in range(601):
            if observer.bundle.world.current_time-started >= seconds:
                break
            await observer.backend._next_update()
        else:
            raise RuntimeError('Post-Home observation did not advance one simulation second')
    finally:
        observer.end_operation(row)
    samples = observer.samples[row['first_sample']:row['last_sample']]
    gaps = [max(abs(value) for value in sample['actual_minus_applied_target_arm_rad']) for sample in samples]
    position_errors = [float(np.linalg.norm(np.asarray(sample['tcp_pose']['position'])-home['target']['position'])) for sample in samples]
    orientation_errors = [float(observer.backend._quat_error(np.asarray(sample['tcp_pose']['orientation']),
        np.asarray(home['target']['orientation']))) for sample in samples]
    bad_since, maximum_bad_interval = None, 0.
    for sample, gap in zip(samples, gaps):
        if gap > .15:
            bad_since = sample['simulation_time'] if bad_since is None else bad_since
            maximum_bad_interval = max(maximum_bad_interval, sample['simulation_time']-bad_since)
        else:
            bad_since = None
    report = dict(available=True, requested_seconds=seconds,
        observed_simulation_seconds=row['after']['simulation_time']-started,
        additional_arm_commands=row['last_action']-row['first_action'],
        maximum_applied_target_gap_rad=max(gaps), maximum_continuous_gap_over_015_rad_seconds=maximum_bad_interval,
        maximum_position_error_m=max(position_errors), final_position_error_m=position_errors[-1],
        maximum_orientation_error_rad=max(orientation_errors), final_orientation_error_rad=orientation_errors[-1],
        final_actual_arm_velocity_rad_s=np.asarray(samples[-1]['feedback']['velocities'])[list(observer.indices)].tolist(),
        all_requested_arm_velocity_targets_zero=all(zero_arm_velocity(sample['requested_action'], observer.indices) for sample in samples),
        all_applied_arm_velocity_targets_zero=all(zero_arm_velocity(sample['feedback']['action'], observer.indices) for sample in samples),
        expected_native_actor_position_count=expected_position_count,
        all_native_actor_counts_expected=all(sample.get('native_actor_solver_properties', {}).get('available')
            and sample['native_actor_solver_properties']['position'] == expected_position_count
            and sample['native_actor_solver_properties']['velocity'] == 1 for sample in samples),
        meaning='Separate post-Home stability observation; excluded from original Home duration and acceptance.')
    report['all_native_actor_counts_nominal'] = expected_position_count == 32 and report['all_native_actor_counts_expected']
    report['kinematic_stability_passed'] = (report['additional_arm_commands'] == 0
        and report['all_requested_arm_velocity_targets_zero'] and report['all_applied_arm_velocity_targets_zero']
        and report['all_native_actor_counts_expected'] and max(gaps) <= .40 and maximum_bad_interval < .25
        and max(position_errors) <= controlled.fixture.POSITION_TOLERANCE
        and max(orientation_errors) <= controlled.fixture.ORIENTATION_TOLERANCE)
    row[label] = report
    observer.metadata[label] = report


def probe_observer_class(baseline, *, position_iterations=None, retain_home_solver=False):
    base = controlled.Observer
    property_name = POSITION if position_iterations is not None else VELOCITY
    requested_count = position_iterations if position_iterations is not None else 0
    axis = 'position' if position_iterations is not None else 'velocity'

    class HomeProbe(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._lease = None

        def _restore_solver(self):
            lease = self._lease
            if lease is None:
                return
            from pxr import Usd
            with Usd.EditContext(lease['stage'], lease['session']):
                getattr(self.robot._articulation_view, 'set_solver_'+axis+'_iteration_counts')(lease['old'])
                for item in lease['opinions']:
                    prim = lease['stage'].GetPrimAtPath(item['path'])
                    if not item['existed']:
                        prim.RemoveProperty(property_name)
                    elif not item['default_present']:
                        lease['session'].GetAttributeAtPath(item['path']+'.'+property_name).ClearInfo('default')
                    else:
                        lease['session'].GetAttributeAtPath(item['path']+'.'+property_name).SetInfo('default', item['default'])
                removed = remove_owned_empty_overs(lease['stage'], lease['session'], lease['owned_empty_ancestor_candidates'])
            record = lease['record']
            record['restored_usd_requested_velocity_counts'] = controlled.fixture.serial(
                self.robot._articulation_view.get_solver_velocity_iteration_counts())
            record['restored_usd_requested_position_counts'] = controlled.fixture.serial(
                self.robot._articulation_view.get_solver_position_iteration_counts())
            record['native_actor_properties_after_restore'] = native_properties(self.robot)
            record['session_layer_sha256_after_restore'] = hashlib.sha256(lease['session'].ExportToString().encode()).hexdigest()
            record['restored_session_attribute_opinions'] = attribute_opinions(lease['session'],
                [item['path'] for item in lease['opinions']], property_name)
            record['original_session_opinion_restored'] = record['restored_session_attribute_opinions'] == lease['opinions']
            record['restoration_scope'] = 'Exact original count attribute/default opinions; unrelated session opinions are preserved.'
            record['owned_empty_session_prims_removed'] = removed
            record['entire_session_layer_equal'] = record['session_layer_sha256_before'] == record['session_layer_sha256_after_restore']
            self._lease = None

        def begin_operation(self, label, target, *, kind='motion'):
            if label != 'home':
                return super().begin_operation(label, target, kind=kind)
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            session = stage.GetSessionLayer()
            if stage.GetEditTarget().GetLayer() != session:
                raise RuntimeError('The diagnostic requires the existing session edit target')
            solver = self.bundle.world.get_physics_context().get_solver_type()
            view = self.robot._articulation_view
            position, velocity = np.asarray(view.get_solver_position_iteration_counts()), np.asarray(view.get_solver_velocity_iteration_counts())
            scene = scene_configuration(stage)
            if solver != 'TGS' or not np.all(position == 32) or not np.all(velocity == 1):
                raise RuntimeError('Probe requires observed TGS32-position/1-velocity baseline')
            if position_iterations is None and not scene['scene_minimum_allows_zero']:
                raise RuntimeError('Scene minimum velocity iterations does not establish that zero is allowed')
            opinions = attribute_opinions(session, view.prim_paths, property_name)
            owned_ancestors = missing_session_ancestors(session, view.prim_paths)
            before = self.state()
            record = dict(solver_type=solver, old_usd_requested_position_counts=position.tolist(),
                old_usd_requested_velocity_counts=velocity.tolist(), changed_count=axis,
                requested_position_iterations=position_iterations if position_iterations is not None else 32,
                requested_velocity_iterations=1 if position_iterations is not None else 0,
                scene_configuration=scene, original_session_attribute_opinions=opinions,
                owned_empty_ancestor_candidates=owned_ancestors,
                session_layer_identifier=session.identifier,
                session_layer_sha256_before=hashlib.sha256(session.ExportToString().encode()).hexdigest(),
                native_actor_properties_before=native_properties(self.robot),
                getter_scope='Articulation SDK count getters read USD; do not infer native effective island count from them.',
                scene_or_vendor_file_saved=False, additional_physics_updates=0,
                restoration_deferred_for_retained_hold=retain_home_solver,
                unchanged_fields=f'{"Velocity iterations 1" if position_iterations is not None else "Position iterations 32"}; all gains, limits, trajectory, IK, scene geometry and preceding operations remain unchanged.')
            self._lease = dict(stage=stage, session=session, old=(position if position_iterations is not None else velocity).copy(),
                opinions=opinions, owned_empty_ancestor_candidates=owned_ancestors, record=record)
            try:
                getattr(view, 'set_solver_'+axis+'_iteration_counts')(np.full_like(velocity, requested_count, dtype=np.int32))
                record['after_set_usd_requested_velocity_counts'] = controlled.fixture.serial(view.get_solver_velocity_iteration_counts())
                record['after_set_usd_requested_position_counts'] = controlled.fixture.serial(view.get_solver_position_iteration_counts())
                record['native_actor_properties_after_set'] = native_properties(self.robot)
                after = self.state()
                record['no_physics_step_or_time_advance_during_set'] = (
                    before['physics_step'] == after['physics_step'] and before['simulation_time'] == after['simulation_time'])
                record['joint_state_change_during_set_rad'] = maximum_difference(before['feedback']['positions'], after['feedback']['positions'])
                row = super().begin_operation(label, target, kind=kind)
                row['solver_velocity_probe'] = record
                self.metadata['home_solver_velocity_probe'] = record
                return row
            except BaseException:
                self._restore_solver()
                raise

        def sample(self):
            super().sample()
            if self._lease is not None or self.phase == 'post_home_hold':
                self.samples[-1]['native_actor_solver_properties'] = native_properties(self.robot)

        def end_operation(self, row, result=None, *, cancelled=False):
            completed = False
            try:
                super().end_operation(row, result, cancelled=cancelled)
                if row['label'] == 'home':
                    row['solver_velocity_probe']['baseline_comparison'] = compare_home(baseline, row)
                completed = True
            finally:
                if row['label'] == 'home' and (not retain_home_solver or not completed):
                    self._restore_solver()

        def restore(self):
            try:
                self._restore_solver()
            finally:
                super().restore()

    return HomeProbe


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('B', 'C'), required=True)
    parser.add_argument('--reference-result', required=True)
    parser.add_argument('--position-iterations', type=int, choices=(64, 255), default=None,
        help='One isolated position-count probe (32 to 64 or scene ceiling 255); velocity stays 1. Omit for the original velocity-count probe.')
    parser.add_argument('--post-hold-seconds', type=float, choices=(1.,), default=None,
        help='Observe existing arm targets for one simulation second after Home restores nominal solver counts.')
    parser.add_argument('--retained-hold-seconds', type=float, choices=(1.,), default=None,
        help='Keep position count 255 for one second of observation, then restore 32 before the required post hold.')
    known, _ = parser.parse_known_args(arguments)
    if known.retained_hold_seconds is not None and (known.position_iterations != 255 or known.post_hold_seconds != 1.):
        parser.error('--retained-hold-seconds requires --position-iterations 255 and --post-hold-seconds 1')
    for option in ('--position-iterations', '--post-hold-seconds', '--retained-hold-seconds'):
        for index, argument in enumerate(arguments):
            if argument == option:
                del arguments[index:index+2]
                break
            if argument.startswith(option+'='):
                del arguments[index]
                break
    reference = json.loads(Path(known.reference_result).read_text())
    trace = reference['controlled']['raw_trace']
    raw = Path(trace['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != trace['sha256']:
        raise ValueError('Reference trace SHA mismatch')
    baseline = json.loads(raw)
    original_hashes, original_reduce = controlled.source_hashes, controlled.reduce_observations
    original_callback_for = controlled.callback_for

    def hashes():
        rows = original_hashes()
        rows[str(Path(__file__).relative_to(controlled.fixture.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return rows

    def reduce(observer, capture):
        rows = original_reduce(observer, capture)
        for compact, full in zip(rows, observer.operations):
            if 'solver_velocity_probe' in full:
                compact['solver_velocity_probe'] = full['solver_velocity_probe']
            for label in ('retained_home_hold', 'post_home_hold'):
                if label in full:
                    report = full[label]
                    report['validation_passed'] = (report['kinematic_stability_passed']
                        and compact['contacts']['reported_points'] == 0
                        and compact['clearance']['minimum_m'] is not None and compact['clearance']['minimum_m'] > 0.)
                    compact[label] = report
                    compact['strict_success'] = None
        return rows

    def callback_for(mode, state):
        original = original_callback_for(mode, state)
        async def callback(*args):
            result = await original(*args)
            if known.retained_hold_seconds is not None:
                try:
                    await observe_post_home_hold(state['observer'], known.retained_hold_seconds,
                        label='retained_home_hold', expected_position_count=255)
                finally:
                    state['observer']._restore_solver()
            if known.post_hold_seconds is not None:
                await observe_post_home_hold(state['observer'], known.post_hold_seconds)
            return result
        return callback

    with patch.object(controlled, 'Observer', probe_observer_class(baseline, position_iterations=known.position_iterations,
            retain_home_solver=known.retained_hold_seconds is not None)), \
            patch.object(controlled, 'source_hashes', hashes), patch.object(controlled, 'reduce_observations', reduce), \
            patch.object(controlled, 'callback_for', callback_for):
        return controlled.main(arguments)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
