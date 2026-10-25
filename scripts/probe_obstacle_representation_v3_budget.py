#!/usr/bin/env python3
"""Actual observed-cell budget sensitivity through the frozen public Lula probe.

One saved C-narrow canonical snapshot, its original clock, production exact
packing/selection, and budgets16/32/64/128. No SimulationApp or motion is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
CASE = 'C_asymmetric_1.15_0.65-narrow/job-00018'
BUDGETS = (16, 32, 64, 128)
NAMES = {16: 'b16', 32: 'b32', 64: 'baseline', 128: 'b128'}
PROTECTED = tuple(ROOT/path for path in (
    'scripts/benchmark_obstacle_representation_v3.py', 'scripts/probe_obstacle_representation_v3.py',
    'scripts/summarize_obstacle_representation_v3.py', 'scripts/obstacle_representation_v3.py',
    'scripts/obstacle_representation_v3_cylinder.py', 'scripts/evaluate_obstacle_representation_v3.py',
    'scripts/run_obstacle_representation_v3_jobs.py', 'scripts/diagnose_reactive_motion.py',
    'robot_skill_stack/world/spatial_mapping/obstacles.py', 'robot_skill_stack/world/spatial_mapping/occupancy.py',
    'robot_skill_stack/world/spatial_mapping/runtime.py', 'robot_skill_stack/world/spatial_mapping/rgbd.py',
    'robot_skill_stack/integrations/isaac/manipulation/motion/reactive.py',
    'robot_skill_stack/integrations/isaac/manipulation/motion/obstacles.py'))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    from scripts.evaluate_obstacle_representation_v3 import plain
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(plain(value), separators=(',', ':'), allow_nan=False)+'\n')


def norm(value):
    if isinstance(value, (tuple, list)):
        return math.sqrt(sum(norm(item)**2 for item in value))
    return abs(float(value))


def nondecreasing(values, *, absolute_tolerance=1e-9, relative_tolerance=1e-12):
    return all(last+absolute_tolerance+relative_tolerance*max(abs(first), abs(last)) >= first
        for first, last in zip(values, values[1:]))


def nested_selections(packets, snapshot):
    from scripts.probe_obstacle_representation_v3 import selected_cell_union
    checks = []
    for first, last in zip(BUDGETS, BUDGETS[1:]):
        low, high = packets[NAMES[first]], packets[NAMES[last]]
        low_cells, high_cells = selected_cell_union(low, snapshot), selected_cell_union(high, snapshot)
        checks.append(dict(from_budget=first, to_budget=last, selected_cells_subset=low_cells <= high_cells,
            numeric_cuboid_prefix=low['boxes'] == high['boxes'][:len(low['boxes'])],
            added_observed_cells=len(high_cells-low_cells)))
    if not all(row['selected_cells_subset'] and row['numeric_cuboid_prefix'] for row in checks):
        raise ValueError('Budget selections do not preserve nested original numerical geometry')
    return checks


def prepare(args):
    import numpy as np
    from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractionSettings, ObstacleExtractor
    from scripts import evaluate_obstacle_representation_v3 as evaluate
    from scripts.obstacle_representation_v3 import prepare_evidence
    record = next(row for row in read(args.manifest)['records'] if row['case_id'] == args.case_id)
    primary = next(row for row in read(args.primary_report)['cases'] if row['case_id'] == args.case_id)
    budget_reference = next(row for row in read(args.budget_report)['cases'] if row['case_id'] == args.case_id)
    original_bundle = next(row for row in read(args.primary_bundle)['cases'] if row['case_id'] == args.case_id)
    now, ttl = record['snapshot']['evaluated_at'], record['snapshot']['evidence_ttl']
    clock = [row['value'] for row in record['source_artifacts']['clock_reads'] if row['phase'] == 'processor']
    if not clock or now != clock[-1]:
        raise ValueError('Extraction clock is not the original recorded processor extraction clock')
    snapshot = evaluate.load_snapshot(record)
    digest = evaluate.array_digest(snapshot)
    if digest != primary['snapshot_arrays_sha256'] or digest != budget_reference['snapshot_arrays_sha256']:
        raise ValueError('Actual canonical snapshot differs from the published primary/budget audit inputs')
    evidence = prepare_evidence(snapshot, now=now, evidence_ttl=ttl)
    packets, audits = {}, []
    fields = ('occupied_voxels', 'supported_voxels', 'candidate_boxes', 'proxy_count',
        'represented_voxels', 'omitted_voxels', 'selected_floor_boxes', 'selected_hazard_boxes')
    for budget in BUDGETS:
        key = 'baseline' if budget == 64 else 'baseline_'+str(budget)
        reference = primary['variants'][key] if budget == 64 else budget_reference['variants'][key]
        settings = ObstacleExtractionSettings(budget, record.get('support_plane_z', 0.))
        packet = ObstacleExtractor(settings, temporal_probabilities=True).extract(snapshot, now=now, evidence_ttl=ttl)
        bounds, selected, errors = evaluate.audit_packet(snapshot, packet, evidence.supported, evidence.eligible, now, ttl)
        if reference['status'] != 'PASS_EXACT_AUDIT' or any(reference['audit'].values()) or any(errors.values()):
            raise ValueError('Prior or new exact occupancy/expiry audit failed')
        if any(packet.diagnostics[key] != reference['diagnostics'][key] for key in fields):
            raise ValueError('Budget counts differ from the existing audit')
        with np.load(reference['cell_audit_npz'], allow_pickle=False) as saved:
            equal = all(np.array_equal(current, saved[key]) for key, current in
                (('selected', selected), ('supported', evidence.supported), ('eligible', evidence.eligible), ('selected_bounds', bounds)))
        if not equal:
            raise ValueError('Original audited cell sets or exact numeric bounds differ')
        packets[NAMES[budget]] = evaluate.plain(dict(boxes=[asdict(box) for box in packet.boxes], diagnostics=dict(packet.diagnostics)))
        audits.append(dict(budget=budget, representation=NAMES[budget], counts={key: packet.diagnostics[key] for key in fields},
            exact_cell_and_expiry_audit=errors, matches_existing_audited_cells_and_bounds=True,
            original_expiry_range=[min(box.expires_at for box in packet.boxes), max(box.expires_at for box in packet.boxes)],
            existing_audit_artifact=reference['cell_audit_npz']))
    if packets['baseline']['boxes'] != original_bundle['packets']['baseline']['boxes']:
        raise ValueError('64-cuboid geometry differs from the original336-condition static probe bundle')
    nested = nested_selections(packets, record['snapshot'])
    if evaluate.array_digest(snapshot) != digest:
        raise ValueError('Canonical snapshot was mutated')
    bundle = dict(schema_version=1, cases=[dict(case_id=record['case_id'], layout_id=record['layout_id'],
        scenario=record['scenario'], snapshot=record['snapshot'], packets=packets,
        source_artifacts=record['source_artifacts'], robot_state_reference=record['robot_state_reference'])])
    proof = dict(case_id=record['case_id'], snapshot=record['snapshot'], original_extraction_clock=now,
        canonical_arrays_digest=digest, canonical_arrays_unchanged=True, original64_packet_matches=True,
        baseline_packing_and_selection_only=True, no_gt_inputs=True, budgets=audits, nested_subsets=nested,
        inputs={key: dict(path=str(getattr(args, key)), sha256=sha(getattr(args, key))) for key in
            ('manifest', 'primary_report', 'budget_report', 'primary_bundle')},
        snapshot_source_artifacts=record['source_artifacts'])
    return bundle, proof


def compact(raw, proof):
    nominal = [row for row in raw['rows'] if row['perturbation']['id'] == 'nominal']
    responses = []
    for row in nominal:
        offsets = [value for value in raw['rows'] if value['representation'] == row['representation']
            and value['posture'] == row['posture'] and value['perturbation']['id'] != 'nominal']
        empty = raw['empty_world_baselines'][row['posture']]['canonical_acceleration_rad_s2']
        responses.append(dict(representation=row['representation'], posture=row['posture'],
            registered_obstacles=row['registered_obstacles'], natural_force_norm=norm(row['natural_policy_force']),
            natural_metric_frobenius_norm=norm(row['natural_policy_metric']),
            collision_force_norm=row['collision_force_norm'], collision_metric_frobenius_norm=row['collision_metric_norm'],
            canonical_acceleration_norm=row['canonical_acceleration_norm'],
            canonical_acceleration_rad_s2=row['canonical_acceleration_rad_s2'],
            natural_policy_force=row['natural_policy_force'], collision_force_delta=row['collision_force_delta'],
            acceleration_delta_from_empty_norm=norm([a-b for a, b in zip(row['canonical_acceleration_rad_s2'], empty)]),
            acceleration_direction_cosine_to64=row['joint_acceleration_direction_cosine_to_baseline'],
            maximum_acceleration_change_for_1mm=max(value['perturbation_acceleration_delta_norm'] for value in offsets)))
    monotonic = {}
    for field in ('collision_force_norm', 'collision_metric_norm', 'canonical_acceleration_norm'):
        chains = []
        for posture in raw['postures']:
            for offset in ('nominal', 'x-', 'x+', 'y-', 'y+', 'z-', 'z+'):
                values = [next(row[field] for row in raw['rows'] if row['representation'] == NAMES[budget]
                    and row['posture'] == posture['id'] and row['perturbation']['id'] == offset) for budget in BUDGETS]
                chains.append(dict(posture=posture['id'], offset=offset, values=values, nondecreasing=nondecreasing(values)))
        monotonic[field] = dict(condition_chains=len(chains), nondecreasing_chains=sum(row['nondecreasing'] for row in chains),
            nominal=[row for row in chains if row['offset'] == 'nominal'],
            exceptions=[row for row in chains if not row['nondecreasing']])
    return dict(schema_version=1, status=raw['status'], case_id=proof['case_id'], snapshot=proof['snapshot'],
        query_conditions=len(raw['rows']), scope='Baseline observed-geometry BUDGET sensitivity only; installed public Lula CPU queries, no SimulationApp/physical motion or native candidate adoption.',
        budgets=proof['budgets'], nested_subsets=proof['nested_subsets'], canonical_arrays_digest=proof['canonical_arrays_digest'],
        canonical_arrays_unchanged=proof['canonical_arrays_unchanged'], original64_packet_matches=True,
        original_clock_preserved=proof['original_extraction_clock'] == proof['snapshot']['evaluated_at'],
        postures=raw['postures'], base_frame_reference=raw['base_frame_reference'], nominal_responses=responses,
        monotonicity=monotonic, source_hashes=raw['source_hashes'],
        limitations=['Only existing exact supported OCCUPIED cells are added; no FREE/UNKNOWN/expired geometry, padding, completion or TTL extension.',
            'Natural force/metric are policy terms, not Newtons, torques or physical inertia. Vector cancellation can make force norm nonmonotone.',
            'Canonical joint acceleration has the unchanged configured50rad/s² norm cap; direction and sensitivity can change even when its norm barely changes.',
            'Three actual recorded q postures use declared qdot=0; they are not claimed exposure-matched to the occupancy snapshot.',
            'Counterfactual1mm shifts remain static queries and never enter the native mapping/robot runtime.',
            '128 is a diagnostic capacity sensitivity, not a new native candidate, production limit or adoption recommendation.',
            'Larger natural terms or lower sensitivity do not certify target completion, continuous clearance or physical avoidance.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'outputs/obstacle-representation-v3/primary-final-snapshots.json')
    parser.add_argument('--primary-report', type=Path, default=ROOT/'outputs/obstacle-representation-v3/primary-offline/full_report.json')
    parser.add_argument('--budget-report', type=Path, default=ROOT/'outputs/obstacle-representation-v3/budget-offline/full_report.json')
    parser.add_argument('--primary-bundle', type=Path, default=ROOT/'outputs/obstacle-representation-v3/primary-offline/packet_bundle.json')
    parser.add_argument('--case-id', default=CASE)
    parser.add_argument('--postures', type=Path, default=ROOT/'docs/assets/obstacle_representation_v3/static_postures.json')
    parser.add_argument('--base-reference-result', type=Path, default=ROOT/'outputs/obstacle-representation-v3/pilot/A_current-empty-baseline-capture/result.json')
    parser.add_argument('--artifacts-dir', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args(argv)
    protected = {str(path.relative_to(ROOT)): sha(path) for path in PROTECTED}
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    bundle, proof = prepare(args)
    bundle_path = args.artifacts_dir/'packet_bundle.json'
    write(bundle_path, bundle)
    write(args.artifacts_dir/'exact_input_proof.json', proof)
    from scripts import probe_obstacle_representation_v3 as probe
    options = SimpleNamespace(packet_bundle=bundle_path, base_reference_result=args.base_reference_result,
        base_trace=None, case_ids=[args.case_id], max_cases=1,
        representations=[NAMES[budget] for budget in BUDGETS], perturbation_mm=1.)
    raw = probe.run(options, bundle, read(args.postures)['postures'])
    write(args.artifacts_dir/'raw_probe.json', raw)
    result = compact(raw, proof)
    result['frozen_sources_unchanged'] = protected == {str(path.relative_to(ROOT)): sha(path) for path in PROTECTED}
    result['protected_source_sha256'] = protected
    result['diagnostic_sha256'] = sha(__file__)
    result['raw_artifacts'] = {name: dict(path=str(args.artifacts_dir/name), sha256=sha(args.artifacts_dir/name))
        for name in ('packet_bundle.json', 'exact_input_proof.json', 'raw_probe.json')}
    if not result['frozen_sources_unchanged']:
        result['status'] = 'FAIL'
    write(args.result, result)
    if args.result.stat().st_size > 50_000:
        raise ValueError('Compact budget response exceeds explicit50KB bound; raw results retained')
    print(json.dumps(dict(status=result['status'], bytes=args.result.stat().st_size,
        query_conditions=result['query_conditions'], frozen_sources_unchanged=result['frozen_sources_unchanged'])))


if __name__ == '__main__':
    main()
