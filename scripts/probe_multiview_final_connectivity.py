"""Research-only 4 mm first / selective 4.5 mm seed retry on saved depth.

The original refinement body, guards and shared numerical budget are reused.
No production module is edited. Oracle truth is consumed only after automatic
current-frame candidate generation. This compares geometric candidates before
three-frame identity confirmation; it makes no native-rate or freshness claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception_multiview import provider, segmentation
from robot_skill_stack.world.perception_multiview.fitting import SurfaceFitter
from robot_skill_stack.world.perception_multiview.provider import _FitTimer
from scripts.benchmark_multiview_robustness import load_frames
from tests.support.multiview_final_evaluation import plain
from tests.support.multiview_iteration2 import SnapshotRig, object_from_payload, surface_memberships


class SelectiveSeedRetry:
    """Record the promoted implementation; no duplicate retry policy lives here."""
    def __init__(self):
        self.events = []
        self.original = segmentation.refine_cluster
        self.metric = segmentation._metric_components
        self.view_evidence = segmentation._partition_view_evidence

    def __call__(self, points, fitter, *, support_plane_z=0., parent_fit=None, frames=()):
        graphs, verdicts = [], []
        def components(cloud, **kwargs):
            groups = self.metric(cloud, **kwargs)
            graphs.append(dict(radius_m=kwargs.get('radius', .004), seeds=[len(p) for p in groups]))
            return groups
        def evidence(*args, **kwargs):
            result = self.view_evidence(*args, **kwargs)
            verdicts.append(result)
            return result
        before = getattr(fitter, 'remaining', None)
        with patch.object(segmentation, '_metric_components', components), \
                patch.object(segmentation, '_partition_view_evidence', evidence):
            result = self.original(points, fitter, support_plane_z=support_plane_z,
                                   parent_fit=parent_fit, frames=frames)
        retried = any(graph['radius_m'] == .0045 for graph in graphs)
        self.events.append(dict(points=len(points), graphs=graphs, view_verdicts=verdicts,
            budget_before=before, budget_after=getattr(fitter, 'remaining', None),
            retried=retried, retry_changed_partition=retried and len(result)>1,
            final_partitions=len(result)))
        return result


def regression_checks():
    from tests.unit import test_multiview_segmentation as simple
    from tests.unit import test_multiview_segmentation_readiness as readiness
    retry = SelectiveSeedRetry()
    suite = unittest.TestSuite()
    names = (
        (simple, 'SegmentationTests', ('test_nearby_spheres_split_only_with_supported_separate_models',
            'test_disconnected_sparse_unknown_patches_do_not_invent_objects',
            'test_supported_primitive_can_be_recovered_beside_dense_large_background')),
        (readiness, 'SegmentationReadinessTests', (
            'test_sparse_current_islands_complete_distinct_sphere_clouds',
            'test_missing_band_does_not_create_two_known_boxes',
            'test_measured_table_gap_supports_distinct_solids',
            'test_valid_occupied_bridge_supports_one_intact_solid',
            'test_invalid_depth_is_neither_a_gap_nor_an_occupied_bridge',
            'test_insufficient_shared_budget_does_no_partial_numerical_work',
            'test_fragmented_elevated_clutter_has_one_global_numerical_budget',
            'test_excessive_dense_fragments_skip_all_child_fits')),
    )
    for module, cls, cases in names:
        suite.addTests(getattr(module, cls)(name) for name in cases)
    with patch.object(simple, 'refine_cluster', retry), patch.object(readiness, 'refine_cluster', retry):
        outcome = unittest.TestResult()
        suite.run(outcome)
    # A one-component cloud must make exactly the original metric graph call.
    dense = readiness.box_surface()
    calls = []
    original_metric = retry.metric
    def metric(cloud, **kwargs):
        calls.append(kwargs.get('radius', .004))
        return original_metric(cloud, **kwargs)
    retry.metric = metric
    retry(dense, _FitTimer(SurfaceFitter()))
    one_component = calls == [.004]
    failures = [dict(test=str(test), traceback=traceback) for test,traceback in outcome.failures]
    errors = [dict(test=str(test), traceback=traceback) for test,traceback in outcome.errors]
    return dict(tests=outcome.testsRun, failures=failures, errors=errors,
        one_component_graph_radii=calls, one_component_graph_not_repeated=one_component,
        passed=outcome.wasSuccessful() and one_component, events=retry.events)


def score_candidates(record, snapshot, source):
    truths = [object_from_payload(obj) for obj in snapshot.get('objects', record['objects'])]
    candidates = [track.detection for track in source.tracker.tracks.values() if track.visible]
    matched, assigned = {}, set()
    for distance, ti, ci in sorted((float(np.linalg.norm(candidate.fit.center-truth.center)), ti, ci)
            for ti, truth in enumerate(truths) for ci, candidate in enumerate(candidates)):
        if distance < .04 and ti not in matched and ci not in assigned:
            matched[ti] = ci
            assigned.add(ci)
    rows, raw = [], []
    for ti, truth in enumerate(truths):
        candidate = candidates[matched[ti]] if ti in matched else None
        shape = 'missing' if candidate is None else candidate.fit.geometry.shape.value
        rows.append(dict(object=truth.name, truth=truth.shape, predicted=shape,
            correct=shape == truth.shape, detected=candidate is not None))
    for candidate in candidates:
        fit, points = candidate.fit, candidate.points
        geometry = fit.geometry
        labels = surface_memberships(points, truths)
        counts = [int(np.count_nonzero(labels == ti)) for ti in range(len(truths))]
        significant = [n >= 8 and n/max(1,len(points)) >= .10 for n in counts]
        raw.append(dict(shape=geometry.shape.value, center=fit.center, points=len(points),
            scores=geometry.scores, metadata=geometry.metadata, truth_surface_counts=counts,
            mixed=sum(significant) > 1, graspable=geometry.shape.value != 'unknown' and
                bool(geometry.metadata.get('parameter_observable', False))))
    return rows, raw


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'result.json').exists():
        raise FileExistsError('Use a fresh output directory')
    modules = list(Path(provider.__file__).parent.glob('*.py'))
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in modules}
    start = hashes()
    checks = regression_checks()
    manifest = json.loads((args.capture/'capture_manifest.json').read_text())
    baseline_rows = json.loads((args.baseline_evaluation/'rows.json').read_text())
    baseline_samples = json.loads((args.baseline_evaluation/'samples.json').read_text())
    scene = load_scene_config('config/scenes/multiview_playground.toml').scene
    all_rows, all_raw, samples = [], [], []
    for record in manifest['scenes']:
        if args.layout and record['layout']['name'] not in args.layout:
            continue
        snapshots = record['frames'][-1:]
        for snapshot in snapshots:
            frames = load_frames(args.capture, snapshot['files'])
            rig, retry = SnapshotRig(), SelectiveSeedRetry()
            source = provider.MultiViewProvider(rig, scene)
            rig.push(frames)
            try:
                with patch.object(provider, 'refine_cluster', retry), patch('time.monotonic', return_value=min(
                        frame.acquired_at for frame in frames)):
                    source.observe()
                rows, raw = score_candidates(record, snapshot, source)
                previous = [row for row in baseline_rows if row['trial'] == record['key'] and
                    row['frame_index'] == snapshot['index'] and row['method'] == 'current_fused']
                old = {row['object']:row for row in previous}
                for row in rows:
                    row.update(layout=record['layout']['name'], scene_key=record['scene_key'],
                        frame_index=snapshot['index'], baseline_predicted=old[row['object']]['predicted'],
                        baseline_correct=old[row['object']]['correct'],
                        changed=row['predicted'] != old[row['object']]['predicted'])
                context = dict(layout=record['layout']['name'], scene_key=record['scene_key'], frame_index=snapshot['index'])
                all_rows.extend(rows)
                all_raw.extend(dict(item, **context) for item in raw)
                old_sample = next(row for row in baseline_samples if row['trial']==record['key'] and
                    row['frame_index']==snapshot['index'] and row['method']=='current_fused')
                samples.append(dict(**context, diagnostics=source.diagnostics,
                    baseline_diagnostics=old_sample['diagnostics'], events=retry.events))
            finally:
                source.close()
    summaries = {}
    for layout in sorted({row['layout'] for row in all_rows}):
        subset = [row for row in all_rows if row['layout']==layout]
        candidate_rows = [row for row in all_raw if row['layout']==layout]
        diagnostics = [row['diagnostics'] for row in samples if row['layout']==layout]
        summaries[layout] = dict(expected=len(subset), baseline_correct=sum(r['baseline_correct'] for r in subset),
            retry_correct=sum(r['correct'] for r in subset),
            changed_rows=sum(r['changed'] for r in subset),
            improved=sum(r['correct'] and not r['baseline_correct'] for r in subset),
            regressed=sum(r['baseline_correct'] and not r['correct'] for r in subset),
            unknown=sum(r['predicted']=='unknown' for r in subset), missing=sum(not r['detected'] for r in subset),
            mixed=sum(r['mixed'] for r in candidate_rows),
            known_mixed=sum(r['mixed'] and r['shape']!='unknown' for r in candidate_rows),
            graspable_mixed=sum(r['mixed'] and r['graspable'] for r in candidate_rows),
            max_numerical_fits=max(r['numerical_fits'] for r in diagnostics),
            fit_budget_exhaustions=sum(r['fit_budget_exhaustions'] for r in diagnostics))
    end = hashes()
    result = dict(status='PASS' if start==end and checks['passed'] else 'FAIL',
        kind='research_selective_seed_retry_geometric_candidates',
        rules=dict(first_radius_m=.004, retry_radius_m=.0045, minimum_retained_first_seeds=2,
            maximum_retained_first_seeds=8, original_guards_unchanged=True, shared_maximum_fits=24,
            retry_explicit_unknown=False, retry_occupied_bridge=False),
        source_hashes_stable=start==end, source_hashes=start, summaries=summaries,
        regression_checks={k:v for k,v in checks.items() if k!='events'},
        retried_clusters=sum(event['retried'] for row in samples for event in row['events']),
        changed_partitions=sum(event['retry_changed_partition'] for row in samples for event in row['events']),
        all_fit_budgets_respected=all(row['diagnostics']['numerical_fits']<=24 for row in samples),
        scope='One original saved frame per scene, candidates before tracker confirmation; no rate, temporal, native or safety readiness claim')
    for name,value in (('rows',all_rows),('candidates',all_raw),('samples',samples),
                       ('regression_checks',checks),('result',result)):
        (args.output/(name+'.json')).write_text(json.dumps(plain(value),indent=2)+'\n')
    print(json.dumps(plain(result),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--baseline-evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--layout',action='append',default=[])
    main(parser.parse_args())
