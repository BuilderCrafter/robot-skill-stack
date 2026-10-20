"""Separate selection ablation: partial views of physically oversized boxes.

Wrap the unchanged native grasp-readiness runner. Truth constructs static
occluders and audits saved depth only; it is never supplied to perception.
All targets exceed the existing gripper's 75 mm opening on both horizontal
axes. No Pick, trajectory, gripper command, or production tuning is performed.

Example (root-owned native session):
  ./run_isaac_python.sh scripts/benchmark_multiview_partial_grasp.py \
    --result outputs/multiview-readiness-partial/result.json --no-ik
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, project_depth
from scripts import benchmark_multiview_grasp_readiness as runner
from scripts.benchmark_multiview_robustness import ROOT, load_frames
from tests.support.multiview_iteration2 import (
    RobustnessScene, _blocker, box, object_payload, surface_memberships,
)
from tests.support.multiview_readiness_research import EYES
from tests.support.multiview_research import ResearchObject
from tests.support.results import fail_result


def partial_scenes(split='selection'):
    """Fixed research fixtures; final width selection/holdout stay unchanged."""
    if split != 'selection':
        raise ValueError('Partial-occlusion ablation is selection-only')
    scenes = []
    for width in (80, 90):
        name = f'partial_box_reference_{width}'
        target = box(f'{name}_target', (.445, -.01), (width/1000, width/1000, .05))
        scenes.append(RobustnessScene(name, split, 825000+width, (target,),
            note='Both physical horizontal dimensions exceed75mm; full-view reference.'))
    reference = scenes[-1].objects[0]
    # This vertical panel clips A's measured Y extent while B sees the target.
    a_panel = _blocker(reference, EYES['selection'][0], partial=True)
    # A static elevated plate clips B's X extent; A retains the full span.
    b_roof = ResearchObject('roof_B', 'cube', np.array([.36, -.01, .085]),
                            np.array([.18, .30, .014]))
    # A joined cover assembly hides the same Y extent from both viewpoints.
    roof = ResearchObject('roof_both', 'cube', np.array([.445, .04, .085]),
                          np.array([.30, .17, .014]))
    curtain = ResearchObject('curtain_B', 'cube', np.array([.37, -.015, .105]),
                             np.array([.012, .19, .21]))
    for name, blockers, note in (
        ('partial_box_A', (a_panel,), 'A Y-extent clipped; B full-view control.'),
        ('partial_box_B', (b_roof,), 'B X-extent clipped; A full-view control.'),
        ('partial_box_both', (roof, curtain),
         'Static cover assembly clips the same Y extent in both cameras; panels may join each other, never the target.'),
    ):
        target = replace(reference, name=f'{name}_target')
        clutter = tuple(replace(obj, name=f'{name}_occluder_{index}')
                        for index, obj in enumerate(blockers))
        scenes.append(RobustnessScene(name, split, 825100+len(scenes), (target,), clutter,
            note=note+' Measured-depth span preconditions must pass before interpreting planner outcomes.'))
    return scenes


def local_extent(points, target):
    points = np.asarray(points, float)
    if not len(points):
        return dict(points=0, span_mm=None, lower_mm=None, upper_mm=None)
    c, s = np.cos(target.yaw), np.sin(target.yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    local = (points-target.center) @ rotation
    return dict(points=len(points), span_mm=np.ptp(local, axis=0)*1000,
                lower_mm=local.min(0)*1000, upper_mm=local.max(0)*1000)


def fixture_manifest(scenes):
    return dict(kind='separate_partial_oversized_box_selection_ablation',
        viewpoints=EYES['selection'], source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection_only=True, frozen_width_fixtures_unchanged=True,
        scoring='Oracle exclusive surface membership is used only after automatic perception and planning.',
        criteria=dict(minimum_raw_target_points_per_camera=30, minimum_span_loss_mm=10,
                      full_reference_tolerance_mm=5),
        scenes=[dict(key=scene.key, note=scene.note,
                     objects=[object_payload(obj) for obj in scene.objects],
                     clutter=[object_payload(obj) for obj in scene.clutter]) for scene in scenes])


def audit_capture(output, scenes):
    """Preserve invalid/all-hidden fixtures instead of scoring them as misses."""
    cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
    capture = json.loads((output/'capture_manifest.json').read_text())
    records = {record['key']: record for record in capture['scenes']}
    measured, failures = {}, []
    for scene in scenes:
        record = records.get(scene.key)
        if record is None:
            failures.append(dict(scene=scene.key, reason='missing_capture_record'))
            continue
        target, solids = scene.objects[0], scene.objects+scene.clutter
        values = []
        for snapshot in record['frames']:
            index = snapshot['index']
            if index not in (2, 3):
                continue
            raw = []
            for frame in load_frames(output, snapshot['files']):
                cloud = crop_workspace(project_depth(frame, stride=1)[0], cfg.scene)
                raw.append(local_extent(cloud[surface_memberships(cloud, solids) == 0], target))
            path = output/'captures'/scene.key/f'workspace_frame_{index:02d}.npz'
            with np.load(path, allow_pickle=False) as archive:
                voxel = [local_extent(archive[f'camera_{i}'][
                    surface_memberships(archive[f'camera_{i}'], solids) == 0], target) for i in range(2)]
                cloud = archive['fused']
                fused = local_extent(cloud[surface_memberships(cloud, solids) == 0], target)
            values.append(dict(frame_index=index, raw_cameras=raw,
                               voxel_cameras=voxel, fused_voxel=fused))
        measured[scene.key] = values
    for scene in scenes:
        entries = measured.get(scene.key, [])
        if len(entries) != 2:
            failures.append(dict(scene=scene.key, reason='need_both_final_native_frames'))
        for entry in entries:
            index, raw = entry['frame_index'], entry['raw_cameras']
            reasons = []
            if any(view['points'] < 30 for view in raw) or entry['fused_voxel']['points'] < 30:
                reasons.append('target_missing_or_insufficient_surface_samples')
            elif 'reference' in scene.name:
                true_span = scene.objects[0].size[:2]*1000
                if any(np.any(np.asarray(view['span_mm'])[:2] < true_span-5) for view in raw):
                    reasons.append('reference_does_not_measure_full_horizontal_extent')
            else:
                reference = next((row for row in measured.get('selection_partial_box_reference_90', [])
                                  if row['frame_index'] == index), None)
                if reference is None or any(view['span_mm'] is None for view in reference['raw_cameras']) or reference['fused_voxel']['span_mm'] is None:
                    reasons.append('missing_or_unobserved_90mm_reference')
                else:
                    if scene.name.endswith('_both'):
                        losses = [reference['raw_cameras'][i]['span_mm'][1]-raw[i]['span_mm'][1]
                                  for i in range(2)]
                        fused_loss = reference['fused_voxel']['span_mm'][1]-entry['fused_voxel']['span_mm'][1]
                        entry['same_y_extent_loss_mm'] = losses
                        entry['fused_y_extent_loss_mm'] = fused_loss
                        if min(losses) < 10 or fused_loss < 10:
                            reasons.append('both_views_do_not_clip_same_physical_extent')
                    else:
                        blocked, axis = (0, 1) if scene.name.endswith('_A') else (1, 0)
                        loss = reference['raw_cameras'][blocked]['span_mm'][axis]-raw[blocked]['span_mm'][axis]
                        clear_loss = np.asarray(reference['raw_cameras'][1-blocked]['span_mm'])[:2]-np.asarray(
                            raw[1-blocked]['span_mm'])[:2]
                        entry['blocked_extent_loss_mm'] = loss
                        entry['clear_view_extent_loss_mm'] = clear_loss
                        if loss < 10:
                            reasons.append('occlusion_reduces_counts_without_required_extent_loss')
                        if np.max(clear_loss) > 5:
                            reasons.append('independent_clear_view_extent_not_retained')
            entry['valid_fixture'] = not reasons
            failures.extend(dict(scene=scene.key, frame_index=index, reason=reason) for reason in reasons)
    rows = json.loads((output/'rows.json').read_text())
    if any(row['truth_feasible'] for row in rows):
        failures.append(dict(reason='full_ground_truth_oversized_box_unexpectedly_feasible'))
    result = dict(kind='partial_oversized_box_measured_visibility_audit',
        valid=not failures, failures=failures, measurements=measured,
        note='Invalid fixtures are retained and cannot establish readiness; planner rows remain unchanged.')
    runner.write_json(output/'partial_visibility_audit.json', result)
    return result


def main(args, app):
    output = Path(args.artifacts_dir).resolve()
    scenes = partial_scenes(args.split)
    if args.limit:
        scenes = scenes[:args.limit]
    runner.write_json(output/'partial_fixture_manifest.json', fixture_manifest(scenes))
    with patch.object(runner, 'grasp_width_scenes', lambda split: scenes):
        code = runner.main(args, app)
    audit = audit_capture(output, scenes)
    result = json.loads(Path(args.result).read_text())
    result['metrics']['partial_occlusion_fixture_valid'] = audit['valid']
    result['metrics']['partial_occlusion_fixture_failures'] = len(audit['failures'])
    result['metrics']['kind'] = 'separate_selection_partial_oversized_box_ablation'
    if not audit['valid']:
        result['status'] = 'FAIL'
        code = 1
    runner.write_json(args.result, result)
    return code


def cli():
    parser = runner.parser()
    parser.description = __doc__
    args = parser.parse_args()
    if args.split != 'selection':
        raise SystemExit('This additional ablation is selection-only; frozen holdout fixtures are unchanged')
    if args.baseline_only and not args.baseline_pythonpath:
        raise SystemExit('--baseline-only needs --baseline-pythonpath')
    if args.limit is not None and args.limit < 1:
        raise SystemExit('--limit must be positive')
    args.artifacts_dir = args.artifacts_dir or str(Path(args.result).parent/'partial_grasp_ablation')
    args.replay = None
    app, code = None, 1
    try:
        if np.__version__ != '1.26.4':
            raise RuntimeError(f'Expected NumPy1.26.4 before SimulationApp, got{np.__version__}')
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': False})
        code = main(args, app)
    except Exception as error:
        fail_result(args.result, error)
    finally:
        provenance = runner.source_provenance(args)
        provenance['sha256'][str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        runner.write_json(Path(args.artifacts_dir)/'partial_provenance_end.json', provenance)
        if app is not None:
            app.close()
    raise SystemExit(code)


if __name__ == '__main__':
    cli()
