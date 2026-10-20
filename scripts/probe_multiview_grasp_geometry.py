"""Replay exact saved selection frames through unchanged model/planner queries.

The explicit replay clock follows each original acquisition timestamp. Inputs
are never retimestamped. This isolates functional geometry decisions; live
freshness, rendering, IK and throughput require the separate native benchmark.
No motion command or Pick request is submitted.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from scripts.benchmark_multiview_grasp_readiness import match_objects, plain_values
from scripts.benchmark_multiview_robustness import ROOT, load_frames
from tests.support.multiview_iteration2 import SnapshotRig, object_from_payload
from tests.support.multiview_readiness_research import evaluate_grasp


def main(args):
    if np.__version__ != '1.26.4':
        raise RuntimeError('Use the established Isaac Python wrapper with NumPy 1.26.4')
    sys.path.insert(0, str(args.baseline_pythonpath.resolve()))
    baseline = importlib.import_module('baseline_multiview.provider').MultiViewProvider
    cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
    manifest = json.loads((args.capture/'capture_manifest.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    source_files = tuple((ROOT/'robot_skill_stack/world/perception_multiview').glob('*.py'))
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    (args.output/'provenance_start.json').write_text(json.dumps(source_hashes, indent=2))
    rows, errors = [], []
    for record in manifest['scenes']:
        rigs = {name: SnapshotRig() for name in ('baseline', 'improved')}
        sources = dict(baseline=baseline(rigs['baseline'], cfg.scene),
                       improved=MultiViewProvider(rigs['improved'], cfg.scene))
        updaters = {name: WorldModelUpdater(WorldModel(), source, stale_object_ttl=2.5)
                    for name, source in sources.items()}
        try:
            for index, snapshot in enumerate(record['frames']):
                frames = load_frames(args.capture, snapshot['files'])
                objects = tuple(object_from_payload(obj) for obj in snapshot['objects'])
                replay_time = min(frame.acquired_at for frame in frames)
                for name, updater in updaters.items():
                    rigs[name].push(frames)
                    with patch('time.monotonic', return_value=replay_time):
                        updater.update()
                        if index != len(record['frames'])-1:
                            continue
                        model = updater.world_model
                        matches, extras = match_objects(objects, model.visible_objects())
                        for truth in objects:
                            observed = matches.get(truth.name)
                            row = evaluate_grasp(truth, observed, model, objects)
                            evidence = None
                            if observed is not None:
                                g = observed.geometry
                                evidence = dict(shape=g.shape.value, radius=g.radius, length=g.length,
                                    axis=g.axis, box_size=g.box_size, yaw=g.yaw, center=observed.pose.position,
                                    graspable=observed.graspable, metadata=g.metadata)
                            rows.append(dict(row, scene=record['key'], method=name,
                                             false_positives=extras, geometry_evidence=evidence))
            (args.output/'rows.json').write_text(json.dumps(plain_values(rows), indent=2))
        except Exception as error:
            errors.append(dict(scene=record['key'], error=f'{type(error).__name__}: {error}'))
            (args.output/'runtime_errors.json').write_text(json.dumps(errors, indent=2))
            raise
        finally:
            for updater in updaters.values():
                updater.stop()
    summary = {}
    for name in sources:
        selected = [row for row in rows if row['method'] == name]
        summary[name] = dict(expected=len(selected), detected=sum(row['detected'] for row in selected),
            truth_feasible=sum(row['truth_feasible'] for row in selected),
            estimated_feasible=sum(row['estimated_feasible'] for row in selected),
            false_feasible=sum(row['false_feasible'] for row in selected),
            false_infeasible=sum(row['false_infeasible'] for row in selected),
            clearance_violations=sum(bool(row['proposed_truth_checks'] and row['proposed_truth_checks']['clearance_violation'])
                                     for row in selected),
            width_violations=sum(bool(row['proposed_truth_checks'] and row['proposed_truth_checks']['width_violation'])
                                 for row in selected))
    result = dict(kind='functional_saved_frame_geometry_replay', input=str(args.capture),
                  replay_clock='Original acquisition timestamps; input frames unchanged',
                  exclusions='No live latency, source-age, rendering, IK or throughput claims',
                  summary=summary, runtime_errors=errors, robot_commands=0)
    end_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    (args.output/'provenance_end.json').write_text(json.dumps(end_hashes, indent=2))
    result['production_sources_unchanged'] = source_hashes == end_hashes
    (args.output/'result.json').write_text(json.dumps(plain_values(result), indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--baseline-pythonpath', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
