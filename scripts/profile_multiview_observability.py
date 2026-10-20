"""SDK-free observability latency/allocation measurements on saved captures.

Run only in a serialized CPU window, with Isaac and other numerical measurements
stopped. Candidate labels are frozen fitter predictions, never invented truth.
All depth/point inputs load before timing. Tracemalloc runs in a separate phase.
Optional full-provider comparison uses three distinct original native frames
per fresh source, without changing acquisition timestamps or render tokens.
These CPU replays do not measure native camera/render/update frequency.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import gc
import hashlib
import importlib
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from robot_skill_stack.world.perception_multiview.fitting import PrimitiveFit
from robot_skill_stack.world.perception_multiview.observability import assess
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from scripts.benchmark_multiview_robustness import ROOT, load_frames
from tests.support.multiview_iteration2 import SnapshotRig, distribution


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def hashes(paths):
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(paths))}


def source_paths():
    return list((ROOT/'robot_skill_stack/world/perception_multiview').glob('*.py'))+[Path(__file__)]


def load_candidates(path, scenes):
    manifest = json.loads(path.read_text())
    cases, input_files = [], [path]
    for record in manifest['rows']:
        if scenes and record['scene'] not in scenes:
            continue
        geometry = PrimitiveGeometry(record['shape'], record['confidence'], box_size=record['box_size'],
            yaw=record['yaw'], radius=record['radius'], length=record['length'], axis=record['axis'],
            metadata=deepcopy(record['metadata']))
        fit = PrimitiveFit(geometry, np.asarray(record['center']), np.asarray(record['size']))
        frame_paths = [Path(value) for value in record['frames']]
        frames = load_frames(Path('/'), [str(value) for value in frame_paths])
        with np.load(record['cloud'], allow_pickle=False) as data:
            points = data['points'].copy()
            views = tuple(data[key].copy() for key in data.files if key.startswith('view_'))
        cases.append(dict(record=record, fit=fit, points=points, frames=frames, views=views))
        input_files.extend([Path(record['cloud']), *frame_paths])
    return cases, input_files


def call(case):
    return assess(case['fit'], case['points'], case['frames'], view_clouds=case['views'])


def candidate_key(case):
    record = case['record']
    return dict(capture=record['capture'], scene=record['scene'], shape=record['shape'],
                cloud=record['cloud'], points=len(case['points']))


def allocation_peak(function):
    """Peak traced allocations above pre-call current usage, without timing."""
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    value = function()
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Retain the result until after the peak is read, as the caller does in the
    # provider. Input arrays were allocated before tracing and are excluded.
    del value
    return dict(additional_peak_traced_bytes=int(peak-before), retained_traced_bytes=int(retained-before))


def microbenchmark(cases, args):
    rows, memory = [], []
    groups = defaultdict(list)
    for index, case in enumerate(cases):
        groups[(case['record']['capture'], case['record']['scene'])].append(case)
        for _ in range(args.warmup):
            call(case)
        decisions = []
        for repeat in range(args.repeats):
            started = time.perf_counter()
            result = call(case)
            elapsed = 1000*(time.perf_counter()-started)
            decisions.append(result['observable'])
            rows.append(dict(candidate=index, repeat=repeat, **candidate_key(case), elapsed_ms=elapsed,
                             observable=result['observable'], reason=result['reason']))
        if len(set(decisions)) != 1:
            raise RuntimeError(f'Non-deterministic decision on cached candidate {index}')
    batch_rows = []
    for (capture, scene), batch in groups.items():
        for repeat in range(args.repeats):
            started = time.perf_counter()
            results = [call(case) for case in batch]
            elapsed = 1000*(time.perf_counter()-started)
            batch_rows.append(dict(capture=capture, scene=scene, repeat=repeat, candidates=len(batch),
                boxes=sum(case['record']['shape'] == 'cube' for case in batch), elapsed_ms=elapsed,
                observable=sum(result['observable'] for result in results)))
    # Allocation tracing is intentionally outside every timed sample.
    if args.allocations:
        for index, case in enumerate(cases):
            memory.append(dict(candidate=index, **candidate_key(case),
                               **allocation_peak(lambda case=case: call(case))))
        for (capture, scene), batch in groups.items():
            memory.append(dict(capture=capture, scene=scene, candidates=len(batch), kind='batch',
                **allocation_peak(lambda batch=batch: [call(case) for case in batch])))
    summary = {shape: distribution([row['elapsed_ms'] for row in rows if row['shape'] == shape])
               for shape in sorted({row['shape'] for row in rows})}
    return dict(candidate_rows=rows, captured_batch_rows=batch_rows, allocation_rows=memory,
                candidate_latency_ms_by_predicted_shape=summary,
                captured_batch_latency_ms=distribution([row['elapsed_ms'] for row in batch_rows]))


def load_provider_inputs(captures, scenes, max_scenes):
    records, inputs = [], []
    for capture in captures:
        capture = capture.resolve()
        path = capture/'capture_manifest.json'
        inputs.append(path)
        for record in json.loads(path.read_text())['scenes']:
            if scenes and record['key'] not in scenes:
                continue
            snapshots = record['frames'][-3:]
            if len(snapshots) < 3:
                raise ValueError(f'{record["key"]} lacks three distinct native confirmation frames')
            frames = [load_frames(capture, snapshot['files']) for snapshot in snapshots]
            for previous, current in zip(frames, frames[1:]):
                if len(previous) != len(current) or any(first.camera_id != second.camera_id or second.token[0] <= first.token[0]
                       or second.token[1] <= first.token[1] or second.acquired_at <= first.acquired_at
                       for first, second in zip(previous, current)):
                    raise ValueError(f'{record["key"]} has repeated/regressed original frames')
            records.append(dict(capture=str(capture), scene=record['key'], frames=frames))
            inputs.extend(capture/file for snapshot in snapshots for file in snapshot['files'])
            if max_scenes and len(records) >= max_scenes:
                return records, inputs
    return records, inputs


def provider_once(factory, record, scene, *, timed):
    rig = SnapshotRig()
    source = factory(rig, scene)
    try:
        for frames in record['frames'][:2]:
            rig.push(frames)
            source.observe()
        rig.push(record['frames'][2])
        started = time.perf_counter() if timed else None
        batch = source.observe()
        elapsed = 1000*(time.perf_counter()-started) if timed else None
        return dict(elapsed_ms=elapsed, published=len(batch.observations), diagnostics=dict(source.diagnostics))
    finally:
        source.close()


def provider_benchmark(records, args):
    sys.path.insert(0, str(args.baseline_pythonpath.resolve()))
    baseline = importlib.import_module(f'{args.baseline_package}.provider').MultiViewProvider
    factories = dict(baseline=baseline, current=MultiViewProvider)
    scene = load_scene_config(ROOT/'config/scenes/multiview_playground.toml').scene
    rows, memory = [], []
    for record in records:
        for repeat in range(args.provider_repeats):
            # Alternate order while keeping all work strictly sequential.
            methods = ('baseline', 'current') if repeat % 2 == 0 else ('current', 'baseline')
            for method in methods:
                measured = provider_once(factories[method], record, scene, timed=True)
                rows.append(dict(capture=record['capture'], scene=record['scene'], method=method,
                                 repeat=repeat, **measured))
        if args.allocations:
            for method, factory in factories.items():
                memory.append(dict(capture=record['capture'], scene=record['scene'], method=method,
                    **allocation_peak(lambda factory=factory: provider_once(factory, record, scene, timed=False))))
        print(json.dumps(dict(progress='provider scene complete', scene=record['scene'])), flush=True)
    summary = {method: distribution([row['elapsed_ms'] for row in rows if row['method'] == method])
               for method in factories}
    return dict(rows=rows, allocation_rows=memory, final_frame_latency_ms_by_method=summary)


def main(args):
    if np.__version__ != '1.26.4':
        raise RuntimeError('Use the unchanged Isaac Python wrapper with NumPy 1.26.4')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'result.json').exists():
        raise FileExistsError('Choose a fresh output directory; preserve previous measurements')
    sources = source_paths()
    start = hashes(sources)
    cases, input_paths = load_candidates(args.cached_manifest, args.scene)
    if not cases:
        raise ValueError('No cached candidates matched the requested scene filter')
    records = []
    if args.provider_repeats:
        captures = args.provider_capture or sorted({Path(case['record']['capture']) for case in cases})
        records, provider_paths = load_provider_inputs(captures, args.scene, args.max_scenes)
        input_paths.extend(provider_paths)
    inputs = hashes(input_paths)
    write_json(args.output/'provenance_start.json', dict(sources=start, inputs=inputs))
    micro = microbenchmark(cases, args)
    write_json(args.output/'helper_measurements.json', micro)
    providers = provider_benchmark(records, args) if records else None
    if providers is not None:
        write_json(args.output/'provider_measurements.json', providers)
    end = hashes(sources)
    write_json(args.output/'provenance_end.json', dict(sources=end))
    result = dict(status='PASS' if start == end else 'FAIL', kind='serialized_saved_frame_observability_cpu_profile',
        candidates=len(cases), captured_candidate_batches=len({(case['record']['capture'], case['record']['scene']) for case in cases}),
        repeats=args.repeats, warmup=args.warmup, provider_repeats=args.provider_repeats,
        source_hashes_stable=start == end,
        helper_latency_ms_by_predicted_shape=micro['candidate_latency_ms_by_predicted_shape'],
        helper_captured_batch_latency_ms=micro['captured_batch_latency_ms'],
        provider_final_frame_latency_ms=None if providers is None else providers['final_frame_latency_ms_by_method'],
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        memory_scope='Separate tracemalloc phase; Python/NumPy traced allocations above loaded inputs, not GPU/native SDK or isolated process RSS',
        scope='Offline CPU only; no native rendering, live source-age or update-frequency claim. Repeated static geometry is not independent statistical evidence.',
        robot_or_backend_commands=[])
    write_json(args.output/'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--allocations', action='store_true')
    parser.add_argument('--provider-repeats', type=int, default=0)
    parser.add_argument('--provider-capture', type=Path, nargs='*')
    parser.add_argument('--baseline-pythonpath', type=Path)
    parser.add_argument('--baseline-package', default='baseline_multiview')
    parser.add_argument('--scene', action='append', default=[])
    parser.add_argument('--max-scenes', type=int)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.provider_repeats < 0:
        parser.error('Repeat counts must be positive (provider repeats may be zero); warmup must be nonnegative')
    if args.provider_repeats and not args.baseline_pythonpath:
        parser.error('--provider-repeats requires --baseline-pythonpath')
    if args.max_scenes is not None and args.max_scenes < 1:
        parser.error('--max-scenes must be positive')
    try:
        main(args)
    except Exception as error:
        # Retain a machine-readable failed experiment, while never replacing an
        # earlier result when the fresh-directory precondition itself fails.
        if not (args.output/'result.json').exists():
            write_json(args.output/'result.json', dict(status='FAIL', error=f'{type(error).__name__}: {error}',
                kind='serialized_saved_frame_observability_cpu_profile'))
        raise
