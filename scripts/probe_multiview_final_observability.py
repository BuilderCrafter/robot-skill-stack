"""Research-only equivalent pixel-set optimization; production stays untouched.

The variant encodes integer pixel pairs for one-dimensional deduplication and
reuses constant offsets. It retains every original pixel, ribbon width, witness,
threshold and evidence field. Generated candidate source and its exact diff are
saved for review. Optional CPU timing requires a quiet serialized window.
"""
import argparse
from dataclasses import replace
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from robot_skill_stack.world.perception_multiview import observability
from scripts.profile_multiview_observability import load_candidates, write_json
from tests.support.multiview_iteration2 import distribution


_PIXEL_HELPER = '''
def _unique_pixels(raw):
    low = raw.min(0)
    high = raw.max(0)
    width = int(high[0])-int(low[0])+1
    height = int(high[1])-int(low[1])+1
    if width*height > np.iinfo(np.int64).max:
        return np.unique(raw, axis=0)
    codes = (raw[:, 1]-low[1])*width+raw[:, 0]-low[0]
    codes = np.unique(codes)
    return np.column_stack((codes % width+low[0], codes // width+low[1]))

'''


def candidate_module(output, path):
    original = path.read_text()
    source = original
    replacements = (
        ('_EDGE_SAMPLES = 48\n', '''_EDGE_SAMPLES = 48
_RIBBON_OFFSETS = np.array([(x, y) for x in range(-_PIXEL_RADIUS, _PIXEL_RADIUS+1)
                                for y in range(-_PIXEL_RADIUS, _PIXEL_RADIUS+1)])
_PIXEL_NEIGHBORS = np.array([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y])
_EDGE_POSITIONS = np.linspace(-.4, .4, _EDGE_SAMPLES)
'''),
        ('np.linspace(-.4, .4, _EDGE_SAMPLES)*size[other]', '_EDGE_POSITIONS*size[other]'),
        ('''offsets = np.array([(x, y) for x in range(-_PIXEL_RADIUS, _PIXEL_RADIUS+1)
                        for y in range(-_PIXEL_RADIUS, _PIXEL_RADIUS+1)])''', 'offsets = _RIBBON_OFFSETS'),
        ('np.unique((np.rint(projected).astype(int)[:, None]+offsets).reshape(-1, 2), axis=0)',
         '_unique_pixels((np.rint(projected).astype(int)[:, None]+offsets).reshape(-1, 2))'),
        ('neighbors = np.array([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y])',
         'neighbors = _PIXEL_NEIGHBORS'),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError(f'Expected exactly one optimization anchor: {before}')
        source = source.replace(before, after)
    source = source.replace('def _view_boundary(', _PIXEL_HELPER+'def _view_boundary(', 1)
    candidate = output/'candidate_observability.py'
    candidate.write_text(source)
    (output/'candidate.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), source.splitlines(True), fromfile=str(path), tofile=str(candidate))))
    spec = importlib.util.spec_from_file_location('_baseline_observability_probe', path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    return baseline, hashlib.sha256(original.encode()).hexdigest(), hashlib.sha256(source.encode()).hexdigest()


def compare(first, second, path='', state=None):
    state = state if state is not None else dict(max_float_difference=0., mismatches=[])
    if isinstance(first, dict) and isinstance(second, dict):
        if set(first) != set(second):
            state['mismatches'].append(dict(path=path, reason='different_keys'))
        for key in first.keys() & second.keys():
            compare(first[key], second[key], f'{path}/{key}', state)
    elif isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            state['mismatches'].append(dict(path=path, reason='different_lengths'))
        for index, (a, b) in enumerate(zip(first, second)):
            compare(a, b, f'{path}/{index}', state)
    elif isinstance(first, (float, np.floating)) and isinstance(second, (float, np.floating)):
        delta = abs(float(first)-float(second))
        state['max_float_difference'] = max(state['max_float_difference'], delta)
        if not np.isfinite(delta) or delta > 1e-12:
            state['mismatches'].append(dict(path=path, baseline=float(first), candidate=float(second)))
    elif first != second:
        state['mismatches'].append(dict(path=path, baseline=first, candidate=second))
    return state


def frame_variants(frames, seed):
    rng = np.random.default_rng(seed)
    yield 'clean', frames
    yield 'depth_noise', tuple(replace(frame, depth=frame.depth+rng.normal(0., .0005, frame.depth.shape)) for frame in frames)
    yield 'dropout', tuple(replace(frame, depth=np.where(rng.random(frame.depth.shape) < .35, np.nan, frame.depth)) for frame in frames)
    perturbed = []
    for frame in frames:
        transform = frame.world_from_camera.copy()
        transform[:3, 3] += [.0005, -.0003, .0002]
        perturbed.append(replace(frame, world_from_camera=transform))
    yield 'calibration', tuple(perturbed)
    outside = []
    for frame in frames:
        intrinsics = frame.intrinsics.copy()
        intrinsics[0, 2] += 10000
        outside.append(replace(frame, intrinsics=intrinsics))
    yield 'outside_fov', tuple(outside)


def call(module, case, frames=None):
    return module.assess(case['fit'], case['points'], case['frames'] if frames is None else frames,
                         view_clouds=case['views'])


def main(args):
    if np.__version__ != '1.26.4':
        raise RuntimeError('Use the established wrapper with NumPy 1.26.4')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'result.json').exists():
        raise FileExistsError('Choose a fresh output directory to preserve previous measurements')
    baseline, original_hash, candidate_hash = candidate_module(args.output, args.baseline_source)
    production_hash = hashlib.sha256(Path(observability.__file__).read_bytes()).hexdigest()
    if production_hash != candidate_hash:
        raise RuntimeError('Production helper differs from the exact generated optimization')
    cases, _ = load_candidates(args.cached_manifest, [])
    checks, timing = [], []
    for ci, case in enumerate(cases):
        for variant, frames in frame_variants(case['frames'], args.seed+ci):
            first, second = call(baseline, case, frames), call(observability, case, frames)
            check = compare(first, second)
            checks.append(dict(candidate=ci, scene=case['record']['scene'], shape=case['record']['shape'],
                variant=variant, baseline_observable=first['observable'], candidate_observable=second['observable'], **check))
    write_json(args.output/'equivalence_rows.json', checks)
    if args.timing_repeats:
        for ci, case in enumerate(cases):
            if case['record']['shape'] != 'cube':
                continue
            for module in (baseline, observability):
                for _ in range(args.warmup):
                    call(module, case)
            for repeat in range(args.timing_repeats):
                order = (('baseline', baseline), ('candidate', observability))
                for method, module in (order if repeat % 2 == 0 else order[::-1]):
                    started = time.perf_counter()
                    call(module, case)
                    timing.append(dict(candidate=ci, scene=case['record']['scene'], method=method,
                        repeat=repeat, elapsed_ms=(time.perf_counter()-started)*1000))
        write_json(args.output/'timing_rows.json', timing)
    unchanged = (hashlib.sha256(Path(observability.__file__).read_bytes()).hexdigest() == production_hash and
                 hashlib.sha256(args.baseline_source.read_bytes()).hexdigest() == original_hash)
    result = dict(status='PASS' if unchanged and not any(check['mismatches'] for check in checks) else 'FAIL',
        kind='research_only_equivalent_pixel_set_optimization', production_source_unchanged=unchanged,
        original_source_sha256=original_hash, candidate_source_sha256=candidate_hash,
        baseline_source=str(args.baseline_source), generated_candidate_matches_production=True,
        functional_cases=len(checks), mismatching_cases=sum(bool(check['mismatches']) for check in checks),
        max_float_difference=max((check['max_float_difference'] for check in checks), default=0.),
        timing_repeats=args.timing_repeats, helper_box_latency_ms={method: distribution(
            [row['elapsed_ms'] for row in timing if row['method'] == method]) for method in ('baseline', 'candidate')},
        scope='Saved helper inputs; no native-rate, classification or live source-contract claim; camera/depth perturbations test equivalence only')
    write_json(args.output/'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached-manifest', type=Path, required=True)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument('--baseline-source', type=Path,
                        help='Frozen pre-optimization observability.py for exact source comparison')
    baseline.add_argument('--baseline-pythonpath', type=Path,
                          help='Preserved package root containing baseline_multiview/observability.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--timing-repeats', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()
    if args.baseline_pythonpath:
        args.baseline_source = args.baseline_pythonpath/'baseline_multiview/observability.py'
    if args.timing_repeats < 0 or args.warmup < 0:
        parser.error('Timing repeats and warmup must be nonnegative')
    main(args)
