"""Independent final-readiness captures/replay, preserving iteration-two cases.

Do not access final holdout observations until the production method is frozen.
The shared research runner supplies identical-frame V1/baseline/improved scoring.
"""
import argparse
import hashlib
from pathlib import Path

from scripts import benchmark_multiview_robustness as benchmark
from tests.support import multiview_readiness_research as fixtures
from tests.support.results import fail_result


def install_fixture():
    benchmark.robustness_scenes = fixtures.readiness_holdout_scenes
    benchmark.EYES, benchmark.SEEDS = fixtures.EYES, fixtures.SEEDS
    original_provenance = benchmark.source_provenance
    def provenance(args):
        result = original_provenance(args)
        for filename in (Path(__file__).resolve(), Path(fixtures.__file__).resolve()):
            result['sha256'][str(filename.relative_to(benchmark.ROOT))] = hashlib.sha256(filename.read_bytes()).hexdigest()
        result['dataset'] = 'final_multiview_readiness_independent_v1'
        result['selection_policy'] = 'Iteration-two selection AND holdout are prior selection data; new holdout outcomes remain unseen until source freeze.'
        return result
    benchmark.source_provenance = provenance


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--result', required=True)
    result.add_argument('--artifacts-dir', required=True)
    result.add_argument('--baseline-pythonpath', default='outputs/multiview-readiness/baseline_snapshot')
    result.add_argument('--baseline-package', default='baseline_multiview')
    result.add_argument('--replay')
    result.add_argument('--capture-only', action='store_true')
    result.add_argument('--split', choices=('selection', 'holdout', 'both'), default='holdout')
    result.add_argument('--scenario', action='append')
    result.add_argument('--limit', type=int)
    result.add_argument('--native-timing-samples', type=int, default=0)
    result.add_argument('--variants', default='clean,depth_noise,calibration,sparse')
    result.add_argument('--no-v1', action='store_true')
    return result


if __name__ == '__main__':
    args = parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit('--limit must be positive')
    if args.native_timing_samples < 0:
        raise SystemExit('--native-timing-samples must be nonnegative')
    if not set(args.variants.split(',')) <= {'clean', 'depth_noise', 'calibration', 'sparse'}:
        raise SystemExit('Unknown perturbation variant')
    install_fixture()
    code = 1
    try:
        code = benchmark.main(args) or 0
    except Exception as error:
        fail_result(args.result, error)
    finally:
        try:
            benchmark.write_json(Path(args.artifacts_dir)/'provenance_end.json', benchmark.source_provenance(args))
        finally:
            if benchmark._native_app is not None:
                benchmark._native_app.close()
    raise SystemExit(code)
