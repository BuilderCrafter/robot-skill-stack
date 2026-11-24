"""Test-only SDK-free timing wrapper, importable by the production worker child."""
from collections import defaultdict
from contextlib import ExitStack
import time
from unittest.mock import patch


class TimedCompute:
    def __init__(self, compute):
        self.compute = compute

    def __call__(self, prepared):
        from robot_skill_stack.world.perception.multiview import provider, segmentation
        timings, clusters = defaultdict(float), []

        def measured(function, name):
            def call(*args, **kwargs):
                started = time.perf_counter()
                try:
                    result = function(*args, **kwargs)
                    if name == 'coarse':
                        clusters.extend(dict(points=len(p), low=p.min(0).tolist(),
                                             high=p.max(0).tolist()) for p in result)
                    return result
                finally:
                    timings[name] += time.perf_counter()-started
            return call

        fitter = self.compute.__self__.fitter
        targets = [(provider, name, 'projection_voxel') for name in ('project_depth', 'crop_workspace', 'voxelize')]
        targets += [(provider, 'cluster_points', 'coarse'), (provider, 'refine_cluster', 'refinement'),
                    (segmentation, 'radius_components', 'metric_graph'),
                    (fitter, 'fit', 'fit'), (fitter, 'finalize', 'parameters'), (provider, 'assess', 'observability')]
        started = time.perf_counter()
        with ExitStack() as stack:
            for owner, attribute, name in targets:
                stack.enter_context(patch.object(owner, attribute, measured(getattr(owner, attribute), name)))
            result = self.compute(prepared)
        return result, dict(seconds=time.perf_counter()-started, stages=dict(timings), coarse=clusters)
