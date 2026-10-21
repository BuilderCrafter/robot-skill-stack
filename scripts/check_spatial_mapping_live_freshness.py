#!/usr/bin/env python3
"""Check default one-second mapping freshness on real native camera buffers.

Runs the unchanged V2 native fixture and adds an independent fresh-map update
before each unfiltered comparison. Both consumers borrow the same camera batch
and existing exposure-matched robot filter. No timestamps or source limits in
the original fixture are changed. Run serially through run_isaac_python.sh.
Only the added live_default_freshness measurements are timing evidence here;
the fixture's original unfiltered timers include this extra check.
"""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from robot_skill_stack.integrations.isaac import spatial_mapping as mapping
    from robot_skill_stack.integrations.isaac.spatial_mapping.ground_truth import IsaacGroundTruthMapper
    from robot_skill_stack.world.perception_multiview.acquisition import FrameSynchronizer
    from robot_skill_stack.world.spatial_mapping.occupancy import OccupancyMap
    from robot_skill_stack.world.spatial_mapping.rgbd import RgbdMapper
    from tests.support import results

    samples, acquisitions, truth_updates, active = [], {}, [], [None]
    original_factory = mapping.create_spatial_mapping
    original_update = RgbdMapper.update
    original_write = results.write_result
    original_capture = FrameSynchronizer.capture
    original_truth_update = IsaacGroundTruthMapper.update

    def frame_key(frames):
        return tuple((frame.camera_id, tuple(frame.token)) for frame in frames)

    def capture(self):
        started = time.perf_counter()
        frames = original_capture(self)
        elapsed = time.perf_counter()-started
        if frames is not None:
            acquisitions[frame_key(frames)] = elapsed*1000
        return frames

    def truth_update(self, *arguments, **keywords):
        started = time.perf_counter()
        changed = original_truth_update(self, *arguments, **keywords)
        truth_updates.append(dict(elapsed_ms=(time.perf_counter()-started)*1000, updated=bool(changed)))
        return changed

    def factory(*arguments, **keywords):
        session = original_factory(*arguments, **keywords)
        if session is not None and session.robot_filter is not None:
            active[0] = session
        return session

    def update(self, frames, exclusion_masks=None, *, now=None, mask_provider=None):
        if mask_provider is None and active[0] is not None:
            # A fresh independent map avoids presenting a replay as a new live
            # observation. This update uses the production wall-clock checks.
            occupancy = OccupancyMap(self.map.spec, stale_after=1., evidence_ttl=5.)
            candidate = RgbdMapper(occupancy, stride=self.stride, max_rays=self.max_rays,
                max_voxel_visits=self.max_voxel_visits, max_depth=self.max_depth,
                max_skew=self.max_skew, max_frame_age=1.,
                expected_camera_ids=self.expected_camera_ids)
            started = time.perf_counter()
            changed = original_update(candidate, frames, exclusion_masks=exclusion_masks,
                mask_provider=lambda current, pixels: active[0].robot_filter.masks(
                    current, pixels_by_camera=pixels))
            elapsed = time.perf_counter()-started
            completed = time.monotonic()
            oldest = min(frame.acquired_at for frame in frames)
            health = dict(occupancy.health)
            row = dict(sample=len(samples), elapsed_ms=elapsed*1000,
                acquisition_ms=acquisitions.get(frame_key(frames)),
                observation_age_at_completion_ms=(completed-oldest)*1000,
                original_timestamp=oldest, map_timestamp=occupancy.timestamp,
                updated=changed, health=health,
                filter_ms=active[0].robot_filter.diagnostics.get('elapsed_seconds', 0.)*1000,
                allocated_rays=candidate.diagnostics['allocated_rays'],
                retained_rays=candidate.diagnostics['retained_rays'])
            samples.append(row)
            assert changed, (row, candidate.diagnostics)
            assert occupancy.timestamp == oldest, row
            assert health['fresh'] and completed-oldest < 1., row
        return original_update(self, frames, exclusion_masks=exclusion_masks,
                               now=now, mask_provider=mask_provider)

    def write_result(path, *, status, metrics=None, error=None):
        import numpy as np
        complete = len(samples) == 15 and all(row['updated'] and row['health']['fresh'] for row in samples)
        values = [row['elapsed_ms'] for row in samples]
        summary = dict(status='PASS' if status == 'PASS' and complete else 'FAIL',
            samples=len(samples), max_frame_age=1., stale_after=1.,
            explicit_now_used=False, timestamps_rewritten=False,
            independent_map_per_sample=True, extra_camera_acquisitions=0,
            fixture_comparison_timings_usable=False,
            paired_acquisition_ms=dict(median=float(np.median([row['acquisition_ms'] for row in samples])),
                p95=float(np.percentile([row['acquisition_ms'] for row in samples], 95)),
                maximum=max(row['acquisition_ms'] for row in samples)) if samples else None,
            ground_truth_updates=truth_updates,
            ground_truth_scope='Unchanged full-arm Isaac reference mapper in this native scene; timings include stage access and dense evidence replacement.',
            mapping_ms=dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                            maximum=max(values)) if values else None,
            maximum_completion_age_ms=max((row['observation_age_at_completion_ms'] for row in samples), default=None),
            measurements=samples)
        payload = dict(metrics or {}, live_default_freshness=summary)
        original_write(path, status=summary['status'], metrics=payload, error=error)

    fixture = ROOT/'tests/spatial_mapping/test_robot_self_filter.py'
    sys.argv = [str(fixture), '--result', str(Path(args.result).resolve()),
                '--artifacts-dir', str(Path(args.artifacts_dir).resolve())]
    with patch.object(mapping, 'create_spatial_mapping', factory), \
            patch.object(RgbdMapper, 'update', update), \
            patch.object(FrameSynchronizer, 'capture', capture), \
            patch.object(IsaacGroundTruthMapper, 'update', truth_update), \
            patch.object(results, 'write_result', write_result):
        exec(compile(fixture.read_text(), str(fixture), 'exec'),
             {'__name__': '__main__', '__file__': str(fixture)})


if __name__ == '__main__':
    main()
