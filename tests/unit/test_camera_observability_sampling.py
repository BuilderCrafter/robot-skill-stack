"""Research sampling keeps the canonical pipeline and exposes coverage tradeoffs."""
from types import SimpleNamespace
import unittest

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.spatial_mapping.occupancy import GridSpec, OccupancyMap
from robot_skill_stack.world.spatial_mapping.rgbd import RgbdMapper
from robot_skill_stack.integrations.isaac.spatial_mapping.config import SpatialMappingConfig
from scripts.camera_observability_sampling import (
    ResearchMapper, ResearchMappingProcessor, ReplayClock, _stage_snapshot,
    candidate_pixels, physical_matches, normalize_object, selection_indices)


def frame(depth=None):
    depth = np.full((32, 32), .5) if depth is None else depth
    k = np.array([[40., 0., 16.], [0., 40., 16.], [0., 0., 1.]])
    return DepthFrame('/World/Camera', (1., .1), depth, k, np.eye(4), 10.)


class CameraObservabilitySamplingTests(unittest.TestCase):
    def test_fixed_research_path_matches_canonical_map_exactly(self):
        maps = [OccupancyMap(GridSpec((-.4, -.4, 0.), (.4, .4, .8), .02),
                             clock=lambda: 10.1) for _ in range(2)]
        canonical = RgbdMapper(maps[0], stride=4, max_rays=32)
        research = ResearchMapper(maps[1], stride=4, max_rays=32, policy='fixed1024')
        f = frame()
        a = canonical.update((f,), now=10.1)
        b = research.update((f,), now=10.1)
        self.assertEqual(a, b)
        self.assertEqual(canonical.diagnostics, research.diagnostics)
        for name in ('states', 'probabilities', 'observed', 'last_observed', 'expired'):
            np.testing.assert_array_equal(getattr(maps[0].snapshot(now=10.1), name),
                                          getattr(maps[1].snapshot(now=10.1), name))

    def test_cycle_changes_stride_pixels_at_same_per_update_budget(self):
        f = frame()
        union = set()
        for iteration in range(4):
            y, x = candidate_pixels(f, 4, 'cycle1024', iteration)
            indices = selection_indices(f, y, x, f.depth[y, x], 16, 'cycle1024', iteration)
            selected = set(zip(y[indices], x[indices]))
            self.assertEqual(len(selected), 16)
            union.update(selected)
        self.assertEqual(len(union), 64)

    def test_depth_edge_selector_keeps_uniform_half_and_uses_depth_only(self):
        depth = np.full((32, 32), .5)
        depth[:, 16:] = .8
        f = frame(depth)
        y, x = candidate_pixels(f, 2, 'edge1024', 0)
        baseline = selection_indices(f, y, x, depth[y, x], 32, 'fixed1024', 0)
        chosen = selection_indices(f, y, x, depth[y, x], 32, 'edge1024', 0)
        self.assertEqual(len(set(chosen)), 32)
        self.assertTrue(set(baseline[::2]) <= set(chosen))
        extras = [index for index in chosen if index not in baseline[::2]]
        self.assertTrue(all(8 <= x[index] <= 22 for index in extras))
        np.testing.assert_array_equal(chosen,
            selection_indices(f, y, x, depth[y, x], 32, 'edge1024', 0))

    def test_clock_preserves_phase_times_and_explicit_logical_age(self):
        clock = ReplayClock()
        f = frame()
        clock.load(dict(clock_reads=[dict(phase='rgbd_update', value=10.1),
            dict(phase='rgbd_update', value=10.2), dict(phase='processor', value=10.3)]), (f,))
        clock.phase = 'rgbd_update'
        self.assertEqual((clock(), clock()), (10.1, 10.2))
        clock.phase = 'processor'
        self.assertEqual(clock(), 10.3)
        with self.assertRaises(RuntimeError):
            clock()
        clock.load({}, (f,))
        self.assertEqual(clock.evaluated_at, 10.001)
        self.assertEqual(f.acquired_at, 10.)

    def test_stage_audit_rejects_proxy_geometry_in_unknown_cells(self):
        spec = GridSpec((0., 0., 0.), (.03, .03, .03), .01)
        occupancy = OccupancyMap(spec, clock=lambda: 10.)
        packet = SimpleNamespace(boxes=[SimpleNamespace(id='0:1:0:1:0:1')], diagnostics={})
        with self.assertRaisesRegex(AssertionError, 'invented FREE/UNKNOWN'):
            _stage_snapshot(occupancy.snapshot(now=10.), packet, [], {}, spec)

    def test_processor_cycle_filter_override_is_explicit_and_local(self):
        cfg = SpatialMappingConfig(source='rgbd')
        ordinary = ResearchMappingProcessor(cfg, policy='cycle_stride')
        prototype = ResearchMappingProcessor(cfg, policy='cycle_stride', allow_cycle_offsets=True)
        ordinary._reset(0)
        prototype._reset(0)
        self.assertEqual(ordinary.filter.stride, 8)
        self.assertEqual(prototype.filter.stride, 1)
        self.assertEqual(prototype.mapper.stride, 8)
        self.assertEqual(prototype.mapper.max_rays, ordinary.mapper.max_rays)
        with self.assertRaisesRegex(ValueError, 'configured total ray budget'):
            ResearchMappingProcessor(cfg, policy='fixed4096')

    def test_curved_fixture_depth_is_not_scored_using_its_aabb_corners(self):
        f = frame()
        f.depth[:] = .5
        shape = normalize_object(dict(id='sphere', kind='sphere', center=[0., 0., .6], radius=.1))
        # Center ray touches the sphere atz=.5; a diagonal AABB corner ray does
        # not. Geometric labels are evaluation outputs, never sampler inputs.
        y, x = np.array([16, 22]), np.array([16, 22])
        matches = physical_matches(f, y, x, shape)
        self.assertTrue(matches[0])
        self.assertFalse(matches[1])


if __name__ == '__main__':
    unittest.main()
