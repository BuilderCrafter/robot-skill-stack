"""Immutable map publication and worker failure barriers without SDK imports."""
import unittest
import numpy as np
from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap, OccupancyState
from robot_skill_stack.world.spatial_mapping.runtime import MappingReadView

class MappingReadViewTests(unittest.TestCase):
    def setUp(self):
        self.map = OccupancyMap(GridSpec((0, 0, 0), (.3, .1, .1), .1),
                                clock=lambda: 100.9, evidence_ttl=1., stale_after=1.)
        self.map.integrate([[0, 0, 0]], [[1, 0, 0]], timestamp=100., now=100.)
        self.map.integrate([[0, 0, 0]], [[1, 0, 0]], timestamp=100.1, now=100.1)
        self.map.integrate([], [[1, 0, 0]], timestamp=100.9, now=100.9)
        self.base = self.map.snapshot(now=100.9)
        self.points = [[.05, .05, .05], [.15, .05, .05], [.25, .05, .05]]

    def view(self, now):
        return MappingReadView(self.base, now=now, evidence_ttl=1., stale_after=1.)

    def test_sparse_ttl_expiry_with_globally_fresh_newer_evidence(self):
        view = self.view(101.2)
        values = view.query_many(self.points)
        np.testing.assert_array_equal(values.states, [OccupancyState.UNKNOWN,
            OccupancyState.OCCUPIED, OccupancyState.UNKNOWN])
        self.assertEqual(values.probabilities[0], .5)
        self.assertTrue(values.observed[0]); self.assertTrue(values.expired[0])
        self.assertEqual(values.last_observed[0], 100.1)
        self.assertFalse(values.observed[2]); self.assertFalse(values.expired[2])
        self.assertTrue(view.health['fresh'])
        self.assertEqual(view.timestamp, 100.9)
        self.assertEqual(view.revision, self.base.revision)

    def test_old_read_view_and_published_source_remain_internally_consistent(self):
        old, new = self.view(101.), self.view(101.2)
        self.assertEqual(old.query(self.points[0]).state, OccupancyState.FREE)
        self.assertEqual(new.query(self.points[0]).state, OccupancyState.UNKNOWN)
        self.assertEqual(old.query(self.points[0]).state, OccupancyState.FREE)
        self.assertEqual(self.base.query(self.points[0]).state, OccupancyState.FREE)
        self.assertEqual(self.base.timestamp, new.timestamp)

    def test_ttl_equality_and_global_source_age_are_separate(self):
        at_deadline = self.view(101.1)
        self.assertEqual(at_deadline.query(self.points[0]).state, OccupancyState.UNKNOWN)
        self.assertTrue(at_deadline.health['fresh'])
        stale = self.view(102.)
        self.assertFalse(stale.health['fresh']); self.assertTrue(stale.health['stale'])
        self.assertEqual(stale.timestamp, 100.9)
        self.assertEqual(stale.query(self.points[1]).last_observed, 100.9)

    def test_sparse_and_explicit_dense_exports_are_immutable_and_agree(self):
        view = self.view(101.2)
        sparse = view.query_many(self.points)
        np.testing.assert_array_equal(view.states[:, 0, 0], sparse.states)
        np.testing.assert_array_equal(view.probabilities[:, 0, 0], sparse.probabilities)
        np.testing.assert_array_equal(view.expired[:, 0, 0], sparse.expired)
        for values in (view.states, view.probabilities, view.observed, view.last_observed,
                       view.expired, sparse.states, sparse.last_observed):
            with self.assertRaises(ValueError): values.setflags(write=True)
        outside = view.query([1., 0., 0.])
        self.assertEqual(outside.state, OccupancyState.OUT_OF_BOUNDS)
        self.assertFalse(outside.observed)

from dataclasses import FrozenInstanceError
from unittest.mock import patch
from robot_skill_stack.integrations.isaac.spatial_mapping.config import SpatialMappingConfig
from robot_skill_stack.runtime.worker import WorkResult
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.spatial_mapping.geometry import (
    GeometryItem, GroundTruthInput, _Primitive, _primitive_bounds,
)
from robot_skill_stack.world.spatial_mapping.runtime import (
    MappingInput, MappingProcessor, MappingService,
)


def settings(**overrides):
    values = dict(source='ground_truth', bounds_min=(0, 0, 0), bounds_max=(.3, .1, .1),
                  voxel_size=.1, evidence_ttl=1., stale_after=1., max_frame_age=1.)
    values.update(overrides)
    return SpatialMappingConfig(**values)


def gt(stamp=100., center=(.15, .05, .05)):
    matrix = np.eye(4)
    matrix[:3, 3] = center
    primitive = _Primitive('cube', matrix, np.array([.02]*3))
    lower, upper, inverse = _primitive_bounds(primitive)
    return GroundTruthInput(stamp, (GeometryItem(lower, upper, primitive, inverse),), {})


class ManualWorker:
    """Delayed delivery even after invalidation exercises the service's barrier."""
    def __init__(self, process, *, name):
        self.process, self.pending, self.result = process, None, None
        self.closed, self.invalidations = False, 0
    def submit(self, payload, tag=None):
        self.pending = payload, tag
        return not self.closed
    def finish(self):
        payload, tag = self.pending
        value = self.process(payload)
        self.result = WorkResult(tag, value, None, 99.9, 100., 100.01)
    def poll(self):
        result, self.result = self.result, None
        return result
    def invalidate(self):
        self.invalidations += 1
    def close(self): self.closed = True
    @property
    def health(self): return dict(running=False, pending=self.pending is not None, closed=self.closed)


class MappingRuntimeTests(unittest.TestCase):
    def test_input_owns_depth_and_gt_arrays_without_sdk_objects(self):
        depth, k, t = np.ones((1, 1)), np.eye(3), np.eye(4)
        frame = DepthFrame('A', (1, 1), depth, k, t, 100.)
        payload = MappingInput(frames=(frame,))
        depth[:] = 9; k[0, 0] = 9; t[2, 3] = 9
        self.assertEqual(payload.frames[0].depth[0, 0], 1.)
        self.assertEqual(payload.frames[0].intrinsics[0, 0], 1.)
        self.assertEqual(payload.frames[0].world_from_camera[2, 3], 0.)
        for array in (payload.frames[0].depth, gt().items[0].primitive.world_from_local):
            with self.assertRaises(ValueError): array.setflags(write=True)
        with self.assertRaises(ValueError): MappingInput(frames=(frame,), ground_truth=gt())

    def test_gt_processor_preserves_source_timestamp_and_rebuilds_generation(self):
        processor = MappingProcessor(settings(), clock=lambda: 100.2)
        one = processor.compute(MappingInput(ground_truth=gt()))
        self.assertTrue(one.accepted)
        self.assertEqual(one.snapshot.timestamp, 100.)
        self.assertEqual(one.snapshot.query([.15, .05, .05]).state, OccupancyState.OCCUPIED)
        self.assertEqual(one.snapshot.query([.05, .05, .05]).state, OccupancyState.FREE)
        repeated = processor.compute(MappingInput(ground_truth=gt()))
        self.assertFalse(repeated.accepted)
        reset = processor.compute(MappingInput(generation=1, ground_truth=gt()))
        self.assertTrue(reset.accepted)
        self.assertEqual(reset.snapshot.generation, 1)

    def test_rgbd_worker_uses_original_time_and_never_sdk(self):
        cfg = settings(source='rgbd', bounds_min=(-.1, -.1, 0.), bounds_max=(.1, .1, 1.1),
                       stride=1)
        frame = DepthFrame('A', (1, 1), np.ones((1, 1)),
                           np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]]), np.eye(4), 100.)
        processor = MappingProcessor(cfg, clock=lambda: 100.1)
        completed = processor.compute(MappingInput(frames=(frame,)))
        self.assertTrue(completed.accepted)
        self.assertEqual(completed.snapshot.timestamp, 100.)
        self.assertEqual(completed.snapshot.query([0., 0., 1.]).state, OccupancyState.OCCUPIED)
        self.assertFalse(processor.compute(MappingInput(frames=(frame,))).accepted)

    def test_failure_barrier_discards_running_success_but_next_input_recovers(self):
        service = MappingService(settings(), worker_factory=ManualWorker, clock=lambda: 100.2)
        service.submit(MappingInput(ground_truth=gt()))
        service._worker.finish()
        service.note_source_failure('pose history unavailable')
        self.assertIsNone(service.poll())
        self.assertIsNone(service.snapshot().timestamp)
        self.assertEqual(service.health['status'], 'error')
        service.submit(MappingInput(ground_truth=gt(100.1)))
        service._worker.finish(); result = service.poll()
        self.assertTrue(result.accepted)
        self.assertEqual(service.snapshot().timestamp, 100.1)
        self.assertTrue(service.health['fresh'])
        self.assertAlmostEqual(service.health['worker_seconds'], .01)
        self.assertEqual(service.health['published'], 1)

    def test_reset_hides_old_map_and_discards_late_result(self):
        service = MappingService(settings(), worker_factory=ManualWorker, clock=lambda: 100.2)
        service.submit(MappingInput(ground_truth=gt()))
        service._worker.finish(); service.poll()
        old = service.snapshot()
        service.submit(MappingInput(ground_truth=gt(100.1)))
        service._worker.finish()
        service.reset()
        self.assertIsNone(service.snapshot().timestamp)
        self.assertEqual(service.snapshot().generation, 1)
        self.assertIsNone(service.poll())
        self.assertEqual(old.timestamp, 100.)
        self.assertFalse(service.submit(MappingInput(ground_truth=gt(100.2))))
        service.submit(MappingInput(generation=1, ground_truth=gt(100.)))
        service._worker.finish(); self.assertTrue(service.poll().accepted)

    def test_close_is_nonblocking_and_late_result_never_publishes(self):
        service = MappingService(settings(), worker_factory=ManualWorker, clock=lambda: 100.2)
        service.submit(MappingInput(ground_truth=gt()))
        service._worker.finish(); service.close()
        self.assertIsNone(service.poll())
        self.assertIsNone(service.snapshot().timestamp)
        self.assertFalse(service.health['fresh'])
        self.assertFalse(service.submit(MappingInput(ground_truth=gt())))

    def test_snapshot_and_sparse_reads_never_touch_worker_map_or_dense_age(self):
        service = MappingService(settings(), worker_factory=ManualWorker, clock=lambda: 100.2)
        service.submit(MappingInput(ground_truth=gt()))
        service._worker.finish(); service.poll()
        with patch.object(service._processor.map, 'snapshot', side_effect=AssertionError('owner copied map')):
            with patch.object(MappingReadView, 'states', property(lambda self: (_ for _ in ()).throw(
                    AssertionError('dense scan during sparse read')))):
                view = service.snapshot(now=101.2)
                query = view.query([.15, .05, .05])
                self.assertTrue(query.expired)
                self.assertEqual(query.state, OccupancyState.UNKNOWN)
                self.assertEqual(query.last_observed, 100.)
                self.assertFalse(view.health['fresh'])
        with self.assertRaises(FrozenInstanceError): view.evaluated_at = 1.

    def test_delayed_completion_is_rejected_without_replacing_previous_map(self):
        clock = [100.2]
        service = MappingService(settings(), worker_factory=ManualWorker, clock=lambda: clock[0])
        service.submit(MappingInput(ground_truth=gt()))
        service._worker.finish(); service.poll()
        service.submit(MappingInput(ground_truth=gt(100.1, center=(.25, .05, .05))))
        service._worker.finish()
        clock[0] = 101.1  # Publication is at the source-age boundary.
        self.assertFalse(service.poll().accepted)
        self.assertEqual(service.snapshot().timestamp, 100.)
        self.assertEqual(service.health['published'], 1)
        self.assertEqual(service.health['last']['status'], 'stale_completion')
        self.assertAlmostEqual(service.health['last']['publish_age_seconds'], 1.)
        self.assertEqual(service.health['last']['input_timestamp'], 100.1)

    def test_worker_failure_has_separate_reason_and_counter(self):
        service = MappingService(settings(), worker_factory=ManualWorker, clock=lambda: 100.2)
        service.submit(MappingInput(ground_truth=gt()))
        tag = service._worker.pending[1]
        service._worker.result = WorkResult(tag, None, 'ValueError: invalid geometry', 100., 100., 100.1)
        self.assertIsNone(service.poll())
        self.assertEqual(service.health['reason'], 'worker_failure')
        self.assertEqual(service.health['worker_failures'], 1)
        self.assertEqual(service.health['source_failures'], 0)
        service.note_source_failure('camera disconnected')
        self.assertEqual(service.health['source_failures'], 1)
        self.assertEqual(service.health['last_error'], 'camera disconnected')

    def test_no_frame_does_not_cancel_running_current_input(self):
        service = MappingService(settings(), worker_factory=ManualWorker, clock=lambda: 100.2)
        service.submit(MappingInput(ground_truth=gt()))
        service.note_no_frame()
        self.assertEqual(service._worker.invalidations, 0)
        service._worker.finish()
        self.assertTrue(service.poll().accepted)
        self.assertTrue(service.health['fresh'])

    def test_explicit_reset_generation_cannot_reuse_old_processor_state(self):
        service = MappingService(settings(), worker_factory=ManualWorker, clock=lambda: 100.2)
        with self.assertRaisesRegex(ValueError, 'newer'):
            service.reset(generation=0)
        service.reset(generation=4)
        self.assertEqual(service.generation, 4)
        self.assertIsNone(service.snapshot().timestamp)

    def test_worker_filter_and_mapper_match_synchronous_inputs_exactly(self):
        from robot_skill_stack.world.spatial_mapping.robot_filter import (
            TriangleMesh, MeshInstance, RobotSnapshot, RobotSelfFilter,
        )
        from robot_skill_stack.world.spatial_mapping.rgbd import RgbdMapper
        cfg = settings(source='rgbd', automatic_robot_filter=True, stride=1,
                       bounds_min=(-.2, -.2, 0.), bounds_max=(.2, .2, 1.6))
        depth = np.array([[.5, 1., 1.5], [1., 1., 1.], [1., 1., 1.]])
        frame = DepthFrame('A', (10, 1.), depth,
            np.array([[100., 0., 1.5], [0., 100., 1.5], [0., 0., 1.]]), np.eye(4), 100.)
        mesh = TriangleMesh(np.array([[-.2, -.2, 1.], [.2, -.2, 1.], [.2, .2, 1.], [-.2, .2, 1.]]),
                            np.array([[0, 1, 2], [0, 2, 3]]))
        snapshots = {'A': RobotSnapshot((MeshInstance(mesh, np.eye(4)),), 1., frame.token)}
        mask = np.zeros((3, 3), bool); mask[2, 2] = True
        payload = MappingInput(frames=(frame,), robot_snapshots=snapshots, exclusion_masks={'A': mask})
        computed = MappingProcessor(cfg, clock=lambda: 100.1).compute(payload)
        ordinary = OccupancyMap(cfg.grid_spec(), clock=lambda: 100.1,
            stale_after=cfg.stale_after, evidence_ttl=cfg.evidence_ttl)
        filter_ = RobotSelfFilter(stride=1)
        mapper = RgbdMapper(ordinary, stride=1, clock=lambda: 100.1)
        self.assertTrue(mapper.update((frame,), exclusion_masks={'A': mask},
            mask_provider=lambda frames, pixels: filter_.masks(frames, snapshots, pixels_by_camera=pixels)))
        original = ordinary.snapshot(now=100.1)
        for field in ('states', 'probabilities', 'observed', 'last_observed', 'expired'):
            np.testing.assert_array_equal(getattr(computed.snapshot, field), getattr(original, field))
        self.assertEqual(computed.snapshot.timestamp, original.timestamp)
        self.assertEqual(computed.diagnostics['automatic_masked_selected'], 7)
        self.assertEqual(computed.diagnostics['retained_rays'], 1)

    def test_health_keeps_authoritative_status_and_published_timestamp(self):
        clock = [100.2]
        service = MappingService(settings(), worker_factory=ManualWorker, clock=lambda: clock[0])
        service.submit(MappingInput(ground_truth=gt()))
        service._worker.finish(); service.poll()
        self.assertEqual(service.health['status'], 'ok')
        self.assertEqual(service.health['last']['status'], 'published')
        service.note_source_failure('camera failed after successful publication')
        self.assertEqual(service.health['status'], 'error')
        self.assertFalse(service.health['fresh'])
        service.submit(MappingInput(ground_truth=gt(100.1)))
        service._worker.finish(); clock[0] = 101.1
        service.poll()
        self.assertEqual(service.health['timestamp'], 100.)
        self.assertEqual(service.health['last']['timestamp'], 100.1)
        self.assertEqual(service.health['status'], 'no_frame')
        service.close()
        self.assertEqual(service.health['status'], 'closed')
        self.assertFalse(service.health['fresh'])

    def test_rgbd_returned_failure_reports_actual_error_and_attempt_timestamp(self):
        cfg = settings(source='rgbd', bounds_min=(-.1, -.1, 0.), bounds_max=(.1, .1, 1.1), stride=1)
        clock = [100.1]
        service = MappingService(cfg, worker_factory=ManualWorker, clock=lambda: clock[0])
        def packet(number, acquired, depth):
            return MappingInput(frames=(DepthFrame('A', (number, float(number)), np.array([[depth]]),
                np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]]), np.eye(4), acquired),))
        service.submit(packet(1, 100., 1.))
        service._worker.finish(); self.assertTrue(service.poll().accepted)
        clock[0] = 100.3
        service.submit(packet(2, 100.2, np.nan))
        service._worker.finish(); self.assertFalse(service.poll().accepted)
        health = service.health
        self.assertEqual(health['timestamp'], 100.)
        self.assertEqual(health['last']['timestamp'], 100.)
        self.assertEqual(health['last']['input_timestamp'], 100.2)
        self.assertEqual(health['reason'], 'processing_failure')
        self.assertIn('All depth pixels invalid', health['last_error'])
        self.assertEqual(health['processing_failures'], 1)
        self.assertEqual(health['worker_failures'], 1)
        self.assertEqual(health['source_failures'], 0)

    def test_configured_camera_batch_is_complete_before_generic_integration(self):
        cfg = settings(source='rgbd', bounds_min=(-.1, -.1, 0.), bounds_max=(.1, .1, 1.1),
                       stride=1, camera_prim_paths=('/A', '/B'))
        def frame(key, number=1):
            return DepthFrame(key, (number, float(number)), np.ones((1, 1)),
                np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]]), np.eye(4), 100.)
        for ids in (('/A',), ('/A', '/wrong')):
            processor = MappingProcessor(cfg, clock=lambda: 100.1)
            failed = processor.compute(MappingInput(frames=tuple(frame(key) for key in ids)))
            self.assertFalse(failed.accepted)
            self.assertEqual(failed.snapshot.health['status'], 'error')
            self.assertIn('Missing or unexpected cameras', failed.diagnostics['reason'])
            self.assertIsNone(failed.snapshot.timestamp)
            self.assertFalse(failed.snapshot.observed.any())
        accepted = MappingProcessor(cfg, clock=lambda: 100.1).compute(
            MappingInput(frames=(frame('/B'), frame('/A'))))
        self.assertTrue(accepted.accepted)
        self.assertEqual(set(accepted.diagnostics['cameras']), {'/A', '/B'})

    def test_compute_and_transport_durations_are_separate_from_roundtrip(self):
        from dataclasses import replace
        service = MappingService(settings(), worker_factory=ManualWorker, clock=lambda: 100.2)
        service.submit(MappingInput(ground_truth=gt()))
        service._worker.finish()
        work = service._worker.result
        value = replace(work.value, diagnostics=dict(work.value.diagnostics, compute_seconds=.004))
        service._worker.result = replace(work, value=value)
        self.assertTrue(service.poll().accepted)
        last = service.health['last']
        self.assertAlmostEqual(last['worker_seconds'], .01)
        self.assertEqual(last['compute_seconds'], .004)
        self.assertAlmostEqual(last['transport_seconds'], .006)
        self.assertAlmostEqual(last['queue_seconds'], .1)
        self.assertAlmostEqual(last['publish_age_seconds'], .2)
