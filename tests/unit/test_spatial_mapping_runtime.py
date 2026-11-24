"""Mapping computation, fixed-time evidence and worker publication barriers."""
from unittest.mock import patch
import unittest

import numpy as np

from robot_skill_stack.runtime.worker import WorkResult
from robot_skill_stack.world.spatial_mapping import (
    GridSpec, MappingInput, MappingReadView, MappingService, OccupancyMap, OccupancyState, SpatialMappingConfig,
)
from robot_skill_stack.world.spatial_mapping.geometry import GroundTruthInput, _primitive_item
from robot_skill_stack.world.spatial_mapping.runtime import MappingProcessor
from tests.unit.test_spatial_mapping_robot_filter import frame, snapshot


def settings(**overrides):
    values = dict(source='ground_truth', bounds_min=(0,0,0), bounds_max=(.3,.1,.1),
                  voxel_size=.1, evidence_ttl=1., stale_after=1., max_frame_age=1.)
    return SpatialMappingConfig(**(values | overrides))


def gt(stamp=100., center=(.15,.05,.05)):
    matrix = np.eye(4)
    matrix[:3,3] = center
    return GroundTruthInput(stamp,(_primitive_item('cube',matrix,dimensions=np.array([.02]*3)),))


class ManualWorker:
    """Deliver delayed results even after invalidation to exercise the barrier."""
    def __init__(self, process, *, name):
        self.process, self.pending, self.result = process,None,None
        self.closed, self.invalidations = False,0
    def submit(self, payload, tag=None):
        self.pending = payload,tag
        return not self.closed
    def finish(self):
        payload,tag = self.pending
        self.result = WorkResult(tag,self.process(payload),None,99.9,100.,100.01)
    def poll(self):
        result,self.result = self.result,None
        return result
    def invalidate(self):
        self.invalidations += 1
    def close(self):
        self.closed = True
    @property
    def health(self):
        return dict(running=False,pending=self.pending is not None,closed=self.closed)


class MappingRuntimeTests(unittest.TestCase):
    def service(self, **options):
        self.clock = [100.2]
        service = MappingService(settings(**options),worker_factory=ManualWorker,clock=lambda:self.clock[0])
        self.addCleanup(service.close)
        return service

    def publish(self, service, stamp=100.):
        service.submit(MappingInput(generation=service.generation,ground_truth=gt(stamp)))
        service._worker.finish()
        return service.poll()

    def test_camera_input_copies_mutable_arrays_and_reuses_acquisition_buffers(self):
        image = frame()
        payload = MappingInput(frames=(image,))
        image.depth[:] = 9
        image.intrinsics[0,0] = 9
        image.world_from_camera[2,3] = 9
        frozen = payload.frames[0]
        self.assertEqual(frozen.depth[0,0],1.)
        self.assertEqual(frozen.intrinsics[0,0],1000.)
        self.assertEqual(frozen.world_from_camera[2,3],0.)
        self.assertIs(MappingInput(frames=(frozen,)).frames[0].depth,frozen.depth)
        with self.assertRaises(ValueError):
            frozen.depth.setflags(write=True)

    def test_gt_computation_preserves_time_and_reinitializes_on_new_generation(self):
        processor = MappingProcessor(settings(),clock=lambda:100.2)
        with patch('robot_skill_stack.world.spatial_mapping.runtime.RgbdMapper',side_effect=AssertionError('GT needs no cameras')):
            one = processor.compute(MappingInput(ground_truth=gt()))
        self.assertTrue(one.accepted)
        self.assertEqual(one.snapshot.timestamp,100.)
        self.assertEqual(one.snapshot.query([.15,.05,.05]).state,OccupancyState.OCCUPIED)
        self.assertEqual(one.snapshot.query([.05,.05,.05]).state,OccupancyState.FREE)
        self.assertFalse(processor.compute(MappingInput(ground_truth=gt())).accepted)
        newer = processor.compute(MappingInput(generation=1,ground_truth=gt()))
        self.assertTrue(newer.accepted)
        self.assertEqual(newer.snapshot.generation,1)

    def test_rgbd_computation_filters_robot_returns_and_keeps_foreground(self):
        cfg = settings(source='rgbd',automatic_robot_filter=True,stride=1,
            bounds_min=(-.2,-.2,0.),bounds_max=(.2,.2,1.5))
        payload = MappingInput(frames=(frame([[.98,1.,1.02]]),),robot_snapshots={'A':snapshot()})
        result = MappingProcessor(cfg,clock=lambda:100.2).compute(payload)
        self.assertTrue(result.accepted)
        self.assertEqual(result.snapshot.timestamp,100.)
        self.assertEqual(result.snapshot.query([-.00098,0.,.98]).state,OccupancyState.OCCUPIED)
        self.assertEqual(result.snapshot.query([0.,0.,1.02]).state,OccupancyState.UNKNOWN)
        self.assertEqual(result.source_frames[0]['token'],payload.frames[0].token)

    def test_fixed_time_queries_expire_to_unknown_without_mutating_frozen_evidence(self):
        occupancy = OccupancyMap(GridSpec((0,0,0),(.3,.1,.1),.1),evidence_ttl=1.)
        occupancy.integrate([[0,0,0]],[[1,0,0]],timestamp=100.,now=100.)
        occupancy.integrate([[0,0,0]],[[1,0,0]],timestamp=100.1,now=100.1)
        occupancy.integrate([],[[1,0,0]],timestamp=100.9,now=100.9)
        base = occupancy.snapshot(now=100.9)
        old = MappingReadView(base,100.9,1.,1.)
        later = MappingReadView(base,101.2,1.,1.)
        values = later.query_many([[.05,.05,.05],[.15,.05,.05],[.25,.05,.05],[1.,0.,0.]])
        np.testing.assert_array_equal(values.states,[OccupancyState.UNKNOWN,OccupancyState.OCCUPIED,
            OccupancyState.UNKNOWN,OccupancyState.OUT_OF_BOUNDS])
        self.assertTrue(values.expired[0] and values.observed[0])
        self.assertFalse(values.expired[2] or values.observed[2])
        self.assertEqual(values.last_observed[0],100.1)
        self.assertEqual(values.probabilities[0],.5)
        self.assertTrue(later.health['fresh'])
        occupancy.replace([[2,0,0]],timestamp=101.,now=101.)
        self.assertEqual(old.query([.05,.05,.05]).state,OccupancyState.FREE)
        self.assertEqual(old.query([.25,.05,.05]).state,OccupancyState.UNKNOWN)
        self.assertFalse(MappingReadView(base,101.9,1.,1.).health['fresh'])

    def test_publication_reset_and_close_reject_late_results_and_keep_old_views(self):
        service = self.service()
        result = self.publish(service)
        old = service.snapshot()
        self.assertIs(old._snapshot,result.snapshot)
        service.submit(MappingInput(ground_truth=gt(100.1)))
        service._worker.finish()
        service.reset()
        self.assertIsNone(service.poll())
        self.assertIsNone(service.snapshot().timestamp)
        self.assertEqual(service.snapshot().generation,1)
        self.assertFalse(service.submit(MappingInput(ground_truth=gt(100.15))))
        self.assertTrue(self.publish(service,100.15).accepted)
        service.submit(MappingInput(generation=1,ground_truth=gt(100.18)))
        service._worker.finish()
        service.close()
        self.assertIsNone(service.poll())
        self.assertFalse(service.submit(MappingInput(generation=1,ground_truth=gt(100.19))))
        self.assertFalse(service.health['fresh'])
        self.assertEqual(old.timestamp,100.)
        self.assertEqual(old.query([.15,.05,.05]).state,OccupancyState.OCCUPIED)

    def test_source_and_worker_failures_invalidate_pending_work_and_recover(self):
        service = self.service()
        service.submit(MappingInput(ground_truth=gt()))
        service._worker.finish()
        service.note_source_failure('lost camera')
        self.assertIsNone(service.poll())
        self.assertIsNone(service.snapshot().timestamp)
        self.assertFalse(service.health['fresh'])
        self.assertTrue(self.publish(service,100.1).accepted)
        service.submit(MappingInput(ground_truth=gt(100.15)))
        service._worker.result = WorkResult(service._worker.pending[1],None,'worker failed',100.,100.,100.1)
        self.assertIsNone(service.poll())
        self.assertEqual(service.health['status'],'error')
        self.assertEqual(service.health['failures'],2)
        self.assertEqual(service.snapshot().timestamp,100.1)
        service.submit(MappingInput(ground_truth=gt(100.18)))
        service.note_no_frame()
        service._worker.finish()
        self.assertTrue(service.poll().accepted)
        self.assertTrue(service.health['fresh'])

    def test_invalid_rgbd_and_delayed_completion_cannot_refresh_the_publication(self):
        from dataclasses import replace
        service = self.service(source='rgbd',stride=1,bounds_min=(-.1,-.1,0.),bounds_max=(.1,.1,1.1))
        for number,stamp,depth in ((1,100.,1.),(2,100.1,np.nan),(3,100.15,1.),(4,100.18,1.)):
            image = replace(frame(depth,token=(number,number*.1)),acquired_at=stamp)
            service.submit(MappingInput(frames=(image,)))
            service._worker.finish()
            if number == 4:
                self.clock[0] = stamp+1.
            result = service.poll()
            self.assertEqual(result.accepted,number in (1,3))
            if number == 2:
                self.assertEqual(service.health['status'],'error')
                self.assertEqual(service.snapshot().timestamp,100.)
        self.assertEqual(service.snapshot().timestamp,100.15)
        self.assertFalse(service.health['fresh'])
        self.assertEqual(service.health['published'],2)
        self.assertEqual(service.health['failures'],1)
