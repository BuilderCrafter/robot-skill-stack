"""Frozen mapping identity and bounded geometry, without SDK or dense rereads."""
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.runtime.worker import WorkResult
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.spatial_mapping.frozen import (
    FrozenRgbdBox, FrozenRgbdSourceError, capture_frozen_rgbd,
)
from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleBox, ObstacleExtractionSettings
from robot_skill_stack.world.spatial_mapping.occupancy import OccupancyState
from robot_skill_stack.world.spatial_mapping.robot_filter import MeshInstance, RobotSnapshot, TriangleMesh
from robot_skill_stack.world.spatial_mapping.runtime import MappingInput, MappingProcessor, MappingService
from tests.unit.test_spatial_mapping_runtime import ManualWorker, settings


class FrozenRgbdWorldTests(unittest.TestCase):
    def service(self, *, maximum=64, ttl=5., support=None, **changes):
        values = dict(enabled=True, source='rgbd', automatic_robot_filter=True, stride=1,
            bounds_min=(-.05,-.05,.8), bounds_max=(.05,.05,1.05), voxel_size=.01,
            max_frame_age=1., stale_after=1., evidence_ttl=ttl, camera_prim_paths=('/A','/B'))
        values.update(changes)
        self.clock = [100.2]
        source = MappingService(settings(**values), worker_factory=ManualWorker, clock=lambda:self.clock[0],
            obstacle_settings=ObstacleExtractionSettings(maximum, support))
        self.addCleanup(source.close)
        return source

    def payload(self, source, number=1, acquired=100., depth=1.):
        k = np.array([[100.,0.,1.5],[0.,100.,1.5],[0.,0.,1.]])
        frames = tuple(DepthFrame(key, (number, number*.1), np.full((3,3),depth), k.copy(),
            np.eye(4), acquired) for key in ('/A','/B'))
        mesh = TriangleMesh(np.array([[-.2,-.2,2.],[.2,-.2,2.],[.2,.2,2.],[-.2,.2,2.]]),
            np.array([[0,1,2],[0,2,3]]))
        robot = {frame.camera_id:RobotSnapshot((MeshInstance(mesh,np.eye(4)),),
            frame.token[1], frame.token) for frame in frames}
        return MappingInput(generation=source.generation, frames=frames, robot_snapshots=robot)

    def publish(self, source, number=1, acquired=100., depth=1.):
        source.submit(self.payload(source, number, acquired, depth))
        source._worker.finish()
        result = source.poll()
        self.assertTrue(result.accepted)
        return result

    def prepared(self, **options):
        source = self.service(**options)
        self.publish(source)
        self.publish(source,2,100.1)
        self.assertTrue(source.snapshot().obstacles.boxes)
        return source

    def rejected(self, source, code, **options):
        with self.assertRaises(FrozenRgbdSourceError) as caught:
            capture_frozen_rgbd(source, **options)
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_real_worker_metadata_retains_original_camera_exposure_and_calibration(self):
        source = self.prepared()
        frozen = capture_frozen_rgbd(source)
        self.assertEqual(frozen.timestamp,100.1)
        self.assertEqual(frozen.evaluated_at,100.2)
        self.assertEqual(frozen.generation,source.generation)
        self.assertEqual(frozen.revision,source.snapshot().revision)
        self.assertEqual(frozen.metadata['camera_ids'],('/A','/B'))
        self.assertEqual(frozen.metadata['accepted_frame_count'],2)
        frames = frozen.metadata['source_frames']
        self.assertEqual(frames[0]['token'],(2, .2))
        self.assertEqual(frames[0]['acquired_at'],100.1)
        self.assertEqual(frames[0]['depth_shape'],(3,3))
        np.testing.assert_array_equal(frames[0]['world_from_camera'],np.eye(4))
        self.assertAlmostEqual(frozen.metadata['map_age_seconds'],.1)
        self.assertEqual(frozen.metadata['mapping_health']['status'],'ok')
        self.assertEqual(frozen.metadata['mapping_diagnostics']['source_type'],'rgbd')

    def test_capture_uses_one_service_clock_and_snapshot_without_dense_access_or_extraction(self):
        source = self.prepared()
        original = source.snapshot
        calls = []
        def once(*,now=None):
            calls.append(now)
            return original(now=now)
        source.snapshot = once
        with (patch.object(source._processor._extractor,'extract',side_effect=AssertionError('Owner extracted')),
             patch('robot_skill_stack.world.spatial_mapping.runtime.MappingReadView.states',
                   new=property(lambda self: (_ for _ in ()).throw(AssertionError('Owner scanned'))))):
            frozen = capture_frozen_rgbd(source)
        self.assertEqual(calls,[100.2])
        self.assertEqual(frozen.boxes[0].id,source._published_obstacles.boxes[0].id)

    def test_later_map_publications_reset_or_close_cannot_change_frozen_world(self):
        source = self.prepared()
        frozen = capture_frozen_rgbd(source)
        original_boxes = frozen.boxes
        original_query = frozen.query([0.,0.,1.]).state
        self.publish(source,3,100.15)
        source.reset()
        source.close()
        self.clock[0] = 1000.
        self.assertIs(frozen.boxes,original_boxes)
        self.assertEqual(frozen.generation,0)
        self.assertEqual(frozen.timestamp,100.1)
        self.assertEqual(frozen.query([0.,0.,1.]).state,original_query)

    def test_unknown_free_occupied_and_out_of_bounds_remain_distinct(self):
        frozen = capture_frozen_rgbd(self.prepared())
        points = [[0.,0.,.9],[0.,0.,1.],[.04,.04,.9],[.5,0.,1.]]
        values = frozen.query_many(points)
        np.testing.assert_array_equal(values.states,[OccupancyState.FREE, OccupancyState.OCCUPIED,
            OccupancyState.UNKNOWN,OccupancyState.OUT_OF_BOUNDS])
        self.assertFalse(values.observed[2]); self.assertFalse(values.observed[3])
        with self.assertRaises(ValueError):values.states.setflags(write=True)

    def test_metadata_box_and_camera_buffers_are_deeply_immutable(self):
        source = self.prepared()
        frozen = capture_frozen_rgbd(source)
        with self.assertRaises(FrozenInstanceError):frozen.timestamp = 900.
        with self.assertRaises(FrozenInstanceError):frozen.boxes[0].center = (1.,2.,3.)
        with self.assertRaises(TypeError):frozen.metadata['mapping_health']['status']='error'
        with self.assertRaises(TypeError):frozen.metadata['source_frames'][0]['camera_id']='wrong'
        with self.assertRaises(ValueError):frozen.metadata['source_frames'][0]['intrinsics'].setflags(write=True)
        source.diagnostics = MappingProxyType({})
        self.assertEqual(frozen.metadata['mapping_diagnostics']['source_type'],'rgbd')

    def test_disabled_wrong_source_and_missing_filter_fail_without_fallback(self):
        for field,value,code in (('enabled',False,'disabled'),('source','ground_truth','wrong_source'),
                                ('automatic_robot_filter',False,'filter_required')):
            with self.subTest(field=field):
                source = self.prepared()
                source.settings = replace(source.settings,**{field:value})
                self.rejected(source,code)
        self.rejected(None,'unavailable')

    def test_no_accepted_publication_cannot_be_an_empty_planning_world(self):
        self.rejected(self.service(),'no_publication')

    def test_stale_and_future_original_timestamps_are_not_refreshed(self):
        source = self.prepared()
        self.clock[0] = 101.1
        self.rejected(source,'stale')
        self.assertEqual(source._published.timestamp,100.1)
        self.clock[0] = 100.05
        self.rejected(source,'stale')

    def test_selected_evidence_expiry_rejects_even_when_source_timestamp_is_fresh(self):
        source = self.prepared(ttl=.2)
        self.clock[0] = min(box.expires_at for box in source._published_obstacles.boxes)
        self.assertLess(self.clock[0]-source._published.timestamp,source.settings.stale_after)
        self.rejected(source,'expired_evidence')

    def test_generation_change_during_capture_is_rejected_before_acceptance(self):
        source = self.prepared()
        original = source.snapshot
        def changed(*,now=None):
            view = original(now=now)
            source.reset()
            return view
        source.snapshot = changed
        self.rejected(source,'generation_changed')

    def test_packet_revision_generation_and_timestamp_must_match_one_publication(self):
        for key,change in (('revision',1),('generation',1),('timestamp',.01)):
            with self.subTest(field=key):
                source = self.prepared()
                old = source._published_obstacles
                source._published_obstacles = replace(old,**{key:getattr(old,key)+change})
                self.rejected(source,'product_mismatch')

    def test_worker_failure_and_camera_failure_are_authoritative_over_old_fresh_evidence(self):
        source = self.prepared()
        source.submit(self.payload(source,3,100.15))
        tag = source._worker.pending[1]
        source._worker.result = WorkResult(tag,None,'ValueError: worker failed',100.,100.,100.1)
        self.assertIsNone(source.poll())
        error = self.rejected(source,'unhealthy')
        self.assertEqual(error.details['health']['reason'],'worker_failure')
        source = self.prepared()
        source.note_source_failure('camera acquisition failed')
        error = self.rejected(source,'unhealthy')
        self.assertEqual(error.details['health']['reason'],'acquisition_failure')

    def test_no_new_frame_and_closed_service_are_not_initially_healthy(self):
        source = self.prepared()
        source.note_no_frame()
        self.rejected(source,'unhealthy')
        source = self.prepared()
        source.close()
        self.rejected(source,'unhealthy')

    def test_packet_missing_and_zero_usable_evidence_are_explicit_source_failures(self):
        source = self.prepared()
        source._published_obstacles = None
        self.rejected(source,'no_publication')
        source = self.prepared()
        packet = source._published_obstacles
        diagnostics = dict(packet.diagnostics,occupied_voxels=0,supported_voxels=0,
            represented_voxels=0,omitted_voxels=0,candidate_boxes=0,proxy_count=0)
        source._published_obstacles = replace(packet,boxes=(),diagnostics=diagnostics)
        self.rejected(source,'no_usable_evidence')

    def test_duplicate_cuboids_and_inflated_voxel_dimensions_fail_closed(self):
        source = self.prepared()
        packet = source._published_obstacles
        source._published_obstacles = replace(packet,boxes=(packet.boxes[0],packet.boxes[0]))
        self.rejected(source,'invalid_product')
        source = self.prepared()
        packet = source._published_obstacles
        old = packet.boxes[0]
        source._published_obstacles = replace(packet,boxes=(replace(old,dimensions=(old.dimensions[0]+.01,*old.dimensions[1:])),))
        self.rejected(source,'invalid_product')

    def test_wrong_budget_is_not_silently_truncated_or_repacked(self):
        source = self.prepared()
        self.rejected(source,'budget_mismatch',max_obstacles=128)
        source = self.prepared(maximum=128)
        frozen = capture_frozen_rgbd(source,max_obstacles=128)
        self.assertEqual(frozen.metadata['max_obstacles'],128)

    def test_distinct_ids_must_not_double_count_overlapping_occupied_cells(self):
        source = self.prepared()
        spec, packet = source.settings.grid_spec(), source._published_obstacles
        boxes = []
        for start,stop in (([1,1,1],[3,3,3]),([2,2,2],[4,4,4])):
            minimum = np.asarray(spec.bounds_min)+np.asarray(start)*spec.voxel_size
            maximum = np.asarray(spec.bounds_min)+np.asarray(stop)*spec.voxel_size
            identifier = ':'.join(str(value) for pair in zip(start,stop) for value in pair)
            boxes.append(ObstacleBox(identifier,tuple((minimum+maximum)/2),
                tuple(maximum-minimum),packet.boxes[0].expires_at,8))
        source._published_obstacles = replace(packet,boxes=tuple(boxes))
        error = self.rejected(source,'invalid_product')
        self.assertIn('overlap',str(error))
        self.assertEqual(error.details['obstacle_ids'],tuple(box.id for box in boxes))

    def test_short_boundary_voxel_dimensions_are_preserved_exactly(self):
        source = self.service(bounds_min=(-.015,-.015,.8),bounds_max=(.015,.015,1.005))
        self.publish(source,depth=1.003)
        frozen = capture_frozen_rgbd(source)
        old = source._published_obstacles.boxes[0]
        self.assertEqual(frozen.boxes[0].dimensions,old.dimensions)
        self.assertLess(frozen.boxes[0].dimensions[2],source.settings.voxel_size)
        self.assertEqual(frozen.boxes[0].center,old.center)
        self.assertEqual(frozen.boxes[0].orientation_wxyz,(1.,0.,0.,0.))

    def test_support_priority_does_not_silently_remove_selected_camera_cells(self):
        source = self.prepared(support=1.01)
        packet = source._published_obstacles
        self.assertGreater(packet.diagnostics['selected_floor_boxes'],0)
        frozen = capture_frozen_rgbd(source)
        self.assertEqual(len(frozen.boxes),len(packet.boxes))
        self.assertEqual(frozen.metadata['extraction']['represented_floor_voxels'],packet.diagnostics['represented_floor_voxels'])

    def test_robot_filter_and_camera_identity_mismatch_are_not_accepted(self):
        source = self.prepared()
        diagnostics = dict(source.diagnostics)
        diagnostics['robot_filter'] = dict(diagnostics['robot_filter'],state='error')
        source.diagnostics = diagnostics
        self.rejected(source,'filter_required')
        source = self.prepared()
        diagnostics = dict(source.diagnostics)
        diagnostics['source_frames'] = (dict(diagnostics['source_frames'][0],acquired_at=100.15),diagnostics['source_frames'][1])
        # Keep one original minimum; replacing the other camera ID is still a mismatch.
        diagnostics['source_frames'] = (diagnostics['source_frames'][0],dict(diagnostics['source_frames'][1],camera_id='/other'))
        source.diagnostics = diagnostics
        self.rejected(source,'invalid_product')

    def test_constructor_owns_sequence_inputs(self):
        center,dimensions = [0.,0.,1.],[.01,.01,.01]
        box = FrozenRgbdBox('0:1:0:1:0:1',center,dimensions,101.,1)
        center[0]=9.;dimensions[0]=9.
        self.assertEqual(box.center,(0.,0.,1.))
        self.assertEqual(box.dimensions,(.01,.01,.01))


if __name__ == '__main__':unittest.main()
