"""Ordinary reconstructed requests own bounded readiness, never exposure search."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.manipulation.motion.global_rrt import GlobalRrtStrategy
from robot_skill_stack.integrations.isaac.manipulation.motion.collision_model import load_enhanced_collision_model
from robot_skill_stack.integrations.isaac.manipulation.motion.frozen_rgbd import capture_frozen_rgbd_world
from robot_skill_stack.runtime.worker import WorkResult
from robot_skill_stack.world.spatial_mapping.reconstruction import InferredGeometry
from tests.unit import test_frozen_rgbd_world as frozen_tests
from tests.unit.test_global_frozen_rgbd_v5 import support
from tests.unit.test_global_motion_strategy import Backend,Planner,SolverScope
from tests.unit.test_rgbd_reconstruction import model


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture=frozen_tests.FrozenRgbdWorldTests();self.source=self.fixture.prepared(support=0.)
        self.addCleanup(self.fixture.doCleanups)
        self.current=self.source;self.number=2;self.attach()
        self.backend=Backend();self.backend.robot.q[[0,8]]=.003
        self.planner=Planner();self.planner.diagnostics['sdk_configuration_sha256']={
            'robot_description_path':load_enhanced_collision_model().descriptor_sha256}
        self.factories=[];self.captures=[]
        def capture(backend,provider,**options):
            self.assertFalse(self.backend.robot.actions)
            self.captures.append(provider())
            return capture_frozen_rgbd_world(backend,provider,support_capture=lambda *a,**k:support(),**options)
        def factory(*a,**k):self.factories.append(True);return self.planner
        self.strategy=GlobalRrtStrategy(self.backend,collision_model='enhanced',obstacle_source='frozen_rgbd',
            rgbd_representation='reconstructed',mapping_source_provider=lambda:self.current,
            rgbd_world_capture=capture,planner_factory=factory,action_factory=lambda **k:NS(**k),
            clock=lambda:self.fixture.clock[0],solver_scope_factory=lambda backend:SolverScope())
        self.addCleanup(self.strategy.close)
        self.target=Pose([.1,0,0],[1.,0,0,0])
        self.backend.on_update=self.update

    def attach(self,status='qualified'):
        self.source._published=replace(self.source._published,inferred=InferredGeometry(
            () if status in ('rejected','empty') else (model(center=(0,0,1),dimensions=(.5,.5,.5)),),
            'unsupported hidden bound' if status=='rejected' else None))

    def publish(self,status='qualified'):
        self.number+=1
        self.fixture.publish(self.source,self.number,self.fixture.clock[0]-.05)
        self.attach(status)

    def update(self,backend):
        self.fixture.clock[0]+=.1
        if self.strategy._state=='waiting_for_rgbd':
            self.assertFalse(backend.robot.actions)
            self.publish()

    async def failed(self,code):
        result=await self.strategy.move_to_pose_async(self.target)
        self.assertFalse(result.ok);self.assertEqual(result.details['failure_code'],code)
        self.assertFalse(self.backend.robot.actions);self.assertFalse(self.factories)
        return result

    async def test_fresh_current_source_deterministically_waits_for_one_new_publication(self):
        initial=self.source.snapshot();self.assertTrue(initial.health['fresh'])
        result=await self.strategy.move_to_pose_async(self.target)
        self.assertTrue(result.ok);audit=result.details['rgbd_source_admission']
        self.assertEqual(audit['initial_identity'][1],initial.revision)
        self.assertEqual(audit['selected_identity'][1],initial.revision+1)
        self.assertEqual(audit['updates'],1);self.assertEqual(len(self.captures),1)
        self.assertEqual(result.details['planning_source_captures'],1)
        self.assertTrue(self.backend.robot.actions)

    async def test_stale_current_source_accepts_new_original_timestamp_without_mutating_old_view(self):
        old=self.source.snapshot();self.fixture.clock[0]+=2.
        result=await self.strategy.move_to_pose_async(self.target)
        self.assertTrue(result.ok)
        frozen=self.strategy.last_plan_trace['frozen_rgbd']
        self.assertNotEqual(frozen.timestamp,old.timestamp)
        self.assertEqual(old.timestamp,100.1)
        self.assertLess(frozen.age_seconds,1.)

    async def test_no_publication_times_out_without_capture_planner_or_commands(self):
        self.backend.on_update=lambda backend:self.fixture.clock.__setitem__(0,self.fixture.clock[0]+1.)
        result=await self.failed('planning_source_timeout')
        self.assertFalse(self.captures);self.assertGreaterEqual(result.details['rgbd_source_admission']['wait_seconds'],5.)

    async def test_stalled_update_is_bounded_by_async_timeout(self):
        values=iter([0.,4.99,5.01])
        self.strategy._clock=lambda:next(values)
        async def stalled():await asyncio.Event().wait()
        self.backend._next_update=stalled
        await self.failed('planning_source_timeout')
        self.assertFalse(self.captures)

    async def test_cancellation_propagates_and_cancels_pending_update_without_commands(self):
        started=asyncio.Event();closed=[]
        async def stalled():
            started.set()
            try:await asyncio.Event().wait()
            finally:closed.append(True)
        self.backend._next_update=stalled
        task=asyncio.create_task(self.strategy.move_to_pose_async(self.target))
        await started.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(closed);self.assertFalse(self.captures);self.assertFalse(self.backend.robot.actions)

    async def test_stop_rejects_wait_before_capture(self):
        self.backend.on_update=lambda backend:setattr(backend,'playing',False)
        await self.failed('lifecycle_invalidated')
        self.assertFalse(self.captures)

    async def test_close_rejects_wait_before_capture(self):
        self.backend.on_update=lambda backend:self.strategy.close()
        await self.failed('lifecycle_invalidated')
        self.assertFalse(self.captures)

    async def test_articulation_invalidation_rejects_wait_before_capture(self):
        self.backend.on_update=lambda backend:setattr(backend,'is_ready',False)
        await self.failed('lifecycle_invalidated')
        self.assertFalse(self.captures)

    async def test_stop_play_reset_cannot_resume_wait_on_new_generation(self):
        self.backend.on_update=lambda backend:self.source.reset()
        await self.failed('planning_source_generation_changed')
        self.assertFalse(self.captures)

    async def test_source_error_and_recovery_cannot_hide_failure_barrier(self):
        def error(backend):
            self.source.note_source_failure('robot model changed')
            self.fixture.clock[0]+=.1;self.publish()
        self.backend.on_update=error
        await self.failed('planning_source_unavailable')
        self.assertEqual(self.source.health['failures'],1)

    async def test_worker_error_rejects_wait_without_resetting_service(self):
        def error(backend):
            self.source._worker.result=WorkResult((self.source.generation,self.source._barrier),None,
                                                 'worker failed',100.,100.,100.1)
            self.source.poll()
        self.backend.on_update=error
        await self.failed('planning_source_unavailable')
        self.assertEqual(self.source.health['failures'],1)

    async def test_first_selected_geometry_rejection_is_not_an_exposure_search(self):
        def rejected(backend):self.fixture.clock[0]+=.1;self.publish('rejected')
        self.backend.on_update=rejected
        result=await self.failed('planning_source_reconstruction')
        self.assertEqual(len(self.captures),1)
        self.assertEqual(result.details['rgbd_source_admission']['updates'],1)
        self.assertEqual(self.source.health['published'],3)

    async def test_selected_publication_expiring_during_capture_is_not_retried(self):
        original=self.strategy._rgbd_world_capture
        def expired(*a,**k):self.fixture.clock[0]+=1.1;return original(*a,**k)
        self.strategy._rgbd_world_capture=expired
        await self.failed('planning_source_stale')
        self.assertEqual(len(self.captures),1)

    async def test_service_or_identity_substitution_cannot_cross_admission(self):
        original=self.strategy._rgbd_world_capture
        def mismatch(*a,**k):
            snapshot,frozen=original(*a,**k)
            return snapshot,NS(generation=frozen.generation,revision=frozen.revision+1,timestamp=frozen.timestamp)
        self.strategy._rgbd_world_capture=mismatch
        await self.failed('planning_source_changed')
        self.assertEqual(len(self.captures),1)

    async def test_replaced_service_is_rejected_before_capture(self):
        def replaced(backend):self.update(backend);self.current=object()
        self.backend.on_update=replaced
        await self.failed('planning_source_changed')
        self.assertFalse(self.captures)

    async def test_multiple_publications_between_updates_cannot_skip_first_candidate(self):
        def skipped(backend):self.update(backend);self.fixture.clock[0]+=.1;self.publish()
        self.backend.on_update=skipped
        await self.failed('planning_source_changed')
        self.assertFalse(self.captures)

    async def test_occupied_only_and_gt_do_not_enter_admission(self):
        for source in ('frozen_rgbd','static'):
            self.strategy.obstacle_source=source;self.strategy.rgbd_representation='occupied'
            self.strategy._await_rgbd_source=Mock(side_effect=AssertionError('Readiness must remain opt-in'))
            self.strategy._world_capture=lambda backend:support()
            if source=='frozen_rgbd':self.attach()
            result=await self.strategy.move_to_pose_async(self.target)
            self.assertTrue(result.ok)
            self.assertNotIn('rgbd_source_admission',result.details)
            self.backend.robot.actions.clear()

    async def test_late_rejected_completion_does_not_count_as_new_publication(self):
        def late(backend):
            self.fixture.clock[0]+=.1
            if backend.updates==1:
                self.source._worker.result=WorkResult((self.source.generation,self.source._barrier-1),
                                                      None,None,100.,100.,100.1)
                self.source.poll()
            elif self.strategy._state=='waiting_for_rgbd':self.publish()
        self.backend.on_update=late
        result=await self.strategy.move_to_pose_async(self.target)
        self.assertTrue(result.ok)
        self.assertEqual(result.details['rgbd_source_admission']['updates'],2)
        self.assertEqual(len(self.captures),1)

    async def test_empty_inference_does_not_search_and_keeps_occupied_cells(self):
        def empty(backend):self.fixture.clock[0]+=.1;self.publish('empty')
        self.backend.on_update=empty
        result=await self.strategy.move_to_pose_async(self.target)
        self.assertTrue(result.ok)
        frozen=self.strategy.last_plan_trace['frozen_rgbd']
        snapshot=self.strategy.last_plan_trace['snapshot']
        self.assertEqual([o.path for o in snapshot.obstacles if o.source_kind=='rgbd_occupied_cells'],
                         ['/__rgbd_occupied__/'+box.id for box in frozen.boxes])
        self.assertEqual(len(self.captures),1)

    async def test_fingers_changed_during_admission_reject_before_capture(self):
        def opening(backend):self.update(backend);backend.robot.q[[0,8]]=.04
        self.backend.on_update=opening
        await self.failed('unsupported_gripper_configuration')
        self.assertFalse(self.captures)
