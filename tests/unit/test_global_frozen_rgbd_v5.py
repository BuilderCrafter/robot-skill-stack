"""Frozen-source isolation and static executor behavior without simulator SDK."""
import asyncio
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.integrations.isaac.manipulation.motion.frozen_rgbd import (
    SUPPORT_PLANE_PATHS, capture_frozen_rgbd_world, capture_known_support, planning_world_from_frozen,
)
from robot_skill_stack.integrations.isaac.manipulation.motion.global_rrt import GlobalRrtStrategy
from robot_skill_stack.integrations.isaac.manipulation.motion.collision_model import load_enhanced_collision_model
from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import (
    StaticObstacle, StaticWorldError, StaticWorldSnapshot, obstacle_from_primitive,
)
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure
from robot_skill_stack.common.types import Pose
from tests.unit.test_global_motion_strategy import Backend, Planner, SolverScope
from tests.unit.test_global_static_world import Collision, Cube, Mesh, Prim, Rigid


MODULE = 'robot_skill_stack.integrations.isaac.manipulation.motion.frozen_rgbd'


def support(timestamp=10.):
    obstacle, domain = obstacle_from_primitive(SUPPORT_PLANE_PATHS[0], 'Plane', np.eye(4))
    return StaticWorldSnapshot((obstacle,), timestamp, 1., '/Franka', plane_domains=(domain,))


def frozen():
    return NS(timestamp=10., metadata=dict(source_type='rgbd', generation=2, revision=7,
        age_seconds=.1, policy={'unknown':'preserved'}), boxes=(NS(id='1:2:3:4:5:6',
        center=(.15,.35,.505), dimensions=(.1,.1,.01), orientation_wxyz=(1.,0.,0.,0.)),))


class NumericSourceTests(unittest.TestCase):
    def test_exact_camera_box_preserves_world_meters_ids_boundaries_and_source_time(self):
        camera = frozen()
        world = planning_world_from_frozen(camera, support())
        self.assertEqual(world.timestamp, camera.timestamp)
        box = world.obstacles[1]
        self.assertEqual(box.path, '/__rgbd_occupied__/1:2:3:4:5:6')
        self.assertEqual(box.position, camera.boxes[0].center)
        self.assertEqual(box.dimensions, camera.boxes[0].dimensions)
        self.assertEqual(box.maximum_surface_error_m, 0.)
        self.assertEqual(box.source_kind, 'rgbd_occupied_cells')
        self.assertEqual(world.source_metadata['camera_obstacle_count'], 1)
        self.assertEqual(world.source_metadata['known_support_paths'], SUPPORT_PLANE_PATHS)
        camera.metadata['policy']['unknown'] = 'free'
        self.assertEqual(world.source_metadata['policy']['unknown'], 'preserved')
        with self.assertRaises(TypeError):world.source_metadata['revision'] = 8
        with self.assertRaises(FrozenInstanceError):world.timestamp = 11.
        self.assertEqual(world.as_dict()['source_metadata']['policy']['unknown'], 'preserved')

    def test_broad_gt_snapshot_cannot_be_substituted_for_known_support(self):
        known = support()
        fixture = StaticObstacle('/World/Fixture', 'cuboid', (.4,0.,.3), dimensions=(.1,.1,.1),source_kind='Cube')
        for items in ((fixture,), (*known.obstacles,fixture)):
            with self.subTest(items=items), self.assertRaisesRegex(StaticWorldError, 'scene obstacle'):
                planning_world_from_frozen(frozen(), StaticWorldSnapshot(items,10.,1.,'/Franka',plane_domains=known.plane_domains))

    def test_unknown_plane_name_or_missing_domain_cannot_expand_support_allowlist(self):
        for path in ('/World/Fixture', SUPPORT_PLANE_PATHS[0]):
            plane, domain = obstacle_from_primitive(path, 'Plane', np.eye(4))
            world = StaticWorldSnapshot((plane,),10.,1.,'/Franka',plane_domains=() if path==SUPPORT_PLANE_PATHS[0] else (domain,))
            with self.assertRaises(StaticWorldError):planning_world_from_frozen(frozen(),world)

    def test_duplicate_camera_ids_cannot_create_duplicate_numeric_obstacles(self):
        camera = frozen()
        camera.boxes = camera.boxes*2
        with self.assertRaisesRegex(StaticWorldError, 'Duplicate'):
            planning_world_from_frozen(camera,support())

    def test_known_support_never_traverses_or_reads_fixture_prims(self):
        floor = Prim(SUPPORT_PLANE_PATHS[0],kind='Plane')
        requested = []
        def get(path):
            requested.append(path)
            if path!=SUPPORT_PLANE_PATHS[0]:raise AssertionError('Fixture GT read')
            return floor
        stage = NS(GetPrimAtPath=get,Traverse=Mock(side_effect=AssertionError('No scene traversal')))
        usd = NS(TimeCode=NS(Default=lambda:None))
        geom = NS(GetStageMetersPerUnit=lambda stage:1.,XformCache=lambda stamp:NS(GetLocalToWorldTransform=lambda prim:prim.transform.T))
        physics = NS(CollisionAPI=Collision,RigidBodyAPI=Rigid)
        backend = NS(robot=NS(prim_path='/Franka',get_world_pose=lambda:(np.zeros(3),[1.,0.,0.,0.])))
        with patch(MODULE+'._usd_modules',return_value=(usd,geom,physics)):
            captured = capture_known_support(backend,stage=stage,timestamp=10.)
        self.assertEqual(requested,list(SUPPORT_PLANE_PATHS))
        self.assertEqual([x.source_kind for x in captured.obstacles],['Plane'])
        stage.Traverse.assert_not_called()

    def test_fixture_cube_named_like_floor_is_rejected_instead_of_leaking_geometry(self):
        floor = Prim(SUPPORT_PLANE_PATHS[0],kind='Cube')
        stage = NS(GetPrimAtPath=lambda path:floor)
        usd = NS(TimeCode=NS(Default=lambda:None))
        geom = NS(GetStageMetersPerUnit=lambda stage:1.,XformCache=lambda stamp:None)
        physics = NS(CollisionAPI=Collision,RigidBodyAPI=Rigid)
        backend = NS(robot=NS(prim_path='/Franka',get_world_pose=lambda:(np.zeros(3),[1.,0.,0.,0.])))
        with patch(MODULE+'._usd_modules',return_value=(usd,geom,physics)):
            with self.assertRaisesRegex(StaticWorldError,'actual Plane'):
                capture_known_support(backend,stage=stage,timestamp=10.)

    def test_existing_static_serialization_stays_camera_free(self):
        world = support()
        self.assertNotIn('source_metadata',world.as_dict())


class SourceAdapterTests(unittest.TestCase):
    def test_actual_worker_publication_is_translated_once_and_retained_after_source_reset(self):
        from tests.unit.test_frozen_rgbd_world import FrozenRgbdWorldTests
        fixture = FrozenRgbdWorldTests()
        self.addCleanup(fixture.doCleanups)
        source = fixture.prepared()
        provider = Mock(return_value=source)
        known_support = Mock(side_effect=lambda backend, timestamp:support(timestamp))
        world, camera = capture_frozen_rgbd_world(object(),provider,support_capture=known_support)
        provider.assert_called_once_with()
        known_support.assert_called_once()
        self.assertEqual(world.timestamp,100.1)
        self.assertEqual(world.source_metadata['camera_ids'],('/A','/B'))
        self.assertEqual(world.source_metadata['revision'],camera.revision)
        self.assertEqual(len(world.obstacles),len(camera.boxes)+1)
        for actual, box in zip(world.obstacles[1:],camera.boxes):
            self.assertEqual(actual.position,box.center)
            self.assertEqual(actual.dimensions,box.dimensions)
        original = world.as_dict()
        source.reset()
        source.close()
        self.assertEqual(world.as_dict(),original)
        self.assertEqual(camera.timestamp,100.1)

    def test_rejected_source_retains_distinct_reason_and_never_reads_known_support(self):
        known_support = Mock(side_effect=AssertionError('Support read before source acceptance'))
        provider = Mock(return_value=None)
        with self.assertRaises(PlanningFailure) as caught:
            capture_frozen_rgbd_world(object(),provider,support_capture=known_support)
        self.assertEqual(caught.exception.code,'planning_source_unavailable')
        provider.assert_called_once_with()
        known_support.assert_not_called()
        from robot_skill_stack.world.spatial_mapping.frozen import FrozenRgbdSourceError
        with patch('robot_skill_stack.world.spatial_mapping.frozen.capture_frozen_rgbd',
                   side_effect=FrozenRgbdSourceError('stale','Original exposure is stale',{'timestamp':1.})):
            with self.assertRaises(PlanningFailure) as caught:
                capture_frozen_rgbd_world(object(),provider,support_capture=known_support)
        self.assertEqual(caught.exception.code,'planning_source_stale')
        self.assertEqual(caught.exception.details['timestamp'],1.)
        known_support.assert_not_called()

    def test_support_failure_has_its_own_source_diagnostic(self):
        with patch('robot_skill_stack.world.spatial_mapping.frozen.capture_frozen_rgbd',return_value=frozen()):
            with self.assertRaises(PlanningFailure) as caught:
                capture_frozen_rgbd_world(object(),lambda:object(),
                    support_capture=Mock(side_effect=StaticWorldError('Support unavailable')))
        self.assertEqual(caught.exception.code,'planning_source_support')


class FrozenExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setup_strategy(self, capture):
        backend, planner = Backend(), Planner()
        backend.robot.q[[0,8]]=.003
        planner.diagnostics['sdk_configuration_sha256'] = {
            'robot_description_path':load_enhanced_collision_model().descriptor_sha256}
        factories=[]
        def factory(backend, **kwargs):factories.append(kwargs);return planner
        strategy = GlobalRrtStrategy(backend,collision_model='enhanced',obstacle_source='frozen_rgbd',mapping_source_provider=object(),
            rgbd_world_capture=capture,planner_factory=factory,
            world_capture=Mock(side_effect=AssertionError('GT traversal forbidden')),
            action_factory=lambda **kw:NS(**kw),solver_scope_factory=lambda backend:SolverScope())
        return backend,planner,strategy,factories

    async def test_mapping_publications_failure_and_reset_after_capture_cannot_refresh_active_world(self):
        calls=[]
        world = planning_world_from_frozen(frozen(),support())
        mutable_service=NS(generation=2,revision=7,healthy=True)
        frozen_source=NS(generation=2,revision=7)
        def capture(*args,**kwargs):calls.append(1);return world,frozen_source
        backend,planner,strategy,factories=self.setup_strategy(capture)
        backend.on_update=lambda backend:(setattr(mutable_service,'generation',99),setattr(mutable_service,'healthy',False))
        result=await strategy.move_to_pose_async(Pose([.1,0.,0.],[1.,0.,0.,0.]))
        self.assertTrue(result.ok)
        self.assertEqual(len(calls),1)
        self.assertEqual(result.details['planning_source_captures'],1)
        self.assertEqual(result.details['active_world_obstacle_refreshes'],0)
        self.assertIs(strategy.last_plan_trace['snapshot'],world)
        self.assertIs(strategy.last_plan_trace['frozen_rgbd'],frozen_source)
        self.assertEqual(world.source_metadata['revision'],7)
        self.assertTrue(planner.closed)

    async def test_source_failure_issues_no_arm_commands_or_planner_fallback(self):
        def capture(*args,**kwargs):raise PlanningFailure('planning_source_stale','Stale camera evidence')
        backend,planner,strategy,factories=self.setup_strategy(capture)
        result=await strategy.move_to_pose_async(Pose([.1,0.,0.],[1.,0.,0.,0.]))
        self.assertFalse(result.ok)
        self.assertEqual(result.details['failure_code'],'planning_source_stale')
        self.assertEqual(backend.robot.actions,[])
        self.assertEqual(factories,[])

    async def test_stop_during_source_acceptance_fails_before_creating_planner(self):
        state={}
        def capture(*args,**kwargs):
            state['backend'].playing=False
            return support(),NS()
        backend,planner,strategy,factories=self.setup_strategy(capture)
        state['backend']=backend
        result=await strategy.move_to_pose_async(Pose([.1,0.,0.],[1.,0.,0.,0.]))
        self.assertFalse(result.ok)
        self.assertEqual(result.details['failure_code'],'lifecycle_invalidated')
        self.assertEqual(backend.robot.actions,[])
        self.assertEqual(factories,[])

    async def test_stopped_or_closed_strategy_does_not_touch_source(self):
        for close in (False,True):
            with self.subTest(close=close):
                capture = Mock(side_effect=AssertionError('Invalid lifecycle must precede source access'))
                backend,planner,strategy,factories=self.setup_strategy(capture)
                if close:strategy.close()
                else:backend.playing=False
                result=await strategy.move_to_pose_async(Pose([.1,0.,0.],[1.,0.,0.,0.]))
                self.assertEqual(result.details['failure_code'],'lifecycle_invalidated')
                capture.assert_not_called()
                self.assertEqual(backend.robot.actions,[])
                self.assertEqual(factories,[])


if __name__=='__main__':unittest.main()
