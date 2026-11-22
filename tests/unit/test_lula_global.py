"""Pre-execution failures and actual joint-space contracts without Isaac imports."""
from pathlib import Path
from dataclasses import replace
import hashlib
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.manipulation.motion.lula_global import (
    LulaGlobalPlanner, adjacent_self_collision_pairs, orientation_error, quaternion_matrix,
)
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure
from robot_skill_stack.integrations.isaac.manipulation.motion.collision_model import load_enhanced_collision_model


NAMES = tuple(f'panda_joint{i}' for i in range(1, 8))


class PlannerFixture:
    def __init__(self, folder, *, collision=None, self_pairs=(), ik_success=True, orientation_ok=True,
                 search_success=True, trajectory_collision=False, wrong_order=False, smoothed_collision=False,
                 changed_endpoint=False):
        self.events = []
        self.urdf = Path(folder)/'robot.urdf'
        self.urdf.write_text('<robot><link name="base"/><link name="arm"/><link name="tool"/>'
            '<joint name="j" type="revolute"><parent link="base"/><child link="arm"/></joint>'
            '<joint name="w" type="fixed"><parent link="arm"/><child link="tool"/></joint></robot>')
        events = self.events
        target_q = np.zeros(7);target_q[0] = .2
        model = NS(default_c_space_configuration=lambda: np.zeros(7))

        class World:
            def add_world_view(self):return object()
            def add_obstacle(self, obstacle, pose):
                events.append(('obstacle', obstacle, pose));return len(events)
            def remove_obstacle(self, handle):events.append(('remove',handle))

        class RRT:
            def __init__(self, **config):
                events.append(('rrt_config', config))
                self._world, self._robot_description = World(), model
                self._rrt = NS(set_param=lambda name,value: events.append(('native_param',name,value)) or True,
                    update_world_view=lambda: events.append(('world_update',)))
            def get_active_joints(self):return list(NAMES)
            def set_robot_base_pose(self, position, orientation):events.append(('rrt_base',position.copy(),orientation.copy()))
            def set_max_iterations(self, count):events.append(('iterations',count))
            def set_random_seed(self, seed):events.append(('wrapper_seed',seed))
            def set_cspace_target(self, q):self.goal = q.copy();events.append(('cspace_goal',q.copy()))
            def compute_path(self, q, watched):
                events.append(('rrt_call',q.copy(),watched.copy()))
                return np.array([q,self.goal]) if search_success else None

        class Kinematics:
            def __init__(self, **config):events.append(('kinematics_config', config))
            def get_joint_names(self):return tuple(reversed(NAMES)) if wrong_order else NAMES
            def set_robot_base_pose(self, *args):pass
            def get_cspace_position_limits(self):return np.full(7,-1.),np.full(7,1.)
            def get_cspace_velocity_limits(self):return np.ones(7)
            def get_cspace_acceleration_limits(self):return np.ones(7)
            def get_cspace_jerk_limits(self):return np.ones(7)
            def set_default_cspace_seeds(self, values):pass
            def compute_inverse_kinematics(self, frame, p, orientation, **kwargs):
                events.append(('ik',frame,p.copy(),orientation.copy(),kwargs));return target_q.copy(),ik_success
            def compute_forward_kinematics(self, frame, q):
                if frame == 'panda_link0': return np.zeros(3), np.eye(3)
                return np.array([.3+q[0],0,.4]), np.eye(3) if orientation_ok else quaternion_matrix([0,1,0,0])

        class Trajectory:
            start_time,end_time = 0.,2.
            def __init__(self):self.trajectory = self
            def get_joint_targets(self,t):
                q=target_q*(t/2.)
                if changed_endpoint:q[0]*=1.5
                if smoothed_collision:q[1]=.2*np.sin(np.pi*t/2.)
                return q,target_q/2.
            def eval(self,t,derivative):return np.zeros(7)

        class Generator:
            def __init__(self, **kwargs):events.append(('generator_config', kwargs))
            def get_active_joints(self):return NAMES
            def compute_c_space_trajectory(self,route):
                events.append(('trajectory',route.copy()))
                return Trajectory()

        class Inspector:
            def update_world_view(self):pass
            def in_collision_with_obstacle(self,q):
                if trajectory_collision and .08 < q[0] < .12:return True
                if smoothed_collision and q[1]>.1:return True
                return bool(collision(q)) if collision else False
            def frames_in_self_collision(self,q):return self_pairs
            def collision_sphere_positions(self,q):return [np.zeros(3)]
            def collision_sphere_radii(self):return [.01]

        class Obstacle:
            Type=NS(CUBE='cube',SPHERE='sphere',CYLINDER='capsule')
            Attribute=NS(SIDE_LENGTHS='dimensions',RADIUS='radius',HEIGHT='height')
            def __init__(self,kind):self.kind,self.attributes=kind,{}
            def set_attribute(self,name,value):self.attributes[name]=value

        lula = NS(create_robot_world_inspector=lambda robot,view:Inspector(), Obstacle=Obstacle,
            create_obstacle=lambda kind:Obstacle(kind), Rotation3=lambda rot:rot,
            Pose3=lambda rot,trans:NS(rotation=rot,translation=trans))
        self.sdk=NS(lula=lula,RRT=RRT,Kinematics=Kinematics,Generator=Generator,
            load_config=lambda *args:dict(robot_description_path=str(self.urdf),urdf_path=str(self.urdf),
                rrt_config_path=str(self.urdf),end_effector_frame_name='right_gripper'))
        self.backend=NS(robot=NS(get_world_pose=lambda:(np.zeros(3),np.array([1,0,0,0]))),
            end_effector_frame='right_gripper')
        self.snapshot=NS(obstacles=(),meters_per_unit=1.,assert_within_domains=lambda p,r:None)
        self.target=Pose([.5,0,.4],[1,0,0,0])


class LulaGlobalTests(unittest.TestCase):
    def fixture(self, **kwargs):
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup)
        return PlannerFixture(temporary.name,**kwargs)

    def plan(self, fixture):
        planner=LulaGlobalPlanner(fixture.backend,seed=17,max_iterations=500,_sdk=fixture.sdk)
        self.addCleanup(planner.close)
        return planner,planner.plan(fixture.target,np.zeros(7),fixture.snapshot,.025,.1)

    def enhanced(self, fixture):
        model = replace(load_enhanced_collision_model(),
            urdf_sha256=hashlib.sha256(fixture.urdf.read_bytes()).hexdigest())
        fixture.backend.robot.dof_names = (*NAMES, 'panda_finger_joint1', 'panda_finger_joint2')
        fixture.backend.robot.get_joint_positions = lambda: np.array([0.]*7+[.003, .003])
        return model

    def test_enhanced_description_is_shared_by_search_ik_and_trajectory_validator(self):
        fixture = self.fixture(); model = self.enhanced(fixture)
        with patch('robot_skill_stack.integrations.isaac.manipulation.motion.lula_global.load_enhanced_collision_model',
                return_value=model):
            planner = LulaGlobalPlanner(fixture.backend, collision_model='enhanced', _sdk=fixture.sdk)
            self.addCleanup(planner.close)
            result = planner.plan(fixture.target, np.zeros(7), fixture.snapshot, .025, .1)
        configs = [row[1] for row in fixture.events if row[0] in ('rrt_config', 'kinematics_config', 'generator_config')]
        self.assertEqual(len(configs), 3)
        for config in configs:
            self.assertEqual(config['robot_description_path'], str(model.descriptor_path))
            self.assertEqual(config['urdf_path'], str(fixture.urdf))
        self.assertEqual(result.diagnostics['global_collision_model'], 'enhanced')
        self.assertTrue(result.diagnostics['base_geometry_preflight']['all_configurations'])

    def test_changed_urdf_cannot_silently_fall_back_to_stock_on_retry(self):
        fixture = self.fixture(); model = self.enhanced(fixture)
        fixture.urdf.write_text('changed asset')
        with patch('robot_skill_stack.integrations.isaac.manipulation.motion.lula_global.load_enhanced_collision_model',
                return_value=model):
            planner = LulaGlobalPlanner(fixture.backend, collision_model='enhanced', _sdk=fixture.sdk)
            self.addCleanup(planner.close)
            for _ in range(2):
                with self.assertRaises(PlanningFailure) as caught: _ = planner.joint_names
                self.assertEqual(caught.exception.code, 'collision_model_invalid')
        self.assertFalse(any(row[0] == 'rrt_config' for row in fixture.events))

    def test_base_preflight_rejects_possible_collision_before_ik_or_search(self):
        from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import StaticObstacle
        fixture = self.fixture(); model = self.enhanced(fixture)
        fixture.snapshot.obstacles = (StaticObstacle('/BaseBall', 'sphere', model.base_center, radius=.01),)
        with patch('robot_skill_stack.integrations.isaac.manipulation.motion.lula_global.load_enhanced_collision_model',
                return_value=model):
            planner = LulaGlobalPlanner(fixture.backend, collision_model='enhanced', _sdk=fixture.sdk)
            self.addCleanup(planner.close)
            with self.assertRaises(PlanningFailure) as caught:
                planner.plan(fixture.target, np.zeros(7), fixture.snapshot, .025, .1)
        self.assertEqual(caught.exception.code, 'base_geometry_rejected')
        self.assertFalse(any(row[0] in ('ik', 'rrt_call', 'trajectory') for row in fixture.events))

    def test_enhanced_pair_certificate_never_excludes_finger_against_link6(self):
        for extra in ((), (('panda_leftfinger', 'panda_link6'),)):
            with self.subTest(extra=extra):
                fixture = self.fixture(self_pairs=(('panda_leftfinger', 'panda_rightfinger'), *extra))
                model = self.enhanced(fixture)
                with patch('robot_skill_stack.integrations.isaac.manipulation.motion.lula_global.load_enhanced_collision_model',
                        return_value=model):
                    planner = LulaGlobalPlanner(fixture.backend, collision_model='enhanced', _sdk=fixture.sdk)
                    self.addCleanup(planner.close)
                    if extra:
                        with self.assertRaises(PlanningFailure) as caught:
                            planner.plan(fixture.target, np.zeros(7), fixture.snapshot, .025, .1)
                        self.assertEqual(caught.exception.code, 'invalid_start')
                        self.assertFalse(any(row[0] == 'rrt_call' for row in fixture.events))
                    else:
                        result = planner.plan(fixture.target, np.zeros(7), fixture.snapshot, .025, .1)
                        self.assertEqual(result.diagnostics['certified_self_collision_exclusions'],
                            [['panda_leftfinger', 'panda_rightfinger']])
        stock = self.fixture(self_pairs=(('panda_leftfinger', 'panda_rightfinger'),))
        with self.assertRaises(PlanningFailure): self.plan(stock)

    def test_preserves_joint_route_full_pose_and_native_seed(self):
        fixture=self.fixture();planner,result=self.plan(fixture)
        self.assertEqual(result.joint_names,NAMES)
        np.testing.assert_array_equal(result.route[0],np.zeros(7))
        self.assertIn(('native_param','seed',17),fixture.events)
        ik=next(row for row in fixture.events if row[0]=='ik')
        np.testing.assert_array_equal(ik[3],[1,0,0,0])
        self.assertEqual(result.diagnostics['trajectory_validation']['collision_validation'],'PASS_SAMPLED_SDK_MODEL')
        self.assertEqual(result.diagnostics['ik_candidates_attempted'],1)
        self.assertGreater(result.diagnostics['configuration_validation_calls'], 0)
        self.assertGreaterEqual(result.diagnostics['configuration_validation_wall_seconds'], 0.)
        self.assertEqual(result.diagnostics['route_validation_seconds'],
            sum(row.get('route_validation_seconds', 0.) for row in result.diagnostics['ik_attempts']))
        self.assertIsNone(result.diagnostics['native_rrt_collision_query_count'])

    def test_colliding_start_never_searches_or_generates_trajectory(self):
        fixture=self.fixture(collision=lambda q:True)
        with self.assertRaises(PlanningFailure) as caught:self.plan(fixture)
        self.assertEqual(caught.exception.code,'invalid_start')
        self.assertFalse(any(row[0] in ('rrt_call','trajectory') for row in fixture.events))

    def test_already_at_full_pose_is_validated_without_motion_generation(self):
        fixture=self.fixture();fixture.target=Pose([.3,0,.4],[1,0,0,0])
        planner,result=self.plan(fixture)
        self.assertEqual(result.duration,0.)
        self.assertIsNone(result.trajectory)
        self.assertEqual(result.route.shape,(1,7))
        self.assertTrue(result.diagnostics['already_at_goal'])
        self.assertFalse(any(row[0] in ('rrt_call','trajectory','ik') for row in fixture.events))

    def test_bad_ik_orientation_is_rejected_before_search(self):
        fixture=self.fixture(orientation_ok=False)
        with self.assertRaises(PlanningFailure) as caught:self.plan(fixture)
        self.assertEqual(caught.exception.code,'invalid_goal')
        self.assertFalse(any(row[0]=='rrt_call' for row in fixture.events))

    def test_colliding_goal_is_rejected_before_search(self):
        fixture=self.fixture(collision=lambda q:q[0]>.15)
        with self.assertRaises(PlanningFailure) as caught:self.plan(fixture)
        self.assertEqual(caught.exception.code,'invalid_goal')
        self.assertEqual(caught.exception.details['ik_attempts'][0]['ik_q'], [.2, 0., 0., 0., 0., 0., 0.])
        self.assertFalse(any(row[0]=='rrt_call' for row in fixture.events))

    def test_exhausted_search_has_no_direct_or_trajectory_fallback(self):
        fixture=self.fixture(search_success=False)
        planner=LulaGlobalPlanner(fixture.backend,seed=17,max_iterations=500,_sdk=fixture.sdk)
        self.addCleanup(planner.close)
        with self.assertRaises(PlanningFailure) as caught:
            planner.plan(fixture.target,np.zeros(7),fixture.snapshot,.025,.1)
        self.assertEqual(caught.exception.code,'search_exhausted')
        self.assertFalse(any(row[0]=='trajectory' for row in fixture.events))
        diagnostics=planner.diagnostics
        self.assertEqual(diagnostics['state'],'failed')
        self.assertEqual(diagnostics['ik_candidates_attempted'],len(diagnostics['ik_attempts']))
        self.assertEqual(diagnostics['rrt_seconds'],sum(row.get('rrt_seconds',0.) for row in diagnostics['ik_attempts']))
        self.assertGreater(diagnostics['rrt_seconds'],0.)
        self.assertEqual(diagnostics['trajectory_generation_seconds'],0.)

    def test_failed_start_records_zero_phase_counts(self):
        fixture=self.fixture(collision=lambda q:True)
        planner=LulaGlobalPlanner(fixture.backend,_sdk=fixture.sdk)
        self.addCleanup(planner.close)
        with self.assertRaises(PlanningFailure):planner.plan(fixture.target,np.zeros(7),fixture.snapshot,.025,.1)
        self.assertEqual(planner.diagnostics['state'],'failed')
        for key in ('ik_candidates_attempted','rrt_seconds','trajectory_generation_seconds'):
            self.assertEqual(planner.diagnostics[key],0)

    def test_midsegment_collision_is_rejected_before_trajectory_generation(self):
        fixture=self.fixture(trajectory_collision=True)
        with self.assertRaises(PlanningFailure) as caught:self.plan(fixture)
        self.assertEqual(caught.exception.code,'route_invalid')
        self.assertIn('route_validation_seconds', caught.exception.details['ik_attempts'][0])
        self.assertGreater(caught.exception.details['configuration_validation_calls'], 0)
        self.assertFalse(any(row[0]=='trajectory' for row in fixture.events))

    def test_safe_rrt_waypoints_do_not_authorize_colliding_smoothed_trajectory(self):
        fixture=self.fixture(smoothed_collision=True)
        with self.assertRaises(PlanningFailure) as caught:self.plan(fixture)
        self.assertEqual(caught.exception.code,'trajectory_failed')
        self.assertTrue(any(row[0]=='trajectory' for row in fixture.events))

    def test_trajectory_must_preserve_validated_rrt_and_ik_endpoints(self):
        fixture=self.fixture(changed_endpoint=True)
        with self.assertRaises(PlanningFailure) as caught:self.plan(fixture)
        self.assertEqual(caught.exception.code,'trajectory_failed')
        self.assertIn('endpoints',str(caught.exception))

    def test_self_mask_is_from_urdf_adjacency_not_current_collisions(self):
        fixture=self.fixture(self_pairs=[('base','tool')]);self.plan(fixture)
        pairs=adjacent_self_collision_pairs(fixture.urdf)
        self.assertIn(frozenset(('base','tool')),pairs)
        fixture=self.fixture(self_pairs=[('base','unknown_link')])
        with self.assertRaises(PlanningFailure) as caught:self.plan(fixture)
        self.assertEqual(caught.exception.code,'invalid_start')

    def test_mismatched_sdk_joint_order_is_rejected(self):
        fixture=self.fixture(wrong_order=True)
        with self.assertRaises(PlanningFailure) as caught:self.plan(fixture)
        self.assertEqual(caught.exception.code,'joint_mapping_failed')

    def test_native_world_registration_uses_robot_base_coordinates(self):
        fixture=self.fixture();fixture.backend.robot.get_world_pose=lambda:(np.array([1.,2.,0]),np.array([1,0,0,0]))
        fixture.snapshot.obstacles=(NS(kind='cuboid',path='/box',dimensions=(.1,.2,.3),
            position=(1.5,2.25,.4),orientation=(1,0,0,0)),)
        planner,result=self.plan(fixture)
        row=next(row for row in fixture.events if row[0]=='obstacle')
        np.testing.assert_allclose(row[2].translation,[.5,.25,.4])
        planner.clear()
        self.assertTrue(any(row[0]=='remove' for row in fixture.events))

    def test_cleanup_failure_still_attempts_every_registration_and_closes(self):
        fixture=self.fixture()
        fixture.snapshot.obstacles=tuple(NS(kind='cuboid',path='/box'+str(index),dimensions=(.1,.2,.3),
            position=(1.5+index,2.25,.4),orientation=(1,0,0,0)) for index in range(2))
        planner,result=self.plan(fixture)
        removals=[]
        def broken_remove(handle):
            removals.append(handle)
            if len(removals)==1:raise RuntimeError('one registration cleanup failed')
        planner._planner._world.remove_obstacle=broken_remove
        with self.assertRaisesRegex(RuntimeError,'cleanup failed'):planner.close()
        self.assertEqual(len(removals),2)
        self.assertIsNone(planner._planner)
        self.assertEqual(planner._handles,[])
        with self.assertRaises(PlanningFailure):_ = planner.joint_names

    def test_geometry_orientation_uses_sign_invariant_quaternions(self):
        np.testing.assert_allclose(quaternion_matrix([0,0,0,1]),quaternion_matrix([0,0,0,-1]))
        self.assertAlmostEqual(orientation_error(np.eye(3),quaternion_matrix([0,1,0,0])),np.pi)


if __name__=='__main__':unittest.main()
