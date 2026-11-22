"""Independent physical-veto regression cases retained from source cdd2be9."""
import copy,itertools,json
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from tests.support.global_physical import (ARM_NAMES,BODY_NAMES,FINGER_NAMES,CookedRobotPart,
    PhysicalMotionEvaluator,PhysicalObstacle,densify_joint_route,physical_obstacles_from_static,robot_parts_from_capture)

def box(center=(0.,0.,.024), half=.004):
    return np.array(list(itertools.product(*[(-half, half)]*3)))+center


def urdf(path):
    links=[*(f'panda_link{i}' for i in range(8)), 'panda_hand','panda_leftfinger','panda_rightfinger']
    rows=['<robot name="synthetic_numeric_fixture">',*[f'<link name="{name}"/>' for name in links]]
    for i in range(1,8):
        kind='prismatic' if i<=2 else 'fixed'
        axis='1 0 0' if i==1 else '0 1 0'
        rows.append(f'<joint name="panda_joint{i}" type="{kind}"><parent link="panda_link{i-1}"/>'
            f'<child link="panda_link{i}"/><axis xyz="{axis}"/></joint>')
    rows.append('<joint name="hand" type="fixed"><parent link="panda_link7"/><child link="panda_hand"/></joint>')
    for index,side,direction in ((1,'left',1),(2,'right',-1)):
        rows.append(f'<joint name="panda_finger_joint{index}" type="prismatic"><parent link="panda_hand"/>'
            f'<child link="panda_{side}finger"/><axis xyz="0 {direction} 0"/></joint>')
    rows.append('</robot>');path.write_text(''.join(rows));return path


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'Native physical reference uses the existing Isaac SciPy installation')
class PhysicalVetoTests(unittest.TestCase):
    def setUp(self):
        self.folder=TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.path=urdf(Path(self.folder.name)/'fixture.urdf')
        self.parts=[]
        for index,name in enumerate(sorted(BODY_NAMES)):
            center=(0.,0.,.024) if name=='panda_hand' else (0.,10.+index,.024)
            self.parts.append(CookedRobotPart(name,box(center)))
        self.names=(*ARM_NAMES,*FINGER_NAMES)

    def payload(self, *, curved=False):
        x=np.linspace(0.,.2,81);q=np.zeros((len(x),7));q[:,0]=x
        if curved:q[:,1]=.05*np.sin(np.linspace(0.,np.pi,len(x)))
        start=np.array([0.]*7+[.003,.003])
        poses={name:np.eye(4) for name in BODY_NAMES}
        poses['panda_leftfinger'][1,3]=.003
        poses['panda_rightfinger'][1,3]=-.003
        route=np.zeros((2,7));route[-1,0]=.2
        return dict(joint_names=self.names, route_joint_names=ARM_NAMES,route=route,
            trajectory_q=q,trajectory_times=np.linspace(0.,.3,len(x)),finger_positions=[.003,.003],
            base_transform=np.eye(4),actual_start_q=start,actual_start_body_poses=poses)

    def evaluator(self, center=(.1,.2,.024), **kwargs):
        return PhysicalMotionEvaluator(self.parts,[PhysicalObstacle.convex('/Box',box(center,.012))],self.path,**kwargs)

    def test_positive_volume_collision_is_veto_not_rgbd_avoidance(self):
        payload=self.payload();before=payload['route'].copy()
        result=self.evaluator(center=(.1,0.,.024))(payload)
        self.assertFalse(result['execution_allowed'])
        self.assertEqual(result['status'],'VETO_PREDICTED_PHYSICAL_COLLISION')
        self.assertTrue(result['collision_witnesses'])
        self.assertGreater(result['collision_witnesses'][0]['intersection_ball']['radius_m'],0.)
        self.assertEqual(result['phases_evaluated']['trajectory'],81)
        self.assertFalse(result['planner_world_modified'])
        np.testing.assert_array_equal(payload['route'],before)

    def test_spline_is_checked_even_when_linear_route_is_clear(self):
        result=self.evaluator(center=(.1,.05,.024))(self.payload(curved=True))
        self.assertFalse(result['execution_allowed'])
        phases={row['phase'] for row in result['collision_witnesses']}
        self.assertEqual(phases,{'trajectory'})
        self.assertEqual(result['phases_evaluated']['route'],21)

    def test_clear_route_and_spline_receive_only_sampled_numeric_allowance(self):
        result=self.evaluator()(self.payload())
        self.assertTrue(result['execution_allowed'])
        self.assertEqual(result['status'],'ALLOW_SAMPLED_NUMERIC_CLEARANCE')
        self.assertTrue(result['samples_complete'])
        self.assertGreater(result['minimum_nonmount_clearance_lower_bound_m'],0.)
        self.assertIn('No continuous',result['qualification'])
        self.assertGreater(result['broadphase_separations'],0)

    def test_per_sample_nonmount_bound_excludes_qualified_base_plane(self):
        parts=[CookedRobotPart(part.body,box((0.,10.,.004))) if part.body=='panda_link0' else part for part in self.parts]
        evaluator=PhysicalMotionEvaluator(parts,[PhysicalObstacle.plane('/Ground',[0,0,0],[0,0,1])],self.path)
        result=evaluator(self.payload())
        self.assertTrue(result['execution_allowed'])
        self.assertEqual(result['minimum_external_clearance_lower_bound_m'],0.)
        self.assertGreater(result['minimum_nonmount_clearance_lower_bound_m'],.01)
        self.assertTrue(all(row['minimum_nonmount_clearance_lower_bound_m']>.01 for row in result['per_sample']))

    def test_actual_frame_disagreement_vetoes_before_geometry_queries(self):
        payload=self.payload();payload['actual_start_body_poses']['panda_hand'][0,3]=.001
        result=self.evaluator()(payload)
        self.assertEqual(result['status'],'VETO_UNVERIFIED_FK_ALIGNMENT')
        self.assertFalse(result['execution_allowed']);self.assertEqual(result['queries'],0)

    def test_coarse_or_endpoint_changed_spline_is_not_accepted(self):
        for mutation in ('coarse_time','coarse_q','different_goal'):
            payload=self.payload()
            if mutation=='coarse_time':payload['trajectory_times']*=2
            if mutation=='coarse_q':payload['trajectory_q'][30,0]+=.1
            if mutation=='different_goal':payload['trajectory_q'][-1,0]+=.0001
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.evaluator()(payload)

    def test_measured_start_and_finger_coordinates_are_required(self):
        for key in ('actual_start_q','finger_positions'):
            payload=self.payload()
            if key=='actual_start_q':payload[key][0]=.001
            else:payload[key][0]=.004
            with self.subTest(key=key),self.assertRaises(ValueError):self.evaluator()(payload)

    def test_joint_name_permutation_is_mapped_by_name(self):
        original=self.payload();result=self.evaluator()(original)
        changed=self.payload();changed['joint_names']=changed['joint_names'][::-1]
        changed['actual_start_q']=changed['actual_start_q'][::-1]
        changed['route_joint_names']=ARM_NAMES[::-1]
        changed['route']=changed['route'][:,::-1]
        changed['trajectory_q']=changed['trajectory_q'][:,::-1]
        self.assertEqual(self.evaluator()(changed)['status'],result['status'])

    def test_query_budget_is_fail_closed_without_fake_clearance(self):
        result=self.evaluator(max_queries=1)(self.payload())
        self.assertFalse(result['execution_allowed'])
        self.assertEqual(result['status'],'VETO_QUERY_BUDGET')
        self.assertFalse(result['samples_complete'])
        self.assertEqual(result['queries'],1)
        self.assertEqual(result['evaluated_samples'],0)
        self.assertFalse(result['per_sample'][0]['complete'])

    def test_early_collision_keeps_unchecked_phase_explicit(self):
        payload=self.payload();payload['stop_on_collision']=True
        result=self.evaluator(center=(.1,0.,.024))(payload)
        self.assertFalse(result['execution_allowed'])
        self.assertFalse(result['samples_complete'])
        self.assertEqual(result['phases_evaluated']['trajectory'],0)

    def test_geometry_owns_immutable_bytes(self):
        value=box();part=CookedRobotPart('panda_hand',value);value[:]=100.
        self.assertLess(part.vertices.max(),1.)
        with self.assertRaises(ValueError):part.vertices.setflags(write=True)
        obstacle=PhysicalObstacle.convex('/Box',value)
        with self.assertRaises(ValueError):obstacle.vertices.setflags(write=True)

    def test_empty_or_incomplete_physical_reference_is_not_silent_clear(self):
        with self.assertRaises(ValueError):PhysicalMotionEvaluator(self.parts,[],self.path)
        with self.assertRaises(ValueError):PhysicalMotionEvaluator(self.parts[:-1],[PhysicalObstacle.convex('/Box',box())],self.path)

    def test_route_density_and_budget_preserve_original_bound(self):
        route=np.zeros((2,7));route[-1,0]=.2
        sampled=densify_joint_route(route)
        self.assertLessEqual(np.max(abs(np.diff(sampled,axis=0))),.0100000001)
        with self.assertRaises(ValueError):densify_joint_route(route,max_joint_step_rad=.1)
        with self.assertRaises(ValueError):densify_joint_route(route,max_samples=3)
