import copy
import itertools
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest

import numpy as np

from robot_skill_stack.world.spatial_mapping.occupancy import GridSpec, OccupancyMap
from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractor
from robot_skill_stack.world.spatial_mapping.runtime import MappingReadView
from scripts.evaluate_global_rgbd_v5 import (
    ARM_NAMES, BODY_NAMES, FINGER_NAMES, CookedRobotPart, PhysicalMotionEvaluator,
    PhysicalObstacle, densify_joint_route, physical_obstacles_from_static,
    point_exposure, robot_parts_from_capture,
    write_frozen_archive, load_frozen_archive, occupancy_quality,
    physically_free_proxy_audit, robot_motion_exposure,
    write_read_view_archive, summarize_case,
)


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

    def test_full_cooked_body_exposure_keeps_unknown_and_actual_phase(self):
        grid=OccupancyMap(GridSpec((-1.,-1.,-1.),(2.,25.,2.),1.),clock=lambda:1.)
        payload=self.payload()
        result=robot_motion_exposure(self.evaluator(),payload,grid.snapshot(now=1.),
            surface_points_per_part=8,max_poses_per_phase=8,actual_q=payload['actual_start_q'][None])
        self.assertEqual(set(result['phases']),{'route','trajectory','actual'})
        for phase in result['phases'].values():
            self.assertEqual(phase['surface']['fractions']['unknown'],1.)
            self.assertEqual(phase['surface']['fractions']['free'],0.)
            self.assertGreater(phase['surface']['samples'],phase['hand_origin']['samples'])
            self.assertGreater(phase['interior']['samples'],0)
        self.assertEqual(result['phases']['trajectory']['poses_sampled'],8)
        self.assertIn('not continuous',result['qualification'])


class OccupancyExposureTests(unittest.TestCase):
    def fixture(self):
        grid=OccupancyMap(GridSpec((0.,0.,0.),(.05,.02,.02),.01),clock=lambda:1.)
        grid.replace([[0,0,0],[1,0,0],[2,0,0]],unknown_indices=[[4,0,0]],timestamp=1.,now=1.)
        snapshot=grid.snapshot(now=1.)
        packet=ObstacleExtractor().extract(snapshot,now=1.,evidence_ttl=10.)
        return snapshot,packet

    def test_unknown_and_oob_are_not_relabelled_free(self):
        grid=OccupancyMap(GridSpec((0.,0.,0.),(.03,.01,.01),.01),clock=lambda:1.)
        grid.replace([],unknown_indices=[[1,0,0],[2,0,0]],timestamp=1.,now=1.)
        snapshot=grid.snapshot(now=1.)
        result=point_exposure(snapshot,[[.005,.005,.005],[.015,.005,.005],[.04,.005,.005]])
        self.assertEqual(result['counts'],dict(unknown=1,free=1,occupied=0,out_of_bounds=1))
        self.assertEqual(result['never_observed_unknown_samples'],1)
        self.assertIn('not continuous',result['qualification'])

    def test_archive_preserves_exact_fixed_time_arrays_and_original_packet(self):
        snapshot,packet=self.fixture()
        view=MappingReadView(snapshot,now=1.,evidence_ttl=10.,stale_after=20.,obstacles=packet)
        frozen=NS(view=view,metadata={'source_frames':({'camera_id':'one','frame_token':(1,2)},)})
        with TemporaryDirectory() as folder:
            record=write_frozen_archive(frozen,folder,10.)
            loaded,copied,meta=load_frozen_archive(record)
            np.testing.assert_array_equal(loaded.states,snapshot.states)
            self.assertEqual(copied.boxes[0].id,packet.boxes[0].id)
            with self.assertRaises(ValueError):loaded.states.setflags(write=True)
            record['canonical_arrays_sha256']='modified'
            with self.assertRaises(ValueError):load_frozen_archive(record)

    def test_quality_keeps_unknown_and_evaluation_attribution_separate(self):
        snapshot,packet=self.fixture()
        attributed=np.zeros(snapshot.spec.shape,bool);attributed[0,0,0]=True;attributed[4,0,0]=True
        result=occupancy_quality(snapshot,packet,evaluated_at=1.,evidence_ttl=10.,attribution_masks={'fixture':attributed})
        self.assertEqual(result['states']['unknown'],1)
        self.assertEqual(result['eligible_occupied_cells'],3)
        self.assertEqual(result['selected_cells'],3)
        self.assertFalse(any(result['invariants'].values()))
        self.assertEqual(result['fixtures']['fixture']['depth_attributed_cells'],2)
        self.assertEqual(result['fixtures']['fixture']['eligible_occupied_cells'],1)
        self.assertEqual(result['fixtures']['fixture']['selected_fraction_of_observed_occupied'],1.)

    def test_false_obstacle_requires_complete_physical_reference(self):
        snapshot,packet=self.fixture()
        reference=[PhysicalObstacle.convex('/KnownSolid',box((1.,1.,1.),.1))]
        result=physically_free_proxy_audit(packet,reference)
        self.assertEqual(result['status'],'UNMEASURED_INCOMPLETE_PHYSICAL_REFERENCE')
        result=physically_free_proxy_audit(packet,reference,complete_reference=True)
        self.assertEqual(result['whole_proxy_boxes_in_physical_free'],1)
        self.assertEqual(result['selected_cells_in_wholly_free_proxies'],3)
        touching=[PhysicalObstacle.convex('/KnownSolid',box((.015,.005,.005),.01))]
        result=physically_free_proxy_audit(packet,touching,complete_reference=True)
        self.assertEqual(result['whole_proxy_boxes_in_physical_free'],0)

    def test_rejected_source_archive_does_not_manufacture_a_world_or_fk(self):
        snapshot,_=self.fixture()
        view=MappingReadView(snapshot,now=1.,evidence_ttl=10.,stale_after=20.)
        with TemporaryDirectory() as folder:
            record=write_read_view_archive(view,folder,10.,metadata={'source_accepted_for_planning':False})
            self.assertIsNone(record['packet']);self.assertFalse(record['source_usable'])
            path=Path(folder)/'geometry_reference.json'
            path.write_text(json.dumps(dict(robot_parts=[],physical_obstacles=[],urdf_path=None)))
            summary=summarize_case(dict(frozen_archive=record,geometry=str(path)))
            self.assertEqual(summary['occupancy']['status'],'RAW_VIEW_ONLY_NO_SELECTED_PACKET')
            self.assertFalse(summary['occupancy']['source_usable'])
            self.assertEqual(summary['motion_exposure'],{})


if __name__=='__main__':unittest.main()
