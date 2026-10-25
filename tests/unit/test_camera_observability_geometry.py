"""Independent analytic limits for research-only camera visibility reduction."""
import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.camera_observability_geometry import (
    camera_record, catalog, GeometryPose, look_at_ros, projected, shape_depth, shape_surface,
    reduce_native_captures, visibility, wrist_camera)
from robot_skill_stack.world.spatial_mapping.robot_filter import MeshInstance, TriangleMesh


class CameraObservabilityGeometryTests(unittest.TestCase):
    def test_ros_camera_axis_and_projection_are_consistent(self):
        transform=look_at_ros([1.,-.5,.8],[.3,.1,.2])
        np.testing.assert_allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(transform[:3,:3]),1.)
        camera=camera_record(0,transform=transform)
        xyz,pixels,inside=projected(camera,np.array([[.3,.1,.2],[2.,-1.,1.4]]))
        np.testing.assert_allclose(pixels[0],[320.,240.],atol=1e-10)
        self.assertTrue(inside[0]);self.assertFalse(inside[1])

    def test_quadrature_retains_whole_surface_area(self):
        fixtures=[(dict(kind='box',center=[0,0,0],dimensions=[.1,.2,.3]),.22),
            (dict(kind='sphere',center=[0,0,0],radius=.2),4*math.pi*.2**2),
            (dict(kind='cylinder',center=[0,0,0],radius=.1,height=.4),2*math.pi*.1*(.1+.4))]
        for shape,expected in fixtures:
            points,normals,weights=shape_surface(shape,4)
            self.assertAlmostEqual(weights.sum(),expected)
            self.assertEqual(len(points),len(weights))
            np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1.,atol=1e-12)

    def test_analytic_front_depths_and_occluder_order(self):
        origin=np.array([0.,0.,0.]);directions=np.array([[0.,0.,1.],[.8,0.,1.]])
        shapes=[dict(kind='box',center=[0,0,2],dimensions=[1,1,1]),
                dict(kind='sphere',center=[0,0,2],radius=.5),
                dict(kind='cylinder',center=[0,0,2],radius=.5,height=1.)]
        with np.errstate(invalid='ignore'):
            for shape in shapes:
                distance=shape_depth(shape,origin,directions)
                self.assertAlmostEqual(distance[0],1.5)
                self.assertTrue(np.isinf(distance[1]))

    def test_frustum_membership_does_not_imply_mesh_visibility(self):
        camera=camera_record(0,[0,0,0],[0,0,1])
        mesh=TriangleMesh(np.array([[-.2,-.2,1.],[.2,-.2,1.],[0.,.2,1.]]),np.array([[0,1,2]]))
        part=MeshInstance(mesh,np.eye(4))
        points=np.array([[0.,0.,2.],[.5,.5,2.]])
        self.assertTrue(projected(camera,points)[2].all())
        visible,_=visibility(camera,points,[part])
        np.testing.assert_array_equal(visible,[False,True])

    def test_back_surfaces_are_not_counted_as_observable(self):
        camera=camera_record(0,[0,0,0],[0,0,1])
        points=np.array([[0.,0.,2.],[0.,0.,2.]])
        normals=np.array([[0.,0.,-1.],[0.,0.,1.]])
        visible,_=visibility(camera,points,normals=normals)
        np.testing.assert_array_equal(visible,[True,False])

    def test_endpoint_surface_is_not_an_occluder_in_front(self):
        camera=camera_record(0,[0,0,0],[0,0,1])
        target=np.array([[0.,0.,1.5]])
        box=dict(kind='box',center=[0.,0.,2.],dimensions=[1.,1.,1.])
        visible,_=visibility(camera,target,occluders=[box])
        self.assertTrue(visible[0])
        hidden,_=visibility(camera,np.array([[0.,0.,3.]]),occluders=[box])
        self.assertFalse(hidden[0])

    def test_height_variation_preserves_other_pose_and_optical_parameters(self):
        layouts=catalog()
        family=[row for row in layouts if row['layout_id'].startswith('B_front_height_')]
        self.assertEqual(len(family),3)
        for row in family[1:]:
            for base,camera in zip(family[0]['cameras'],row['cameras']):
                np.testing.assert_allclose(base['position_world_m'][:2],camera['position_world_m'][:2])
                np.testing.assert_allclose(base['intrinsics'],camera['intrinsics'])
                np.testing.assert_allclose(base['aim_world_m'],camera['aim_world_m'])
        self.assertEqual({row['family'] for row in layouts},set('ABCDEFGHIJK')|{'historical'})

    def test_wrist_mount_uses_meter_offsets_and_rigid_hand_roll(self):
        mesh=TriangleMesh(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),np.array([[0,1,2]]))
        transform=np.eye(4);transform[:3,:3]=100*np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        transform[:3,3]=[.4,.1,.5]
        pose=GeometryPose('test',(MeshInstance(mesh,transform),),'test')
        layout=next(row for row in catalog() if row['layout_id']=='J_wrist_offset_+0.05')
        camera=wrist_camera(layout,pose)
        np.testing.assert_allclose(camera['world_from_camera_ros'][:3,3],[.4,.15,.47],atol=1e-12)
        np.testing.assert_allclose(camera['world_from_camera_ros'][:3,:3],transform[:3,:3]/100,atol=1e-12)

    def test_native_union_keeps_overlap_stress_rows_separate_from_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);capture=output/'capture';capture.mkdir()
            geometry=dict(rows=[dict(layout_id='A_current',pose_id='home',objects=[
                dict(object_id='overlap',robot_overlap_evaluation=dict(potential_overlap=True)),
                dict(object_id='clear',robot_overlap_evaluation=dict(potential_overlap=False))])])
            (output/'geometry_screen.json').write_text(json.dumps(geometry))
            scores=[];arrays={}
            for index,(camera,obj,mask) in enumerate([
                    ('view0','overlap',[True,False]),('view0','clear',[True,False]),
                    ('view1','overlap',[False,True]),('view1','clear',[True,False])]):
                arrays[f'score_{index}_predicted_mask']=np.array(mask)
                arrays[f'score_{index}_native_mask']=np.array(mask)
                arrays[f'score_{index}_surface_weights']=np.array([1.,1.])
                scores.append(dict(camera_id=camera,object_id=obj,raw_projected_object_pixels=2,
                    native_surface_return_pixels=1,valid_depth_pixels=2,invalid_depth_pixels=0,
                    occluded_pixels=0,sampled_visibility_agreement=1.))
            np.savez(capture/'record.npz',**arrays)
            document=dict(layout_id='A_current',layout=dict(layout_id='A_current'),status='PASS',
                snapshots=[dict(id='home',tcp_position=[0,0,0],joints=[0,0])],
                records=[dict(posture='home',scenario_id='test',image_npz='record.npz',
                    objects=[dict(id='overlap'),dict(id='clear')],scores=scores,cameras=[])])
            (capture/'capture.json').write_text(json.dumps(document))
            with patch('scripts.camera_observability_geometry.native_figures'):
                report=reduce_native_captures([capture],output)
            summary=report['summaries'][0]
            self.assertEqual(summary['matched_object_posture_conditions'],2)
            self.assertAlmostEqual(summary['native_surface_visible_one_fraction'],.75)
            self.assertEqual(summary['primary_nonoverlap']['matched_object_posture_conditions'],1)
            self.assertAlmostEqual(summary['primary_nonoverlap']['native_surface_visible_one_fraction'],.5)
            self.assertAlmostEqual(summary['primary_nonoverlap']['native_surface_visible_two_fraction'],.5)


if __name__=='__main__':
    unittest.main()
