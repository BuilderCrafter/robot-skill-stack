"""Qualified boxes, exact cuboid containment and worker-to-planner assembly."""
import io
import math
import pickle
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.world.spatial_mapping import reconstruction as rec
from robot_skill_stack.world.spatial_mapping import box_geometry
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame


def model(center=(0.,0.,.2), dimensions=(.14,.06,.6), yaw=0.):
    c,s = math.cos(yaw),math.sin(yaw)
    return dict(center=np.asarray(center), dimensions=np.asarray(dimensions),
                rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]]), yaw_rad=yaw)


def fixture(models=None, reason=None):
    boxes = tuple(NS(id=name,center=center,dimensions=dimensions) for name,center,dimensions in (
        ('floor',(0.,0.,-.005),(.01,.01,.01)),
        ('inside',(0.,0.,.2),(.01,.01,.01)),
        ('partial',(.04,.04,.2),(.04,.04,.04)),
        ('outside',(.5,0.,.2),(.01,.01,.01))))
    return NS(generation=2,revision=3,boxes=boxes,spec=NS(voxel_size=.01),
        view=NS(inferred=rec.InferredGeometry((model(yaw=math.pi/4),) if models is None else models,reason),
                obstacles=NS(support_plane_z=0.)))


def observations():
    """Two calibrated views of a floating box against a farther depth surface."""
    box = model(dimensions=(.12,.08,.16),yaw=.2)
    y,x = np.indices((160,160))
    rays = np.c_[(x.ravel()+.5-80)/240,(y.ravel()+.5-80)/240,np.ones(x.size)]
    k = np.array([[240.,0.,80.],[0.,240.,80.],[0.,0.,1.]])
    frames = []
    for i,position in enumerate(((.45,-.4,.55),(-.45,.4,.55))):
        forward = box['center']-position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward,[0.,0.,1.])
        right /= np.linalg.norm(right)
        transform = np.eye(4)
        transform[:3,:3] = np.column_stack((right,np.cross(forward,right),forward))
        transform[:3,3] = position
        origin = (position-box['center'])@box['rotation']
        directions = rays@transform[:3,:3].T@box['rotation']
        with np.errstate(divide='ignore',invalid='ignore'):
            first = (-box['dimensions']/2-origin)/directions
            second = (box['dimensions']/2-origin)/directions
        enter,leave = np.minimum(first,second).max(1),np.maximum(first,second).min(1)
        depth = np.where((leave>enter)&(enter>0),enter,2.).reshape(160,160)
        frames.append(DepthFrame(str(i),(i,.1),depth,k,transform,10.))
    payload = NS(frames=tuple(frames),robot_snapshots={f.camera_id:NS(parts=()) for f in frames})
    scene = NS(workspace_min=(-.2,-.2,.02),workspace_max=(.2,.2,.4),support_plane_z=0.)
    return payload,scene


class AssemblyTests(unittest.TestCase):
    def test_yaw_geometry_and_full_corner_containment_preserve_floor_and_partial_cells(self):
        frozen = fixture()
        descriptors = rec.combine(frozen)
        inferred = descriptors[0]
        self.assertEqual(inferred['path'],'/__rgbd_inferred__/2/3/0')
        np.testing.assert_allclose(inferred['position'],(0.,0.,.2))
        np.testing.assert_allclose(inferred['dimensions'],(.14,.06,.6))
        np.testing.assert_allclose(inferred['orientation'],(math.cos(math.pi/8),0,0,math.sin(math.pi/8)))
        self.assertEqual(inferred['source_kind'],'rgbd_inferred_box')
        self.assertEqual([d['path'] for d in descriptors[1:]],
                         ['/__rgbd_occupied__/'+name for name in ('floor','partial','outside')])
        for descriptor,box in zip(descriptors[1:],(frozen.boxes[i] for i in (0,2,3))):
            self.assertEqual((descriptor['position'],descriptor['dimensions']),(box.center,box.dimensions))

    def test_absent_or_unusable_reconstruction_refuses_but_empty_retains_all_cuboids(self):
        frozen = fixture(reason='unsupported_hidden_bound')
        with self.assertRaisesRegex(ValueError,'unsupported_hidden_bound'):
            rec.combine(frozen)
        frozen.view.inferred = None
        with self.assertRaisesRegex(ValueError,'No inferred geometry'):
            rec.combine(frozen)
        frozen = fixture(models=())
        self.assertEqual([d['path'] for d in rec.combine(frozen)],
                         ['/__rgbd_occupied__/'+box.id for box in frozen.boxes])

    def test_combined_camera_proxy_budget_is_64(self):
        frozen = fixture()
        frozen.boxes = tuple(NS(id=str(i),center=(1.+i,0.,.2),dimensions=(.01,.01,.01)) for i in range(63))
        self.assertEqual(len(rec.combine(frozen)),64)
        frozen.boxes += (NS(id='extra',center=(100.,0.,.2),dimensions=(.01,.01,.01)),)
        with self.assertRaisesRegex(ValueError,'64'):
            rec.combine(frozen)


class NumericalTests(unittest.TestCase):
    def test_calibrated_depth_reconstructs_measured_box_without_support_height_completion(self):
        payload,scene = observations()
        product = rec.infer(payload,scene)
        self.assertIsNone(product.reason)
        box, = product.models
        np.testing.assert_allclose(box['center'],(0.,0.,.2),atol=.005)
        np.testing.assert_allclose(box['dimensions'],(.12,.08,.16),atol=.012)
        self.assertAlmostEqual(box['yaw_rad'],.2,delta=.002)
        self.assertGreater(box['center'][2]-box['dimensions'][2]/2,.1)

    def test_missing_free_boundaries_contradictory_depth_and_robot_geometry(self):
        payload,scene = observations()
        missing = tuple(replace(f,depth=np.where(f.depth<2.,f.depth,np.nan)) for f in payload.frames)
        free = (replace(payload.frames[0],depth=np.full((160,160),2.)),payload.frames[1])
        for frames,reason in ((missing,'unqualified_hidden_bound'),(free,'contradictory_current_free_rays')):
            product = rec.infer(NS(frames=frames,robot_snapshots=payload.robot_snapshots),scene)
            self.assertFalse(product.models)
            self.assertEqual(product.reason,reason)
        for bounds,reason in ((((-1.,-1.,-1.),(1.,1.,1.)),None),
                              (((.04,-.08,.1),(.1,.08,.3)),'robot_connected_component')):
            part = NS(mesh=NS(vertices=np.array(bounds)),world_from_local=np.eye(4))
            robot = {f.camera_id:NS(parts=(part,)) for f in payload.frames}
            product = rec.infer(NS(frames=payload.frames,robot_snapshots=robot),scene)
            self.assertFalse(product.models)
            self.assertEqual(product.reason,reason)


class PublicationTests(unittest.TestCase):
    def test_mapping_worker_product_reaches_opt_in_adapter_with_original_source_and_support(self):
        from tests.unit.test_frozen_rgbd_world import FrozenRgbdWorldTests
        from tests.unit.test_global_frozen_rgbd_v5 import support
        from robot_skill_stack.runtime.process_worker import _TransportPickler
        from robot_skill_stack.integrations.isaac.manipulation.motion.frozen_rgbd import capture_frozen_rgbd_world
        fixture = FrozenRgbdWorldTests()
        source = fixture.prepared(support=0.)
        self.addCleanup(fixture.doCleanups)
        source._processor._reconstruction_scene = NS()
        geometry = model(center=(0,0,1),dimensions=(.5,.5,.5))
        with patch.object(box_geometry,'reconstruct',return_value=((geometry,),None)):
            result = fixture.publish(source,3,100.15)
        stream = io.BytesIO()
        _TransportPickler(stream,protocol=5).dump(result)
        np.testing.assert_array_equal(pickle.loads(stream.getvalue()).inferred.models[0]['center'],geometry['center'])
        capture = lambda **options: capture_frozen_rgbd_world(None,lambda:source,
            support_capture=lambda *a,**k:support(),**options)
        occupied,old = capture()
        inferred,new = capture(representation='reconstructed')
        self.assertIs(new.view.inferred,result.inferred)
        self.assertIs(old.view._snapshot,new.view._snapshot)
        self.assertEqual(new.timestamp,100.15)
        self.assertEqual(new.source_frames[0]['token'],(3,3*.1))
        self.assertEqual(inferred.obstacles[0],occupied.obstacles[0])
        self.assertTrue(all(o.source_kind!='rgbd_inferred_box' for o in occupied.obstacles))
        box, = (o for o in inferred.obstacles if o.source_kind=='rgbd_inferred_box')
        np.testing.assert_array_equal(box.position,geometry['center'])
        np.testing.assert_array_equal(box.dimensions,geometry['dimensions'])


if __name__ == '__main__':
    unittest.main()
