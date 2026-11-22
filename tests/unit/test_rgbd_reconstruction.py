"""Qualified boxes, exact cuboid containment and worker-to-planner assembly."""
import io
import math
import pickle
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.world.spatial_mapping import reconstruction as rec
from robot_skill_stack.world.spatial_mapping.box_geometry import reference, visibility


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
                obstacles=NS(diagnostics={'support_plane_z':0.})))


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


class WorkerInferenceTests(unittest.TestCase):
    def test_qualification_excludes_robot_returns_and_refuses_mixed_or_unsupported_components(self):
        payload,scene = NS(frames=('A','B'),robot_snapshots={'A':object(),'B':object()}),NS()
        box = dict(component=0,reason='box',points=100,robot_envelope_points=0)
        robot = dict(component=1,reason='possible_robot_component',points=20,robot_envelope_points=20)
        geometry = model()
        with patch.object(reference,'reconstruct',return_value=({'components':[robot,box]},None,None)) as reconstruct, \
             patch.object(visibility,'qualify',return_value=dict(status='qualified',geometry=geometry)):
            product = rec.infer(payload,scene)
            reconstruct.assert_called_once_with(payload.frames,scene,payload.robot_snapshots)
            self.assertIsNone(product.reason)
            self.assertEqual(len(product.models),1)
            np.testing.assert_array_equal(product.models[0]['dimensions'],geometry['dimensions'])
        for components,reason in (([box,dict(robot,robot_envelope_points=19)],'robot_connected_component'),
                                  ([box,box],'unqualified_hidden_bound')):
            with patch.object(reference,'reconstruct',return_value=({'components':components},None,None)), \
                 patch.object(visibility,'qualify',side_effect=[dict(status='qualified',geometry=geometry),
                     dict(status='unqualified',reason='unqualified_hidden_bound')]):
                product = rec.infer(payload,scene)
                self.assertFalse(product.models)
                self.assertIn(reason,product.reason)


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
        with patch.object(reference,'reconstruct',return_value=({'components':[dict(component=0,reason='box')]},None,None)), \
             patch.object(visibility,'qualify',return_value=dict(status='qualified',geometry=geometry)):
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
