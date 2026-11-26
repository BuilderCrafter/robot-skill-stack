"""Prepared USD geometry and conservative affine voxel overlap without Kit."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.integrations.isaac.spatial_mapping.ground_truth import prepare_ground_truth
from robot_skill_stack.world.spatial_mapping import GridSpec
from robot_skill_stack.world.spatial_mapping.geometry import rasterize_ground_truth


def transform(center=(0, 0, 0), scale=(1, 1, 1), yaw=0., pitch=0.):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    rotation = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.]]) @ np.array(
        [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation @ np.diag(scale), center
    return matrix


def world(matrix, local):
    return matrix[:3, :3] @ local+matrix[:3, 3]


class Prim:
    def __init__(self, path, kind='cube', *, matrix=None, size=.1, radius=.05,
                 height=.1, axis='Z', purpose='default', collision=None, bounds=None):
        self.path, self.kind = path, kind
        self.matrix = np.eye(4) if matrix is None else matrix
        self.size, self.radius, self.height, self.axis = size, radius, height, axis
        self.purpose, self.collision = purpose, collision
        self.bounds = bounds

    def GetPath(self): return self.path
    def IsActive(self): return True
    def IsDefined(self): return True
    def IsAbstract(self): return False
    def IsLoaded(self): return True
    def HasAPI(self, schema): return self.collision is not None
    def IsA(self, schema):
        return (self.kind not in ('camera', 'light', 'xform') if schema.kind == 'gprim'
                else self.kind == schema.kind)


def usd_fixture(prims):
    class Schema:
        def __init__(self, prim): self.prim = prim
        def attr(self, key): return NS(Get=lambda *args: getattr(self.prim, key))
    class Cube(Schema):
        kind = 'cube'
        def GetSizeAttr(self): return self.attr('size')
    class Sphere(Schema):
        kind = 'sphere'
        def GetRadiusAttr(self): return self.attr('radius')
    class Cylinder(Sphere):
        kind = 'cylinder'
        def GetHeightAttr(self): return self.attr('height')
        def GetAxisAttr(self): return self.attr('axis')
    class CollisionAPI(Schema):
        def GetCollisionEnabledAttr(self): return self.attr('collision')
    def bound(prim):
        lower, upper = prim.bounds
        return NS(ComputeAlignedRange=lambda: NS(GetMin=lambda: lower, GetMax=lambda: upper))
    usd = NS(TimeCode=NS(Default=lambda: None),
             PrimAllPrimsPredicate=None, TraverseInstanceProxies=lambda _:None)
    geom = NS(Gprim=NS(kind='gprim'), Cube=Cube, Sphere=Sphere, Cylinder=Cylinder,
              Imageable=lambda p: NS(ComputePurpose=lambda: p.purpose),
              Tokens=NS(default_='default', render='render', proxy='proxy', guide='guide'),
              XformCache=lambda t: NS(GetLocalToWorldTransform=lambda p: p.matrix.T),
              BBoxCache=Mock(return_value=NS(ComputeWorldBound=bound)),
              GetStageMetersPerUnit=lambda stage: stage.meters_per_unit)
    stage = NS(Traverse=Mock(return_value=prims), meters_per_unit=1., GetCompositionErrors=Mock(return_value=[]))
    return stage, NS(Usd=usd, UsdGeom=geom, UsdPhysics=NS(CollisionAPI=CollisionAPI))



class GroundTruthGeometryTests(unittest.TestCase):
    def setUp(self):
        self.spec = GridSpec((-.3,-.3,-.3),(.3,.3,.3),.005)

    def prepare(self, prims):
        stage, pxr = usd_fixture(prims)
        with patch.dict('sys.modules', {'pxr': pxr}):
            return prepare_ground_truth(stage, timestamp=100.)

    def raster(self, prims):
        return rasterize_ground_truth(self.spec,self.prepare(prims))

    def occupied(self, indices, point):
        index = tuple(self.spec.world_to_indices(point)[0])
        return index in {tuple(cell) for cell in indices}

    def test_rotated_nonuniform_cube_retains_shear_and_actual_shape(self):
        matrix = transform((.04,-.03,.08),(2,1,.5),np.pi/4)
        matrix[0,2] += .2
        indices = self.raster([Prim('/Box',matrix=matrix)])
        self.assertTrue(self.occupied(indices,world(matrix,[.04,.04,.02])))
        self.assertFalse(self.occupied(indices,matrix[:3,3]+[.10,.10,0.]))

    def test_ellipsoid_does_not_fill_its_world_aabb(self):
        matrix = transform(scale=(2,1,.5),yaw=np.pi/4)
        indices = self.raster([Prim('/Sphere','sphere',matrix=matrix,radius=.08)])
        self.assertTrue(self.occupied(indices,world(matrix,[.06,0.,0.])))
        self.assertFalse(self.occupied(indices,[.122,.122,0.]))
        self.assertFalse(self.occupied(indices,[0.,0.,.06]))

    def test_lying_cylinder_preserves_affine_radial_and_axial_extent(self):
        matrix = transform((.02,-.03,.01),(1.4,.8,1.1),.31,.47)
        indices = self.raster([Prim('/Cylinder','cylinder',matrix=matrix,radius=.04,height=.16,axis='X')])
        self.assertTrue(self.occupied(indices,world(matrix,[.06,.01,0.])))
        self.assertFalse(self.occupied(indices,world(matrix,[0.,.037,.037])))

    def test_small_surface_crosses_cells_and_partial_boundary_cell_is_retained(self):
        self.spec = GridSpec((0.,0.,0.),(.025,.01,.01))
        indices = self.raster([Prim('/Small','sphere',matrix=transform((.0102,.005,.005)),radius=.0015)])
        self.assertEqual({tuple(cell) for cell in indices},{(0,0,0),(1,0,0)})
        edge = self.raster([Prim('/Edge','sphere',matrix=transform((.024,.005,.005)),radius=.0005)])
        self.assertEqual(edge.tolist(),[[2,0,0]])

    def test_candidate_budget_is_shared_and_unsupported_aabbs_form_a_union(self):
        self.spec = GridSpec((0.,0.,0.),(.03,.03,.03))
        prepared = self.prepare([Prim(path,'mesh',bounds=([.011]*3,[.019]*3)) for path in ('/A','/B')])
        with self.assertRaises(RuntimeError):
            rasterize_ground_truth(self.spec,prepared,max_candidate_voxels=1)
        np.testing.assert_array_equal(rasterize_ground_truth(self.spec,prepared,max_candidate_voxels=2),[[1,1,1]])

    def test_prepared_owned_geometry_runs_in_worker_without_usd_or_excluded_geometry(self):
        self.spec = GridSpec((0.,0.,0.),(.3,.3,.3),.02)
        box = Prim('/Box',matrix=transform((.1,.1,.1),yaw=.3))
        prepared = self.prepare([box,Prim('/Franka/Box',matrix=transform((.25,.25,.25))),
            Prim('/Guide',matrix=transform((.25,.25,.05)),purpose='guide')])
        expected = rasterize_ground_truth(self.spec,prepared)
        box.matrix[:3,3] = 9
        with patch.dict('sys.modules', {'pxr': None, 'omni': None, 'isaacsim': None}):
            with ThreadPoolExecutor(max_workers=1) as worker:
                actual = worker.submit(rasterize_ground_truth,self.spec,prepared).result(timeout=1.)
        np.testing.assert_array_equal(actual,expected)
        self.assertTrue(self.occupied(actual,[.1,.1,.1]))
        self.assertFalse(self.occupied(actual,[.25,.25,.25]))
        self.assertFalse(self.occupied(actual,[.25,.25,.05]))
        self.assertEqual(prepared.timestamp,100.)
        with self.assertRaises(ValueError):
            prepared.items[0].inverse.setflags(write=True)


if __name__ == '__main__':
    unittest.main()
