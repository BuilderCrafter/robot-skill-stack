"""USD boundary doubles and analytic volume checks; no Kit or pxr imports."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.integrations.isaac.spatial_mapping.ground_truth import (
    prepare_ground_truth,
)
from robot_skill_stack.world.spatial_mapping import GridSpec
from robot_skill_stack.world.spatial_mapping.geometry import primitive_voxel_indices, rasterize_ground_truth


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
                 height=.1, axis='Z', purpose='default', collision=None,
                 active=True, defined=True, abstract=False, loaded=True, invisible=False, bounds=None):
        self.path, self.kind = path, kind
        self.matrix = np.eye(4) if matrix is None else matrix
        self.size, self.radius, self.height, self.axis = size, radius, height, axis
        self.purpose, self.collision, self.active = purpose, collision, active
        self.defined, self.abstract, self.loaded = defined, abstract, loaded
        self.invisible, self.bounds = invisible, bounds

    def GetPath(self): return self.path
    def GetTypeName(self): return self.kind
    def IsActive(self): return self.active
    def IsDefined(self): return self.defined
    def IsAbstract(self): return self.abstract
    def IsLoaded(self): return self.loaded
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
    proxy_predicate = object()
    usd = NS(TimeCode=NS(Default=lambda: None),
             PrimAllPrimsPredicate=object(),
             TraverseInstanceProxies=Mock(return_value=proxy_predicate))
    geom = NS(Gprim=NS(kind='gprim'), Cube=Cube, Sphere=Sphere, Cylinder=Cylinder,
              Imageable=lambda p: NS(ComputePurpose=lambda: p.purpose),
              Tokens=NS(default_='default', render='render', proxy='proxy', guide='guide'),
              XformCache=lambda t: NS(GetLocalToWorldTransform=lambda p: p.matrix.T),
              BBoxCache=Mock(return_value=NS(ComputeWorldBound=bound)),
              GetStageMetersPerUnit=lambda stage: stage.meters_per_unit)
    stage = NS(Traverse=Mock(return_value=prims), meters_per_unit=1., GetCompositionErrors=Mock(return_value=[]))
    return stage, (usd, geom, NS(CollisionAPI=CollisionAPI)), proxy_predicate


class PrimitiveRasterizationTests(unittest.TestCase):
    def setUp(self):
        self.spec = GridSpec(np.array([-.3]*3), np.array([.3]*3), .005)

    def occupied(self, indices, point):
        index = tuple(np.floor((np.asarray(point)-self.spec.bounds_min)/self.spec.voxel_size).astype(int))
        return index in {tuple(i) for i in indices}

    def test_rotated_nonuniform_cube_uses_full_affine_parent_transform(self):
        matrix = transform((.04, -.03, .08), (2, 1, .5), np.pi/4)
        # A parent shear is retained rather than replaced with extracted yaw/size.
        matrix[0, 2] += .2
        indices = primitive_voxel_indices(self.spec, 'cube', matrix, dimensions=[.1]*3)
        self.assertTrue(self.occupied(indices, world(matrix, [.04, .04, .02])))
        self.assertFalse(self.occupied(indices, matrix[:3, 3]+[.10, .10, 0]))

    def test_ellipsoid_volume_does_not_fill_rotated_world_aabb(self):
        matrix = transform(scale=(2, 1, .5), yaw=np.pi/4)
        indices = primitive_voxel_indices(self.spec, 'sphere', matrix, radius=.08)
        self.assertTrue(self.occupied(indices, world(matrix, [.06, 0, 0])))
        self.assertFalse(self.occupied(indices, [.122, .122, 0]))
        self.assertFalse(self.occupied(indices, [0, 0, .06]))

    def test_all_cylinder_axes_with_orientation_and_nonuniform_scaling(self):
        for axis in ('X', 'Y', 'Z'):
            matrix = transform((.02, -.03, .01), (1.4, .8, 1.1), .31, .47)
            axial = 'XYZ'.index(axis)
            radial = [i for i in range(3) if i != axial]
            inside, outside = np.zeros(3), np.zeros(3)
            inside[axial], inside[radial[0]] = .06, .01
            outside[radial] = .037
            with self.subTest(axis=axis):
                indices = primitive_voxel_indices(self.spec, 'cylinder', matrix,
                                                  radius=.04, height=.16, axis=axis)
                self.assertTrue(self.occupied(indices, world(matrix, inside)))
                self.assertFalse(self.occupied(indices, world(matrix, outside)))

    def test_cell_volume_overlap_includes_small_object_without_inside_cell_center(self):
        spec = GridSpec(np.zeros(3), np.array([.025, .01, .01]), .01)
        indices = primitive_voxel_indices(spec, 'sphere', transform((.0102, .005, .005)), radius=.0015)
        self.assertEqual({tuple(i) for i in indices}, {(0, 0, 0), (1, 0, 0)})
        edge = primitive_voxel_indices(spec, 'sphere', transform((.024, .005, .005)), radius=.0005)
        self.assertEqual(edge.tolist(), [[2, 0, 0]])

    def test_invalid_geometry_and_candidate_budget_fail_explicitly(self):
        for kwargs in ({'radius': 0}, {'radius': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                primitive_voxel_indices(self.spec, 'sphere', np.eye(4), **kwargs)
        with self.assertRaises(np.linalg.LinAlgError):
            primitive_voxel_indices(self.spec, 'cube', np.zeros((4, 4))+np.diag([0, 1, 1, 1]),
                                    dimensions=[.1]*3)
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            primitive_voxel_indices(self.spec, 'sphere', np.eye(4), radius=.1, max_candidates=10)


class GroundTruthPreparationTests(unittest.TestCase):
    def prepare(self, prims, *, excluded_paths=('/Franka',), timestamp=1.):
        stage, modules, marker = usd_fixture(prims)
        with patch('robot_skill_stack.integrations.isaac.spatial_mapping.ground_truth._usd_modules',
                   return_value=modules):
            prepared = prepare_ground_truth(stage, excluded_paths=excluded_paths, timestamp=timestamp)
        stage.Traverse.assert_called_once_with(marker)
        return prepared

    def test_usd_exclusions_visibility_and_unsupported_geometry(self):
        prepared = self.prepare([
            Prim('/Box', matrix=transform((.1, .1, .1))),
            Prim('/Franka/Visual', matrix=transform((.2, .2, .2))),
            Prim('/Invisible', invisible=True),
            Prim('/Guide', purpose='guide'),
            Prim('/Mesh', kind='mesh', bounds=([0., 0., 0.], [.1, .1, .1])),
        ])
        self.assertEqual(prepared.timestamp, 1.)
        self.assertEqual(len(prepared.items), 3)
        self.assertEqual(prepared.diagnostics['excluded_prims'], 1)
        self.assertEqual(prepared.diagnostics['guide_prims'], 1)
        self.assertEqual(prepared.diagnostics['unsupported_aabb_prims'], 1)

    def test_prepared_geometry_rasterizes_without_accessing_usd(self):
        packet = self.prepare([Prim('/Sphere', 'sphere', matrix=transform((.1, .1, .1)),
                                    radius=.05)])
        spec = GridSpec((0., 0., 0.), (.3, .3, .3), .02)
        with patch('robot_skill_stack.integrations.isaac.spatial_mapping.ground_truth._usd_modules',
                   side_effect=AssertionError('worker accessed USD')):
            indices, diagnostics = rasterize_ground_truth(spec, packet, 8_000_000)
        self.assertGreater(len(indices), 0)
        self.assertEqual(diagnostics['supported_primitives'], 1)

    def test_invalid_units_unloaded_payloads_and_composition_fail(self):
        for units in (0., -1., np.nan, np.inf):
            stage, modules, _ = usd_fixture([])
            stage.meters_per_unit = units
            with self.subTest(units=units), patch(
                    'robot_skill_stack.integrations.isaac.spatial_mapping.ground_truth._usd_modules',
                    return_value=modules), self.assertRaisesRegex(ValueError, 'metersPerUnit'):
                prepare_ground_truth(stage)
        self.prepare([Prim('/Franka/Unloaded', loaded=False)])
        with self.assertRaisesRegex(RuntimeError, 'Unloaded USD geometry payload'):
            self.prepare([Prim('/Environment', loaded=False)])
        stage, modules, _ = usd_fixture([])
        stage.GetCompositionErrors.return_value = ['missing geometry']
        with patch('robot_skill_stack.integrations.isaac.spatial_mapping.ground_truth._usd_modules',
                   return_value=modules), self.assertRaisesRegex(RuntimeError, 'Unresolved USD scene composition'):
            prepare_ground_truth(stage)


if __name__ == '__main__':
    unittest.main()
