"""USD boundary doubles and analytic volume checks; no Kit or pxr imports."""
from types import SimpleNamespace as NS
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.integrations.isaac.spatial_mapping.ground_truth import (
    IsaacGroundTruthMapper, primitive_voxel_indices,
)
from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap, OccupancyState


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


class GroundTruthMappingTests(unittest.TestCase):
    def setUp(self):
        self.map = OccupancyMap(GridSpec(np.array([-.5]*3), np.array([.5]*3), .025), clock=lambda: 1.)

    def update(self, mapper, prims, timestamp=1.):
        stage, modules, marker = usd_fixture(prims)
        with patch('robot_skill_stack.integrations.isaac.spatial_mapping.ground_truth._usd_modules',
                   return_value=modules):
            result = mapper.update(stage, timestamp=timestamp, now=timestamp)
        stage.Traverse.assert_called_once_with(marker)
        modules[0].TraverseInstanceProxies.assert_called_once_with(modules[0].PrimAllPrimsPredicate)
        modules[1].BBoxCache.assert_called_once_with(None, ['default', 'render', 'proxy', 'guide'],
                                                   useExtentsHint=False, ignoreVisibility=True)
        return result

    def state(self, point):
        return self.map.query(point, now=1.)

    def test_whole_scene_includes_invisible_geometry_and_collision_guides_excludes_robot_helpers(self):
        mapper = IsaacGroundTruthMapper(self.map)
        prims = [Prim('/UnclassifiedObstacle', matrix=transform((-.2, 0, .1))),
                 Prim('/InvisibleCollider', matrix=transform((-.3, -.3, .1)), invisible=True),
                 Prim('/Franka/Link/Visual', matrix=transform((.3, .3, .1))),
                 Prim('/FrankaExtra', matrix=transform((.3, -.3, .1))),
                 Prim('/Guide', matrix=transform((-.3, .3, .1)), purpose='guide'),
                 Prim('/CollisionGuide', matrix=transform((0, .3, .1)), purpose='guide', collision=True),
                 Prim('/Inactive', matrix=transform((0, -.3, .1)), active=False),
                 Prim('/Undefined', defined=False), Prim('/AbstractClass', abstract=True),
                 Prim('/Camera', kind='camera'), Prim('/Light', kind='light')]
        self.assertTrue(self.update(mapper, prims))
        for point in ([-.2, 0, .1], [-.3, -.3, .1], [.3, -.3, .1], [0, .3, .1]):
            self.assertEqual(self.state(point).state, OccupancyState.OCCUPIED)
        for point in ([.3, .3, .1], [-.3, .3, .1], [0, -.3, .1], [0, 0, .4]):
            self.assertEqual(self.state(point).state, OccupancyState.FREE)
        self.assertEqual(mapper.diagnostics['supported_primitives'], 4)
        self.assertEqual(mapper.diagnostics['guide_prims'], 1)
        self.assertEqual(mapper.diagnostics['excluded_prims'], 1)

    def test_supported_usd_attributes_and_row_vector_transforms(self):
        mapper = IsaacGroundTruthMapper(self.map, excluded_paths=())
        prims = [Prim('/Instances/ChildCube', matrix=transform((-.2, 0, .1), (1, 2, .5), .6)),
                 Prim('/Sphere', 'sphere', matrix=transform((.2, 0, .1), (2, 1, .5)), radius=.04),
                 Prim('/Cylinder', 'cylinder', matrix=transform((0, .25, .1), yaw=.3, pitch=.5),
                      radius=.025, height=.2, axis='X')]
        self.update(mapper, prims)
        for prim in prims:
            self.assertEqual(self.state(prim.matrix[:3, 3]).state, OccupancyState.OCCUPIED)
        self.assertEqual(self.state([0, 0, .1]).state, OccupancyState.FREE)
        self.assertEqual(mapper.diagnostics['supported_primitives'], 3)

    def test_nonmeter_usd_units_match_meter_primitive_and_unsupported_bounds(self):
        prims = [Prim('/Cube', matrix=transform((-.2, 0, .1), (1, 2, .5), .6)),
                 Prim('/Sphere', 'sphere', matrix=transform((.2, 0, .1)), radius=.04),
                 Prim('/Cylinder', 'cylinder', matrix=transform((0, .25, .1)), radius=.03, height=.12),
                 Prim('/Mesh', 'mesh', bounds=([-.3, -.3, .1], [-.2, -.2, .2]))]
        mapper = IsaacGroundTruthMapper(self.map)
        self.update(mapper, prims)
        expected = self.map.snapshot(now=1.).states
        centimeter_prims = deepcopy(prims)
        for prim in centimeter_prims:
            prim.matrix[:3, 3] *= 100
            prim.size *= 100
            prim.radius *= 100
            prim.height *= 100
            if prim.bounds is not None:
                prim.bounds = np.asarray(prim.bounds)*100
        stage, modules, _ = usd_fixture(centimeter_prims)
        stage.meters_per_unit = .01
        with patch('robot_skill_stack.integrations.isaac.spatial_mapping.ground_truth._usd_modules',
                   return_value=modules):
            self.assertTrue(mapper.update(stage, timestamp=2., now=2.))
        np.testing.assert_array_equal(expected, self.map.snapshot(now=2.).states)
        self.assertEqual(mapper.diagnostics['meters_per_unit'], .01)

    def test_invalid_stage_units_fail_without_asserting_free(self):
        mapper = IsaacGroundTruthMapper(self.map)
        for value in (0, -1, np.nan, np.inf):
            stage, modules, _ = usd_fixture([])
            stage.meters_per_unit = value
            with self.subTest(value=value), patch(
                    'robot_skill_stack.integrations.isaac.spatial_mapping.ground_truth._usd_modules',
                    return_value=modules), self.assertRaisesRegex(ValueError, 'metersPerUnit'):
                mapper.update(stage, timestamp=1., now=1.)
            self.assertIsNone(self.map.snapshot(now=1.).timestamp)

    def test_unloaded_payload_fails_but_explicitly_excluded_unloaded_robot_is_allowed(self):
        mapper = IsaacGroundTruthMapper(self.map)
        self.update(mapper, [Prim('/Franka/Unloaded', kind='xform', loaded=False), Prim('/Box')])
        before = self.map.snapshot(now=1.)
        with self.assertRaisesRegex(RuntimeError, 'Unloaded USD geometry payload: /Environment'):
            self.update(mapper, [Prim('/Environment', kind='xform', loaded=False)], timestamp=2.)
        after = self.map.snapshot(now=2.)
        self.assertEqual(after.timestamp, before.timestamp)
        np.testing.assert_array_equal(after.states, before.states)

    def test_reported_scene_composition_errors_fail_closed(self):
        mapper = IsaacGroundTruthMapper(self.map)
        stage, modules, _ = usd_fixture([])
        stage.GetCompositionErrors.return_value = ['missing referenced geometry']
        with patch('robot_skill_stack.integrations.isaac.spatial_mapping.ground_truth._usd_modules',
                   return_value=modules), self.assertRaisesRegex(RuntimeError, 'Unresolved USD scene composition'):
            mapper.update(stage, timestamp=1., now=1.)
        self.assertIsNone(self.map.snapshot(now=1.).timestamp)

    def test_moving_removing_and_empty_scene_rebuilds_authoritative_free_space(self):
        mapper = IsaacGroundTruthMapper(self.map)
        self.update(mapper, [Prim('/Box', matrix=transform((-.2, 0, .1)))])
        self.assertEqual(self.state([-.2, 0, .1]).state, OccupancyState.OCCUPIED)
        self.update(mapper, [Prim('/Box', matrix=transform((.2, 0, .1)))], timestamp=2.)
        self.assertEqual(self.state([-.2, 0, .1]).state, OccupancyState.FREE)
        self.assertEqual(self.state([.2, 0, .1]).state, OccupancyState.OCCUPIED)
        self.update(mapper, [], timestamp=3.)
        self.assertEqual(self.map.snapshot(now=3.).counts()['occupied'], 0)
        self.assertEqual(self.state([.2, 0, .1]).state, OccupancyState.FREE)

    def test_unsupported_mesh_and_zero_thickness_points_are_explicit_aabb_approximations(self):
        mapper = IsaacGroundTruthMapper(self.map)
        self.update(mapper, [Prim('/Mesh', 'mesh', bounds=([.1, .1, .1], [.2, .2, .2])),
                             Prim('/Points', 'points', bounds=([-.2, -.2, .1], [-.1, -.1, .1]))])
        self.assertEqual(self.state([.15, .15, .15]).state, OccupancyState.OCCUPIED)
        self.assertEqual(self.state([-.15, -.15, .1]).state, OccupancyState.OCCUPIED)
        self.assertEqual(self.state([0, 0, .1]).state, OccupancyState.FREE)
        self.assertEqual(mapper.diagnostics['unsupported_aabb_prims'], 2)
        self.assertEqual({a['path'] for a in mapper.diagnostics['approximations']}, {'/Mesh', '/Points'})

    def test_unbounded_or_budget_failure_preserves_last_map_and_reports_unhealthy(self):
        mapper = IsaacGroundTruthMapper(self.map)
        self.update(mapper, [Prim('/Box', matrix=transform((-.2, 0, .1)))])
        before = self.map.snapshot(now=1.)
        with self.assertRaisesRegex(ValueError, 'finite world bounds'):
            self.update(mapper, [Prim('/NewBox', matrix=transform((.2, 0, .1))),
                                 Prim('/Unknown', 'mesh', bounds=([-np.inf]*3, [np.inf]*3))], timestamp=2.)
        after = self.map.snapshot(now=2.)
        np.testing.assert_array_equal(before.states, after.states)
        self.assertEqual((after.timestamp, after.revision), (before.timestamp, before.revision))
        self.assertEqual(after.health['status'], 'error')
        self.assertFalse(after.health['fresh'])
        limited = IsaacGroundTruthMapper(self.map, max_candidate_voxels=500)
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            self.update(limited, [Prim(f'/Box{i}') for i in range(3)], timestamp=3.)
        self.assertFalse(limited.diagnostics['published'])
        self.assertEqual(self.map.snapshot(now=3.).revision, before.revision)

    def test_repeated_timestamp_and_closed_source_never_publish_new_revision(self):
        mapper = IsaacGroundTruthMapper(self.map)
        self.update(mapper, [Prim('/Box')])
        before = self.map.snapshot(now=1.)
        self.assertFalse(self.update(mapper, [Prim('/Moved', matrix=transform((.3, 0, 0)))]))
        after = self.map.snapshot(now=1.)
        self.assertEqual(after.revision, before.revision)
        np.testing.assert_array_equal(after.states, before.states)
        mapper.close()
        mapper.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            mapper.update(NS())


if __name__ == '__main__':
    unittest.main()
