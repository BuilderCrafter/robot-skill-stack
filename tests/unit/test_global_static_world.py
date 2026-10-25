import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.common.rotations import quat_matrix
from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import (
    PlaneDomain, StaticObstacle, StaticWorldError, StaticWorldSnapshot,
    box_from_convex_hull_points, capture_static_world, obstacle_from_primitive,
)


MODULE = 'robot_skill_stack.integrations.isaac.manipulation.motion.static_world'


class PrimitiveTests(unittest.TestCase):
    def test_rotated_scaled_box_retains_pose_dimensions_and_detached_values(self):
        transform = np.array([[0., -3., 0., .4], [2., 0., 0., -.2],
            [0., 0., 4., .5], [0., 0., 0., 1.]])
        box, domain = obstacle_from_primitive('/Box', 'Cube', transform, size=.1)
        self.assertIsNone(domain)
        np.testing.assert_allclose(box.dimensions, [.2, .3, .4])
        np.testing.assert_allclose(quat_matrix(box.orientation), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15)
        transform[:3, 3] = 100
        self.assertEqual(box.position, (.4, -.2, .5))

    def test_reflected_box_has_same_corner_set_with_proper_rotation(self):
        transform = np.diag([-2., 3., 4., 1.])
        box, _ = obstacle_from_primitive('/Box', 'Cube', transform, size=2.)
        rotation = quat_matrix(box.orientation)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)
        corners = np.array(list(itertools.product((-1, 1), repeat=3)))
        actual = corners@np.diag(box.dimensions/np.array([2, 2, 2]))@rotation.T
        expected = corners@transform[:3, :3].T
        self.assertEqual({tuple(row) for row in actual}, {tuple(row) for row in expected})

    def test_sphere_uniform_scaling_and_nonuniform_rejection(self):
        sphere, _ = obstacle_from_primitive('/S', 'Sphere', np.diag([2., 2., 2., 1.]), radius=.1)
        self.assertEqual(sphere.radius, .2)
        with self.assertRaisesRegex(StaticWorldError, 'Nonuniform'):
            obstacle_from_primitive('/S', 'Sphere', np.diag([2., 2., 3., 1.]), radius=.1)

    def test_capsule_axes_align_numeric_z_axis_and_keep_spine_length(self):
        for axis, vector in [('X', [1, 0, 0]), ('Y', [0, 1, 0]), ('Z', [0, 0, 1])]:
            with self.subTest(axis=axis):
                capsule, _ = obstacle_from_primitive('/C', 'Capsule', np.eye(4),
                    radius=.1, height=.4, axis=axis)
                np.testing.assert_allclose(quat_matrix(capsule.orientation)[:, 2], vector, atol=1e-15)
                self.assertEqual(capsule.height, .4)

    def test_cylinder_is_not_silently_replaced_by_capsule(self):
        with self.assertRaisesRegex(StaticWorldError, 'Unsupported.*Cylinder'):
            obstacle_from_primitive('/C', 'Cylinder', np.eye(4), radius=.1, height=.4)

    def test_shear_degenerate_and_invalid_primitive_rejected(self):
        transforms = [np.diag([0., 1., 1., 1.]), np.full((4, 4), np.nan)]
        shear = np.eye(4)
        shear[0, 1] = .2
        transforms.append(shear)
        for transform in transforms:
            with self.assertRaises(StaticWorldError):
                obstacle_from_primitive('/B', 'Cube', transform, size=1.)
        with self.assertRaises(StaticWorldError):
            obstacle_from_primitive('/B', 'Cube', np.eye(4), size=-1.)

    def test_halfspace_slab_top_is_exact_plane_with_certified_domain(self):
        transform = np.eye(4)
        transform[:3, 3] = [0, 0, .7]
        plane, domain = obstacle_from_primitive('/Floor', 'Plane', transform, plane_center=[10, 12, 2])
        rotation = quat_matrix(plane.orientation)
        top_center = np.asarray(plane.position)+rotation[:, 2]*plane.dimensions[2]/2
        np.testing.assert_allclose(top_center, [10, 12, .7])
        self.assertEqual(plane.maximum_surface_error_m, 0.)
        self.assertIn('certified_domain', plane.approximation)
        domain.assert_contains([[10, 12, 3], [10, 12, -4]], [.1, .2])
        with self.assertRaisesRegex(StaticWorldError, 'domain'):
            domain.assert_contains([[35, 12, 0]], [.1])
        with self.assertRaisesRegex(StaticWorldError, 'domain'):
            domain.assert_contains([[10, 12, -49.3]], [.1])

    def test_rotated_reflected_plane_preserves_normal(self):
        transform = np.diag([-2., 3., 4., 1.])
        plane, domain = obstacle_from_primitive('/P', 'Plane', transform, axis='X')
        np.testing.assert_allclose(quat_matrix(plane.orientation)[:, 2], [-1, 0, 0], atol=1e-15)
        domain.assert_contains([[2, 0, 0]], [.1])

    def test_exact_convex_hull_box_and_nonbox_rejection(self):
        points = np.array(list(itertools.product((-1., 1.), repeat=3)))
        transform = np.diag([.2, .3, .4, 1.])
        transform[:3, 3] = [.4, -.2, .5]
        box = box_from_convex_hull_points('/Mesh', points, transform)
        np.testing.assert_allclose(box.dimensions, [.4, .6, .8])
        self.assertEqual(box.source_kind, 'Mesh:convexHull_exact_box')
        with self.assertRaisesRegex(StaticWorldError, 'eight'):
            box_from_convex_hull_points('/Mesh', points[:-1], transform)
        with self.assertRaisesRegex(StaticWorldError, 'exact cuboid'):
            box_from_convex_hull_points('/Mesh', np.vstack([points, [0, 0, 0]]), transform)

    def test_snapshot_is_frozen_unique_and_validates_all_domains(self):
        box = StaticObstacle('/B', 'cuboid', (0, 0, 0), dimensions=(1, 1, 1))
        world = StaticWorldSnapshot([box], 1., 1., '/Robot',
            plane_domains=[PlaneDomain('/P', (0, 0, 0), (1, 0, 0, 0))])
        self.assertIsInstance(world.obstacles, tuple)
        with self.assertRaises(StaticWorldError):
            StaticWorldSnapshot((box, box), 1., 1., '/Robot')
        with self.assertRaises(StaticWorldError):
            world.assert_within_domains([[26, 0, 2]], [.1])
        self.assertEqual(world.as_dict()['obstacles'][0]['kind'], 'cuboid')


class Attr:
    def __init__(self, value):
        self.value = value

    def Get(self, *unused):
        return self.value


class Prim:
    def __init__(self, path, kind='Cube', *, collision=True, enabled=True,
                 parent=None, dynamic=False, kinematic=False, loaded=True, errors=(), transform=None):
        self.path, self.kind = path, kind
        self.collision, self.enabled, self.parent = collision, enabled, parent
        self.dynamic, self.kinematic, self.loaded, self.errors = dynamic, kinematic, loaded, errors
        self.transform = np.eye(4) if transform is None else transform

    def GetPath(self): return self.path
    def GetTypeName(self): return self.kind
    def IsActive(self): return True
    def IsDefined(self): return True
    def IsAbstract(self): return False
    def IsLoaded(self): return self.loaded
    def IsValid(self): return True
    def IsPseudoRoot(self): return False
    def GetParent(self): return self.parent
    def GetPrimIndex(self): return SimpleNamespace(localErrors=self.errors)
    def GetAttribute(self, name): return Attr('Z' if name == 'axis' else None)
    def GetAttributes(self): return ()
    def IsA(self, kind): return kind is Mesh and self.kind == 'Mesh'
    def HasAPI(self, api):
        return (api is Collision and self.collision) or (api is Rigid and self.dynamic)


class Collision:
    def __init__(self, prim): self.prim = prim
    def GetCollisionEnabledAttr(self): return Attr(self.prim.enabled)


class Rigid:
    def __init__(self, prim): self.prim = prim
    def GetRigidBodyEnabledAttr(self): return Attr(True)
    def GetKinematicEnabledAttr(self): return Attr(self.prim.kinematic)


class Cube:
    def __init__(self, prim): self.prim = prim
    def GetSizeAttr(self): return Attr(2.)


class Mesh:
    pass


class CollectorTests(unittest.TestCase):
    def collect(self, prims, *, units=1.):
        robot = Prim('/Franka', kind='Xform', collision=False)
        stage = SimpleNamespace(GetPrimAtPath=lambda path: robot,
            Traverse=lambda unused: [robot]+prims)
        usd = SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: None),
            TraverseInstanceProxies=lambda value: value, PrimAllPrimsPredicate=None)
        geom = SimpleNamespace(GetStageMetersPerUnit=lambda unused: units, Cube=Cube, Mesh=Mesh,
            XformCache=lambda unused: SimpleNamespace(GetLocalToWorldTransform=lambda prim: prim.transform.T))
        physics = SimpleNamespace(CollisionAPI=Collision, RigidBodyAPI=Rigid)
        with patch(MODULE+'._usd_modules', return_value=(usd, geom, physics)):
            return capture_static_world(stage, timestamp=4.)

    def test_enabled_colliders_only_and_exact_robot_subtree_exclusion(self):
        snapshot = self.collect([Prim('/Franka/Link'), Prim('/FrankaExtra/Box'),
            Prim('/Disabled', enabled=False), Prim('/Visual', collision=False)])
        self.assertEqual([item.path for item in snapshot.obstacles], ['/FrankaExtra/Box'])
        self.assertEqual(snapshot.excluded_colliders, 1)
        self.assertEqual(snapshot.disabled_colliders, 1)

    def test_stage_units_scale_translation_and_dimensions_once(self):
        transform = np.eye(4)
        transform[0, 3] = 20.
        snapshot = self.collect([Prim('/Box', transform=transform)], units=.01)
        self.assertEqual(snapshot.obstacles[0].position, (.2, 0., 0.))
        self.assertEqual(snapshot.obstacles[0].dimensions, (.02, .02, .02))

    def test_dynamic_ancestor_rejected_but_kinematic_supported(self):
        parent = Prim('/Actor', kind='Xform', collision=False, dynamic=True)
        with self.assertRaisesRegex(StaticWorldError, 'Dynamic external'):
            self.collect([Prim('/Actor/Collider', parent=parent)])
        parent.kinematic = True
        self.assertEqual(len(self.collect([Prim('/Actor/Collider', parent=parent)]).obstacles), 1)

    def test_unsupported_enabled_geometry_fails_whole_snapshot(self):
        with self.assertRaisesRegex(StaticWorldError, 'Unsupported.*Cylinder'):
            self.collect([Prim('/Box'), Prim('/FlatCylinder', kind='Cylinder')])

    def test_animated_ancestor_and_shape_are_not_frozen_at_default_time(self):
        parent = Prim('/AnimatedParent', kind='Xform', collision=False)
        parent.GetAttributes = lambda: [SimpleNamespace(GetName=lambda: 'xformOp:translate',
            ValueMightBeTimeVarying=lambda: True)]
        with self.assertRaisesRegex(StaticWorldError, 'Animated.*transform'):
            self.collect([Prim('/AnimatedParent/Box', parent=parent)])
        shape = Prim('/AnimatedSize')
        shape.GetAttributes = lambda: [SimpleNamespace(GetName=lambda: 'size',
            ValueMightBeTimeVarying=lambda: True)]
        with self.assertRaisesRegex(StaticWorldError, 'Animated.*geometry'):
            self.collect([shape])

    def test_external_unloaded_and_composition_errors_fail_closed(self):
        for prim in [Prim('/Unloaded', collision=False, loaded=False),
                     Prim('/MissingReference', collision=False, errors=['unresolved'])]:
            with self.assertRaises(StaticWorldError):
                self.collect([prim])

    def test_physical_plane_is_included_even_with_no_fixture(self):
        snapshot = self.collect([Prim('/Ground', kind='Plane')])
        self.assertEqual(snapshot.obstacles[0].source_kind, 'Plane')
        self.assertEqual(len(snapshot.plane_domains), 1)


if __name__ == '__main__':
    unittest.main()
