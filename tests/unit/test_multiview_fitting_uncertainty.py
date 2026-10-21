"""Current-sample fallback and geometric uncertainty guards."""
import unittest

import numpy as np

from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from robot_skill_stack.world.perception_multiview.fitting import PrimitiveFit, SurfaceFitter, unknown_fit


class UnknownGeometryFallbackTests(unittest.TestCase):
    def test_budget_fallback_uses_finite_current_bounds(self):
        points = np.array([[.4, -.1, .02], [.5, .1, .08], [np.nan, 0., 0.]])
        fit = unknown_fit(points, reason='candidate_budget_exhausted')
        self.assertEqual(fit.geometry.shape.value, 'unknown')
        self.assertEqual(fit.geometry.confidence, 0.)
        self.assertEqual(fit.geometry.metadata['reason'], 'candidate_budget_exhausted')
        self.assertEqual(fit.geometry.metadata['fit_points'], 2)
        np.testing.assert_allclose(fit.center, [.45, 0., .05])
        np.testing.assert_allclose(fit.size, [.1, .2, .06])

    def test_empty_fallback_and_explicit_support_prior(self):
        fit = unknown_fit(np.empty((0, 3)))
        np.testing.assert_array_equal(fit.center, np.zeros(3))
        np.testing.assert_array_equal(fit.size, np.zeros(3))
        supported = unknown_fit([[.4, 0., .02], [.5, .1, .08]], support_plane_z=0.)
        self.assertAlmostEqual(supported.center[2], .04)
        self.assertAlmostEqual(supported.size[2], .08)
        with self.assertRaises(ValueError):
            unknown_fit([], support_plane_z=np.nan)


class ConservativeParameterTests(unittest.TestCase):
    def sphere(self, *, center_z=.03):
        radius = .03
        angle, height = np.meshgrid(np.linspace(0, 2*np.pi, 48, endpoint=False), np.linspace(0, 1, 16))
        radial = np.sqrt(1-height.ravel()**2)
        center = np.array([.45, 0., center_z])
        points = center+radius*np.c_[radial*np.cos(angle.ravel()), radial*np.sin(angle.ravel()), height.ravel()]
        fit = PrimitiveFit(PrimitiveGeometry('sphere', .99, radius=radius), center, np.full(3, 2*radius))
        return points, fit

    def cylinder(self, angles, *, lying=False, caps=2):
        radius, length = .03, .09
        theta, along = np.meshgrid(angles, np.linspace(-length/2, length/2, 16))
        axis = np.array([1., 0., 0.]) if lying else np.array([0., 0., 1.])
        center = np.array([.45, 0., radius if lying else length/2])
        side = np.array([0., 1., 0.]) if lying else np.array([1., 0., 0.])
        other = np.cross(axis, side)
        points = (center+along.ravel()[:, None]*axis+
                  radius*np.cos(theta.ravel())[:, None]*side+radius*np.sin(theta.ravel())[:, None]*other)
        cap_angle, cap_radius = np.meshgrid(np.linspace(0, 2*np.pi, 12, endpoint=False), [0., .3*radius, .6*radius])
        interior = cap_radius.ravel()[:, None]*(np.cos(cap_angle.ravel())[:, None]*side+
                                               np.sin(cap_angle.ravel())[:, None]*other)
        ends = [-1., 1.][:caps] if lying else [1.]
        points = np.concatenate([points]+[center+end*length/2*axis+interior for end in ends])
        size = abs(axis)*length+2*radius*np.sqrt(1-axis*axis)
        fit = PrimitiveFit(PrimitiveGeometry('cylinder', .99, radius=radius, length=length, axis=axis), center, size)
        return points, fit

    def test_exact_sphere_has_no_arbitrary_radius_inflation(self):
        points, fit = self.sphere()
        result = SurfaceFitter().finalize(fit, points, view_clouds=(points[::2], points[1::2]))
        self.assertTrue(result.geometry.metadata['parameter_observable'])
        self.assertAlmostEqual(result.geometry.radius, .03, places=12)
        np.testing.assert_allclose(result.center, fit.center, atol=1e-12)
        self.assertNotIn('parameter_uncertainty', fit.geometry.metadata)
        self.assertEqual(result.geometry.confidence, fit.geometry.confidence)

    def test_elevated_sphere_is_not_projected_onto_support(self):
        points, fit = self.sphere(center_z=.05)
        result = SurfaceFitter().finalize(fit, points)
        self.assertFalse(result.geometry.metadata['parameter_observable'])
        self.assertEqual(result.geometry.metadata['parameter_uncertainty']['reason'], 'unobserved_support_contact')
        np.testing.assert_array_equal(result.center, fit.center)

    def test_short_upright_arc_stays_known_but_dimension_unobservable(self):
        points, fit = self.cylinder(np.linspace(0., np.pi/2, 48))
        result = SurfaceFitter().finalize(fit, points)
        self.assertEqual(result.geometry.shape.value, 'cylinder')
        self.assertFalse(result.geometry.metadata['parameter_observable'])
        self.assertEqual(result.geometry.metadata['parameter_uncertainty']['reason'], 'insufficient_curvature_extent')

    def test_full_curvature_and_cap_preserve_supported_cylinder(self):
        points, fit = self.cylinder(np.linspace(0., 2*np.pi, 96, endpoint=False))
        result = SurfaceFitter().finalize(fit, points)
        self.assertTrue(result.geometry.metadata['parameter_observable'])
        self.assertAlmostEqual(result.geometry.radius, fit.geometry.radius, places=12)
        self.assertAlmostEqual(result.geometry.length, fit.geometry.length)

    def test_lying_length_requires_both_measured_endpoints(self):
        for caps, expected in ((1, False), (2, True)):
            points, fit = self.cylinder(np.linspace(0., 2*np.pi, 96, endpoint=False), lying=True, caps=caps)
            result = SurfaceFitter().finalize(fit, points)
            self.assertEqual(result.geometry.metadata['parameter_observable'], expected)


if __name__ == '__main__':
    unittest.main()
