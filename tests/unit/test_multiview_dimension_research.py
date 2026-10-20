"""Ensure rendering ablations change tessellation, not nominal fixture size."""
import unittest
from collections import Counter

import numpy as np

from tests.support.multiview_dimension_research import mesh_surface
from tests.support.multiview_research import ResearchObject


class DimensionResearchMeshTests(unittest.TestCase):
    def assert_closed(self, vertices, faces):
        self.assertTrue(np.isfinite(vertices).all())
        self.assertTrue((faces >= 0).all())
        self.assertTrue((faces < len(vertices)).all())
        edges = Counter(tuple(sorted((int(face[i]), int(face[(i+1) % 3]))))
                        for face in faces for i in range(3))
        self.assertTrue(all(count == 2 for count in edges.values()))
        triangles = vertices[faces]
        area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                      triangles[:, 2]-triangles[:, 0]), axis=1)
        self.assertTrue((area > 0).all())

    def test_sphere_mesh_has_nominal_radius_and_closed_surface(self):
        obj = ResearchObject('sphere', 'sphere', np.array([.45, -.1, .0355]),
                             np.full(3, .071), radius=.0355)
        for segments in (16, 32):
            vertices, faces = mesh_surface(obj, segments)
            self.assert_closed(vertices, faces)
            np.testing.assert_allclose(np.linalg.norm(vertices-obj.center, axis=1), obj.radius, atol=1e-12)

    def test_cylinder_mesh_preserves_radius_length_and_axis(self):
        for axis in ((0., 0., 1.), (np.cos(.43), np.sin(.43), 0.)):
            obj = ResearchObject('cylinder', 'cylinder', np.array([.45, 0., .045]),
                                 np.array([.071, .071, .09]), radius=.0355, length=.09, axis=axis)
            vertices, faces = mesh_surface(obj, 32)
            self.assert_closed(vertices, faces)
            delta = vertices[:64]-obj.center
            axial = delta @ np.asarray(axis)
            radial = np.linalg.norm(delta-axial[:, None]*axis, axis=1)
            np.testing.assert_allclose(abs(axial), obj.length/2, atol=1e-12)
            np.testing.assert_allclose(radial, obj.radius, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
