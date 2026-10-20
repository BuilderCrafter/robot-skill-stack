"""A projected workspace rectangle must never discard an in-workspace sample."""
import unittest
import numpy as np

from robot_skill_stack.integrations.isaac.config import SceneGeometry
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, project_depth
from tests.support.multiview_research import analytic_frames
from tests.support.primitive_scene import Solid


class ProjectionWindowTests(unittest.TestCase):
    def test_window_preserves_all_cropped_points_and_pixel_indices(self):
        objects = [Solid('cube', center=(.35, -.15, .025)),
                   Solid('sphere', center=(.55, .2, .025), radius=.025)]
        scene = SceneGeometry(0., [.2, -.4, 0.], [.7, .4, .22])
        for frame in analytic_frames(objects):
            for stride in (1, 2, 3, 5):
                full, full_pixels = project_depth(frame, stride=stride)
                limited, pixels = project_depth(frame, stride=stride, bounds=(scene.workspace_min, scene.workspace_max))
                np.testing.assert_allclose(crop_workspace(full, scene), crop_workspace(limited, scene), atol=1e-12)
                full_lookup = {tuple(pixel): point for pixel, point in zip(full_pixels, full)}
                np.testing.assert_allclose(limited, [full_lookup[tuple(pixel)] for pixel in pixels], atol=1e-12)

    def test_optical_plane_crossing_uses_complete_projection(self):
        frame = analytic_frames([])[0]
        position = frame.world_from_camera[:3, 3]
        full, pixels = project_depth(frame)
        limited, limited_pixels = project_depth(frame, bounds=(position-.2, position+.2))
        np.testing.assert_array_equal(full, limited)
        np.testing.assert_array_equal(pixels, limited_pixels)

    def test_invalid_bounds_are_rejected(self):
        frame = analytic_frames([])[0]
        for bounds in (([0., 0., 0.], [-1., 1., 1.]), ([np.nan, 0., 0.], [1., 1., 1.])):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                project_depth(frame, bounds=bounds)
