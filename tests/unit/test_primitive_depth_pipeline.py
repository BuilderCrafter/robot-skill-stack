from __future__ import annotations
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception.diagnostics import save_capture, replay_capture
from robot_skill_stack.world.perception.discovery import DepthObjectDiscoverer
from robot_skill_stack.world.perception.factory import build_perception_provider
from robot_skill_stack.world.perception.localizer import RgbdLocalizer
from robot_skill_stack.world.perception.primitives import PrimitiveEstimator
from robot_skill_stack.world.perception.semantics import SemanticPrediction
from tests.support.primitive_scene import FrameCamera, Solid, render


class PrimitiveDepthTests(unittest.TestCase):
    def test_single_view_matrix(self):
        eyes = [(1.05, .55, .9), (.45, 0., .9), (1.05, .10, .40), (-.2, -.5, .7)]
        solids = [Solid('cube', yaw=np.deg2rad(y)) for y in (-30, 0, 15, 30, 45, 60, 89)]
        solids += [Solid('sphere', center=(.45, 0, r), radius=r) for r in (.02, .025)]
        solids += [Solid('cylinder', center=(.45, 0, L/2), radius=r, length=L)
                   for r, L in ((.02, .1), (.025, .03), (.02, .06))]
        solids += [Solid('cylinder', center=(.45, 0, .02), radius=.02, axis=(np.cos(a), np.sin(a), 0))
                   for a in (-.8, 0, .5, 1.25)]
        solids += [Solid('ellipsoid', size=(.08, .05, .05))]
        count = 0
        for eye in eyes:
            for solid in solids:
                for noise in (0., .00025):
                    with self.subTest(eye=eye, solid=solid, noise=noise):
                        frame = render([solid], eye=eye, noise=noise, dropout=.03)
                        candidates = DepthObjectDiscoverer(RgbdLocalizer(frame.intrinsics)).discover(frame)
                        self.assertEqual(len(candidates), 1)
                        candidate = candidates[0]
                        self.assertTrue(candidate.spawnable)
                        g = PrimitiveEstimator().estimate(candidate)
                        expected = 'unknown' if solid.shape == 'ellipsoid' else solid.shape
                        self.assertEqual(g.shape.value, expected, g.scores)
                        if expected != 'unknown':
                            self.assertLess(np.linalg.norm(np.array(g.metadata['center'])-solid.center), .007)
                        if expected == 'cube':
                            error = abs((g.yaw-solid.yaw+np.pi/4) % (np.pi/2)-np.pi/4)
                            self.assertLess(error, np.deg2rad(5))
                            np.testing.assert_allclose(g.box_size, solid.size, atol=.006)
                        elif expected == 'sphere':
                            self.assertIsNone(g.yaw)
                            self.assertIsNone(g.axis)
                            self.assertLess(abs(g.radius-solid.radius), .003)
                        elif expected == 'cylinder':
                            self.assertGreater(abs(g.axis @ solid.axis), np.cos(np.deg2rad(5)))
                            self.assertLess(abs(g.radius-solid.radius), .003)
                            self.assertLess(abs(g.length-solid.length), .01)
                        count += 1
        self.assertEqual(count, 136)

    def test_tilted_cylinder_is_not_forced_into_supported_axis(self):
        axis = np.array([.5, 0., np.sqrt(.75)])
        solid = Solid('cylinder', center=(.45, 0., .05*axis[2]+.02*np.sqrt(1-axis[2]**2)),
                      radius=.02, axis=tuple(axis))
        frame = render([solid])
        candidate = DepthObjectDiscoverer(RgbdLocalizer(frame.intrinsics)).discover(frame)[0]
        g = PrimitiveEstimator().estimate(candidate)
        self.assertEqual(g.shape.value, 'unknown', g.scores)

    def test_invalid_and_sparse_points(self):
        for points in ([], np.full((40, 3), np.nan), np.full((40, 3), np.inf), np.zeros((10, 3)), np.zeros((40, 3))):
            with self.subTest(shape=np.shape(points)):
                g = PrimitiveEstimator().estimate(SimpleNamespace(metadata={'points': points}))
                self.assertEqual(g.shape.value, 'unknown')

    def test_image_neighbors_must_also_be_near_in_3d(self):
        mask = np.ones((3, 4), bool)
        world = np.zeros((3, 4, 3))
        world[:, 2:, 2] = .08
        components = DepthObjectDiscoverer._components(mask, world, .02)
        self.assertEqual(sorted(len(xs) for _, xs in components), [6, 6])

    def test_environment_and_outside_workspace_rejected(self):
        for solid in (Solid('cube', size=(.30, .20, .20), center=(.45, 0, .10)),
                      Solid('sphere', center=(.9, 0, .025))):
            frame = render([solid])
            candidates = DepthObjectDiscoverer(RgbdLocalizer(frame.intrinsics)).discover(frame)
            self.assertFalse(any(c.spawnable for c in candidates))

    def test_support_height_not_hardcoded(self):
        solid = Solid('cylinder', radius=.02, center=(.45, 0., .4))
        frame = render([solid], eye=(.45, 0., .9), plane_z=.35)
        d = DepthObjectDiscoverer(RgbdLocalizer(frame.intrinsics), support_plane_z=.35,
                                 workspace_min=(.2, -.4, .35), workspace_max=(.7, .4, .57))
        candidate = d.discover(frame)[0]
        self.assertTrue(candidate.spawnable)
        g = PrimitiveEstimator().estimate(candidate)
        self.assertEqual(g.shape.value, 'cylinder')
        self.assertLess(abs(g.length-.10), .003)
        self.assertLess(abs(g.metadata['center'][2]-.4), .003)

    def test_depth_provider_works_without_rgb(self):
        camera = FrameCamera(render([Solid('sphere')]))
        camera.get_rgb = lambda: None
        provider = build_perception_provider(camera, load_scene_config('config/scenes/playground.toml'))
        with patch('time.monotonic', return_value=100.):
            for _ in range(3):
                camera.token += 1
                observations = provider.observe()
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].class_name, 'sphere')
        self.assertIsNone(observations[0].pose.orientation)

    def test_semantic_identity_is_separate_from_primitive(self):
        camera = FrameCamera(render([Solid('cylinder', center=(.45, 0, .05), radius=.02)]))
        provider = build_perception_provider(camera, load_scene_config('config/scenes/playground.toml'))
        provider.semantic_classifier = SimpleNamespace(classify=lambda *args: SemanticPrediction('bottle', 1.))
        with patch('time.monotonic', return_value=100.):
            for _ in range(3):
                camera.token += 1
                observations = provider.observe()
        self.assertEqual(observations[0].class_name, 'bottle')
        self.assertEqual(observations[0].geometry.shape.value, 'cylinder')

    def test_capture_replay_round_trip(self):
        camera = FrameCamera(render([Solid('sphere')]))
        provider = build_perception_provider(camera, load_scene_config('config/scenes/playground.toml'))
        provider.observe()
        with tempfile.TemporaryDirectory() as directory:
            path = save_capture(provider, Path(directory)/'capture.npz')
            result = replay_capture(path)
            self.assertEqual(result['candidates'][0]['shape'], 'sphere')
            self.assertEqual(result['candidates'], provider.diagnostics)
            with np.load(path, allow_pickle=False) as capture:
                np.testing.assert_array_equal(capture['depth'], camera.frame.depth)

    def test_rotated_rectangular_box_not_confused_with_cylinder(self):
        solid = Solid('cube', center=(.45, 0, .02), size=(.10, .04, .04), yaw=.65)
        frame = render([solid], noise=.00025)
        candidate = DepthObjectDiscoverer(RgbdLocalizer(frame.intrinsics)).discover(frame)[0]
        geometry = PrimitiveEstimator().estimate(candidate)
        self.assertTrue(candidate.spawnable)
        self.assertEqual(geometry.shape.value, 'cube', geometry.scores)
        self.assertLess(abs(geometry.yaw-.65), np.deg2rad(5))
        np.testing.assert_allclose(geometry.box_size, solid.size, atol=.006)


if __name__ == '__main__':
    unittest.main(verbosity=2)
