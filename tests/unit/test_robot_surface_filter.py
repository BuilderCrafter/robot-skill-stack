from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception.factory import build_tracker
from robot_skill_stack.world.perception.v2.mesh_rays import TriangleBVH, triangulate
from robot_skill_stack.world.perception.v2.mesh_shapes import box_triangles, round_triangles
from robot_skill_stack.world.perception.v2.self_filter import RobotSurfaceModel, RobotSnapshot, SelfFilterResult, filter_robot
from robot_skill_stack.world.perception.v2.pipeline import FrameProcessor
from robot_skill_stack.world.perception.v2.provider import YoloPerceptionProvider
from robot_skill_stack.world.perception.v2.types import Segmentation, Instance
from robot_skill_stack.world.perception.diagnostics import replay_capture
from tests.support.v2_helpers import scene, Solid, INFO, FrameCamera, ManualExecutor, FixtureClient

ROOT = Path(__file__).resolve().parents[2]


def model_for(solids):
    meshes, poses = [], []
    for index, s in enumerate(solids):
        mesh = box_triangles(s.size) if s.shape == 'cube' else round_triangles('Sphere', s.radius)
        meshes.append((f'/Robot/link_{index}', mesh))
        t = np.eye(4); t[:3, 3] = s.center
        c, v = np.cos(s.yaw), np.sin(s.yaw)
        t[:3, :3] = [[c, -v, 0], [v, c, 0], [0, 0, 1]]
        poses.append(t)
    return RobotSurfaceModel(meshes), np.asarray(poses)


class SurfaceFilterTests(unittest.TestCase):
    def setUp(self):
        self.robot = Solid('cube', center=(.45, 0, .15), size=(.15, .10, .12))
        self.frame, self.instances = scene([self.robot], eye=(.45, 0, .9))
        self.model, self.poses = model_for([self.robot])

    def filter(self, frame=None, model=None, poses=None, camera_time=1., pose_time=1., tolerance=.003):
        frame = self.frame if frame is None else frame
        model = self.model if model is None else model
        poses = self.poses if poses is None else poses
        return filter_robot(frame, RobotSnapshot(model, poses, camera_time, pose_time, 'token'), tolerance, .002)

    def test_box_ray_depth_matches_analytic_sensor(self):
        r = self.filter()
        self.assertTrue(np.array_equal(r.mask, self.instances[0].mask))
        self.assertLess(np.max(abs(r.model_depth[r.mask]-self.frame.depth[r.mask])), 1e-6)

    def test_rotated_link_and_oblique_camera(self):
        robot = Solid('cube', center=(.45, 0, .15), size=(.12, .08, .10), yaw=.63)
        frame, masks = scene([robot])
        model, poses = model_for([robot])
        r = self.filter(frame, model, poses)
        self.assertTrue(np.array_equal(r.mask, masks[0].mask))

    def test_foreground_object_not_erased_by_robot_silhouette(self):
        obj = Solid('sphere', center=(.45, 0, .26), radius=.025)
        frame, masks = scene([self.robot, obj], eye=(.45, 0, .9))
        r = self.filter(frame)
        self.assertTrue(np.array_equal(r.mask, masks[0].mask))
        self.assertFalse((r.mask & masks[1].mask).any())
        self.assertGreater(r.metadata['preserved_foreground_pixels'], 0)

    def test_finger_gap_and_object_are_preserved(self):
        fingers = [Solid('cube', center=(.405, 0, .12), size=(.025, .05, .08)),
                   Solid('cube', center=(.495, 0, .12), size=(.025, .05, .08))]
        obj = Solid('sphere', center=(.45, 0, .125), radius=.025)
        frame, masks = scene([*fingers, obj], eye=(.45, 0, .9))
        model, poses = model_for(fingers)
        r = self.filter(frame, model, poses)
        self.assertTrue(np.array_equal(r.mask, masks[0].mask | masks[1].mask))
        self.assertFalse((r.mask & masks[2].mask).any())

    def test_closed_fingers_keep_object_visible_between_pads(self):
        fingers = [Solid('cube', center=(.4125, 0, .12), size=(.025, .06, .06)),
                   Solid('cube', center=(.4875, 0, .12), size=(.025, .06, .06))]
        obj = Solid('cube', center=(.45, 0, .12), size=(.05, .05, .05))
        frame, masks = scene([*fingers, obj], eye=(.45, 0, .9))
        model, poses = model_for(fingers)
        r = self.filter(frame, model, poses)
        self.assertFalse((r.mask & masks[2].mask).any())
        self.assertGreater(r.mask.sum(), 0)

    def test_robot_behind_front_object_remains_depth_aware(self):
        front = Solid('cube', center=(.45, 0, .28), size=(.12, .08, .06))
        frame, masks = scene([self.robot, front], eye=(.45, 0, .9))
        r = self.filter(frame)
        self.assertFalse((r.mask & masks[1].mask).any())
        self.assertTrue(np.isfinite(r.model_depth[masks[1].mask]).all())

    def test_missing_depth_not_interpreted_as_robot(self):
        frame = replace(self.frame, depth=np.full_like(self.frame.depth, np.nan))
        self.assertFalse(self.filter(frame).mask.any())

    def test_empty_robot_projection_removes_nothing(self):
        poses = self.poses.copy(); poses[:, 0, 3] = 9.
        self.assertFalse(self.filter(poses=poses).mask.any())

    def test_behind_camera_mesh_does_not_project(self):
        poses = self.poses.copy(); poses[:, 2, 3] = 2.
        self.assertFalse(self.filter(poses=poses).mask.any())

    def test_moving_link_changes_mask(self):
        a = self.filter()
        moved = replace(self.robot, center=(.60, 0, .15))
        f, masks = scene([moved], eye=(.45, 0, .9))
        model, poses = model_for([moved])
        b = self.filter(f, model, poses)
        self.assertFalse(np.array_equal(a.mask, b.mask))
        self.assertTrue(np.array_equal(b.mask, masks[0].mask))

    def test_mismatched_robot_timestamp_rejected(self):
        with self.assertRaisesRegex(ValueError, 'acquisition time'):
            self.filter(pose_time=1.02)

    def test_snapshot_copies_and_freezes_transforms(self):
        snap = RobotSnapshot(self.model, self.poses, 1., 1.)
        self.poses[0, 0, 3] += 10
        self.assertAlmostEqual(snap.world_from_links[0, 0, 3], .45)
        with self.assertRaises(ValueError):
            snap.world_from_links[0, 0, 3] = 8

    def test_model_mesh_copies_and_freezes_triangles(self):
        t = box_triangles(.1)
        m = RobotSurfaceModel([('link', t)])
        t[:] = 0
        self.assertTrue(np.any(m.bvhs[0].triangles != 0))
        self.assertFalse(m.bvhs[0].triangles.flags.writeable)

    def test_nonrigid_or_nonfinite_poses_rejected(self):
        for value in (2., float('nan')):
            t = self.poses.copy(); t[0, 0, 0] = value
            with self.assertRaises(ValueError):
                RobotSnapshot(self.model, t, 1., 1.)

    def test_ray_axis_parallel_inside_and_outside_box(self):
        bvh = TriangleBVH(box_triangles(2.))
        d = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.]])
        np.testing.assert_allclose(bvh.intersect([0, 0, 0], d), 1.)
        self.assertTrue(np.isinf(bvh.intersect([3, 0, 0], np.array([[0., 0., 1.]]))).all())

    def test_nearest_robot_link_wins_not_farther_link(self):
        near = replace(self.robot, center=(.45, 0, .35))
        f, masks = scene([self.robot, near], eye=(.45, 0, .9))
        m, p = model_for([self.robot, near])
        r = self.filter(f, m, p)
        self.assertTrue(np.array_equal(r.mask, masks[1].mask))

    def test_triangle_validation_and_empty_geometry(self):
        for t in ([], np.zeros((2, 3, 3)), np.full((2, 3, 3), np.nan)):
            with self.assertRaises(ValueError):
                TriangleBVH(t)

    def test_mesh_face_conversion(self):
        v = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        self.assertEqual(triangulate(v, [4], [0, 1, 2, 3]).shape, (2, 3, 3))
        self.assertEqual(triangulate(v, [4], [0, 1, 2, 3], holes=[0]).shape, (0, 3, 3))
        with self.assertRaises(ValueError):
            triangulate(v, [3], [0, 1, 10])

    def test_round_primitive_meshes(self):
        for kind in ('Sphere', 'Cylinder', 'Capsule', 'Cone'):
            for axis in 'XYZ':
                with self.subTest(kind=kind, axis=axis):
                    self.assertGreater(len(TriangleBVH(round_triangles(kind, .025, .1, axis)).triangles), 40)


class RobotProviderIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.robot = Solid('cube', center=(.40, -.05, .15), size=(.08, .08, .12))
        self.obj = Solid('sphere', center=(.51, .08, .025))
        self.frame, self.instances = scene([self.robot, self.obj])
        self.model, self.poses = model_for([self.robot])
        self.camera = FrameCamera(self.frame)
        self.executor = ManualExecutor()
        cfg = load_scene_config(ROOT/'config/scenes/playground_v2.toml')
        settings = replace(cfg.perception.v2, unknown_depth_fallback=True)
        self.now = 100.
        self.source = SimpleNamespace(closed=False)
        self.source.snapshot = lambda token: RobotSnapshot(self.model, self.poses, self.now, self.now, token)
        self.source.close = lambda: setattr(self.source, 'closed', True)
        self.p = YoloPerceptionProvider(self.camera, FrameProcessor(cfg.perception.discovery, cfg.perception.primitives, settings),
                                        build_tracker(cfg), FixtureClient(self.instances), clock=lambda: self.now,
                                        executor=self.executor, robot_source=self.source)
        self.addCleanup(self.p.close)

    def confirm(self):
        for _ in range(3):
            self.camera.token += 1; self.now += .25
            self.p.observe(); self.executor.finish(); obs = self.p.observe()
        return obs

    def test_real_mesh_filter_and_pipeline_confirm_only_environment_object(self):
        obs = self.confirm()
        self.assertEqual(len(obs), 1, self.p.last_error)
        self.assertEqual(obs[0].class_name, 'sphere')
        self.assertTrue(np.any(self.p._last_result.self_filter.mask))

    def test_snapshot_used_after_robot_moves_during_worker_delay(self):
        self.p.observe()
        self.poses[0, 0, 3] += 3.
        self.executor.finish(); self.p.observe()
        self.assertEqual(len(self.p.tracker.tracks()), 1, self.p.last_error)
        self.assertEqual(self.p._last_result.self_filter.metadata['world_from_links'][0][0][3], .40)

    def test_raw_rgb_unchanged_for_yolo(self):
        before = self.frame.rgb.copy()
        self.p.observe(); self.executor.finish(); self.p.observe()
        np.testing.assert_array_equal(self.p.client.calls[0][1], before)

    def test_no_matching_snapshot_never_submits_unfiltered_job(self):
        self.source.snapshot = lambda token: (_ for _ in ()).throw(RuntimeError('time mismatch'))
        self.p.observe()
        self.assertFalse(self.executor.jobs)
        self.assertIn('time mismatch', self.p.last_error)
        self.assertEqual(self.p.status, 'camera error')

    def test_source_subscription_closed_on_provider_stop(self):
        self.p.close()
        self.assertTrue(self.source.closed)

    def test_capture_replay_and_visual_report_preserve_self_mask(self):
        self.confirm()
        with tempfile.TemporaryDirectory() as tmp:
            capture = self.p.save_capture(Path(tmp)/'capture.npz')
            with np.load(capture, allow_pickle=False) as a:
                np.testing.assert_array_equal(a['robot_mask'], self.p._last_result.self_filter.mask)
            report = replay_capture(capture)
            self.assertEqual(report['candidates'], self.p.diagnostics)
            self.assertTrue(report['robot_self_filter']['enabled'])
            from tools.perception_v2.inspect_capture import inspect
            inspect(capture, Path(tmp)/'review')
            self.assertTrue((Path(tmp)/'review'/'review.html').is_file())

    def test_missing_source_errors_without_affecting_v1(self):
        self.p.robot_source = None
        self.p.observe()
        self.assertIn('adapter is missing', self.p.last_error)
        self.assertFalse(self.executor.jobs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
