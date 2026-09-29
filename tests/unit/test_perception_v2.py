from __future__ import annotations
import ast
import builtins
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile
import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.context import ObservationContext, AssociationHint
from robot_skill_stack.world.perception.factory import build_perception_provider, build_tracker
from robot_skill_stack.world.perception.diagnostics import save_capture, replay_capture
from robot_skill_stack.world.perception.v2.settings import V2Config
from robot_skill_stack.world.perception.v2.types import Instance, Segmentation, validate_frame
from robot_skill_stack.world.perception.v2.pipeline import FrameProcessor
from robot_skill_stack.world.perception.v2.provider import YoloPerceptionProvider
from robot_skill_stack.world.perception.v2.client import VisionClient
from robot_skill_stack.world.perception.v2 import wire
from robot_skill_stack.integrations.vision.yolo_segmenter import YoloSegmenter
from tests.support.v2_helpers import INFO, scene, FixtureClient, ManualExecutor, FrameCamera, Solid

ROOT = Path(__file__).resolve().parents[2]


class V2GeometryTests(unittest.TestCase):
    def setUp(self):
        cfg = load_scene_config(ROOT/'config/scenes/playground.toml')
        self.cfg = replace(cfg, perception=replace(cfg.perception, type='yolo_v2', v2=V2Config(unknown_depth_fallback=False)))
        self.processor = FrameProcessor(self.cfg.perception.discovery, self.cfg.perception.primitives, self.cfg.perception.v2)

    def process(self, solids=(Solid('sphere'),), **kwargs):
        frame, instances = scene(solids, **kwargs)
        result = self.processor.process(frame, Segmentation(frame.frame_id, instances, INFO))
        return frame, result

    def test_cube_sphere_and_cylinders_geometry(self):
        solids = [Solid('cube', yaw=.5), Solid('sphere'),
                  Solid('cylinder', center=(.45, 0, .05), radius=.02),
                  Solid('cylinder', center=(.45, 0, .02), radius=.02, axis=(1., 0., 0.))]
        for s in solids:
            with self.subTest(shape=s):
                _, result = self.process([s])
                self.assertEqual(len(result.candidates), 1)
                c = result.candidates[0]
                self.assertEqual(c.metadata['geometry'].shape.value, s.shape)
                self.assertLess(np.linalg.norm(c.position-s.center), .007)
                if s.shape == 'sphere':
                    self.assertIsNone(c.metadata['geometry'].yaw)
                elif s.shape == 'cube':
                    self.assertLess(abs((c.metadata['geometry'].yaw-s.yaw+np.pi/4)%(np.pi/2)-np.pi/4), .1)

    def test_two_touching_cubes_remain_two_masks(self):
        _, result = self.process([Solid('cube', center=(.42, 0, .025)), Solid('cube', center=(.47, 0, .025))])
        self.assertEqual(len(result.candidates), 2)

    def test_no_rgb_depth_frame_mixing(self):
        frame, instances = scene()
        with self.assertRaisesRegex(ValueError, 'frame IDs'):
            self.processor.process(frame, Segmentation(77, instances))

    def test_mask_shape_mismatch_rejected(self):
        frame, _ = scene()
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            self.processor.process(frame, Segmentation(frame.frame_id, [Instance(np.ones((10, 10), bool), 'cube', .9)]))

    def test_invalid_depth_does_not_fabricate_world_objects(self):
        frame, instances = scene()
        frame = replace(frame, depth=np.full_like(frame.depth, np.nan))
        result = self.processor.process(frame, Segmentation(frame.frame_id, instances))
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.diagnostics[0]['rejected'], 'insufficient_valid_depth')

    def test_low_confidence_filtered(self):
        frame, instances = scene()
        weak = Instance(instances[0].mask, 'sphere', .1)
        self.assertFalse(self.processor.process(frame, Segmentation(frame.frame_id, [weak])).candidates)

    def test_wrong_visual_label_never_forces_wrong_geometry(self):
        frame, instances = scene()
        wrong = Instance(instances[0].mask, 'cube', .95)
        result = self.processor.process(frame, Segmentation(frame.frame_id, [wrong]))
        self.assertEqual(result.predictions[0].label, 'cube')
        self.assertEqual(result.candidates[0].metadata['geometry'].shape.value, 'unknown')

    def test_lifted_cube_not_stretched_to_table(self):
        _, result = self.process([Solid('cube', center=(.45, 0, .145))])
        c = result.candidates[0]
        self.assertLess(c.size[2], .06)
        self.assertGreater(c.position[2], .12)
        self.assertEqual(c.metadata['geometry'].shape.value, 'unknown')

    def test_lifted_sphere_keeps_metric_radius(self):
        _, result = self.process([Solid('sphere', center=(.45, 0, .20))])
        c = result.candidates[0]
        self.assertEqual(c.metadata['geometry'].shape.value, 'sphere')
        self.assertAlmostEqual(c.metadata['geometry'].radius, .025, delta=.003)
        self.assertAlmostEqual(c.position[2], .2, delta=.003)

    def test_top_only_cylinder_is_detected_but_height_uncertain(self):
        _, result = self.process([Solid('cylinder', center=(.45, 0, .05), radius=.02)], eye=(.45, 0, .9))
        self.assertEqual(result.predictions[0].label, 'cylinder')
        self.assertEqual(result.candidates[0].metadata['geometry'].shape.value, 'unknown')

    def test_unknown_depth_fallback_and_no_duplicates(self):
        self.processor.config = replace(self.processor.config, unknown_depth_fallback=True)
        frame, instances = scene()
        result = self.processor.process(frame, Segmentation(frame.frame_id, instances))
        self.assertEqual(len(result.candidates), 1)
        empty = self.processor.process(frame, Segmentation(frame.frame_id, []))
        self.assertEqual(len(empty.candidates), 1)
        self.assertIsNone(empty.predictions[0].label)
        self.assertEqual(empty.candidates[0].metadata['geometry'].shape.value, 'unknown')

    def test_outside_workspace_rejected(self):
        _, result = self.process([Solid('sphere', center=(.9, 0., .025))])
        self.assertFalse(result.candidates)

    def test_bad_camera_transform_rejected(self):
        frame, _ = scene()
        T = frame.world_from_camera.copy()
        T[:3, :3] *= 2
        with self.assertRaises(ValueError):
            validate_frame(replace(frame, world_from_camera=T))

    def test_factory_default_and_explicit_switch(self):
        camera = FrameCamera(scene()[0])
        v1 = build_perception_provider(camera, replace(self.cfg, perception=replace(self.cfg.perception, type='geometry_v1')))
        self.assertEqual(v1.source, 'rgbd_geometry_perception')
        v2 = build_perception_provider(camera, self.cfg)
        self.addCleanup(v2.close)
        self.assertIsInstance(v2, YoloPerceptionProvider)

    def test_v1_import_and_v2_factory_do_not_import_ml_or_simulator(self):
        original = builtins.__import__
        def guard(name, *a, **kw):
            if name.split('.')[0] in {'torch', 'torchvision', 'ultralytics', 'cv2', 'isaacsim', 'omni', 'pxr'}:
                raise AssertionError(name)
            return original(name, *a, **kw)
        with patch('builtins.__import__', side_effect=guard):
            provider = build_perception_provider(FrameCamera(scene()[0]), self.cfg)
            provider.close()

    def test_invalid_config_does_not_silently_fall_back(self):
        for args in ({'confidence': float('nan')}, {'confidence': 0}, {'max_detections': 0},
                     {'max_detections': True}, {'request_hz': 0}, {'mask_erode_pixels': 4},
                     {'endpoint': 'https://remote.example'}, {'endpoint': 'http://127.0.0.1:8765/path'},
                     {'max_result_age_s': 8.}, {'unknown_depth_fallback': 'yes'}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                V2Config(**args)
        with self.assertRaises(ValueError):
            replace(self.cfg.perception, type='typo')


class V2ProviderTests(unittest.TestCase):
    def setUp(self):
        self.frame, instances = scene()
        self.camera, self.client = FrameCamera(self.frame), FixtureClient(instances)
        self.executor = ManualExecutor()
        self.now = 100.
        self.cfg = load_scene_config(ROOT/'config/scenes/playground_v2.toml')
        settings = replace(self.cfg.perception.v2, unknown_depth_fallback=False)
        self.p = YoloPerceptionProvider(self.camera, FrameProcessor(self.cfg.perception.discovery, self.cfg.perception.primitives, settings),
                                        build_tracker(self.cfg), self.client, clock=lambda: self.now, executor=self.executor)
        self.addCleanup(self.p.close)

    def complete(self, context=None):
        self.p.observe(context)
        self.executor.finish()
        return self.p.observe(context)

    def confirm(self):
        for _ in range(3):
            self.now += .25
            self.camera.token += 1
            observations = self.complete()
        return observations

    def test_no_blocking_and_only_one_outstanding_job(self):
        self.assertEqual(self.p.observe(), [])
        self.now += 1.
        for _ in range(20):
            self.p.observe()
        self.assertEqual(len(self.executor.jobs), 1)
        self.assertEqual(self.client.calls, [])

    def test_confirmation_requires_distinct_fresh_frames(self):
        self.assertEqual(self.complete(), [])
        for _ in range(8):
            self.now += .2
            self.p.observe()
        self.assertEqual(self.p.tracker.tracks()[0].hits, 1)
        self.camera.token += 1
        self.complete()
        self.now += .25
        self.camera.token += 1
        obs = self.complete()
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].class_name, 'sphere')
        self.assertTrue(obs[0].graspable)
        self.assertEqual(obs[0].timestamp, self.now)

    def test_snapshot_is_copied_and_old_calibration_used(self):
        self.p.observe()
        frame = self.executor.jobs[0][2][0]
        self.camera.frame.rgb[:] = 129
        self.assertFalse(frame.rgb.any())
        moved = self.frame.world_from_camera.copy()
        moved[0, 3] += .4
        self.camera.push(replace(self.frame, world_from_camera=moved))
        self.executor.finish()
        self.p.observe()
        self.assertLess(abs(self.p.tracker.tracks()[0].position[0]-.45), .01)

    def test_stale_result_dropped_without_refreshing_last_seen(self):
        self.p.observe()
        self.now += 3.
        self.executor.finish()
        self.p.observe()
        self.assertEqual(self.p.stale_results, 1)
        self.assertFalse(self.p.tracker.tracks())
        self.assertIn('old', self.p.last_error)

    def test_worker_error_is_visible_not_a_fake_empty_detection(self):
        self.confirm()
        self.client.predict = lambda *a: (_ for _ in ()).throw(RuntimeError('offline'))
        self.now += .25
        self.camera.token += 1
        self.complete()
        self.assertIn('offline', self.p.last_error)
        self.assertEqual(self.p.tracker.tracks()[0].hits, 3)
        self.now += 3.
        obs = self.p.observe()
        self.assertFalse(obs[0].visible)

    def test_frozen_camera_eventually_lost_and_expired(self):
        obs = self.confirm()
        self.now += 3
        obs = self.p.observe()
        self.assertFalse(obs[0].visible)
        self.assertEqual(self.p.status, 'stale')
        self.assertIsNone(self.p.get_mask(obs[0].object_id))
        self.now += 20
        self.assertEqual(self.p.observe(), [])

    def test_held_and_hint_expiry_protection(self):
        obj = self.confirm()[0]
        self.now += 30
        ctx = ObservationContext(held_object_id=obj.object_id)
        self.assertEqual(len(self.p.observe(ctx)), 1)
        hint = AssociationHint(obj.object_id, np.array([.45, 0, .025]), .2, self.now+10, 'place')
        self.assertEqual(len(self.p.observe(ObservationContext(association_hints=(hint,)))), 1)
        self.now += 11
        self.assertFalse(self.p.observe())

    def test_hint_reacquisition_keeps_id(self):
        old = self.confirm()[0]
        self.frame, instances = scene([Solid('sphere', center=(.45, .28, .025))])
        self.camera.push(self.frame)
        self.client.instances = instances
        self.now += .25
        hint = AssociationHint(old.object_id, np.array([.45, .28, .025]), .1, self.now+10, 'place')
        obs = self.complete(ObservationContext(association_hints=(hint,)))
        self.assertEqual(obs[0].object_id, old.object_id)
        self.assertTrue(obs[0].metadata['association_hint_match'])

    def test_clear_lost_clears_tracker_and_world(self):
        model = WorldModel()
        obs = self.confirm()
        with patch('time.monotonic', return_value=self.now):
            model.apply_observations(obs)
        self.now += 3
        model.apply_observations(self.p.observe())
        removed = model.clear_lost(self.p.forget)
        self.assertEqual(removed, (obs[0].object_id,))
        self.assertFalse(self.p.tracker.tracks())
        self.assertFalse(model.objects())

    def test_missing_rgb_never_uses_black_fallback(self):
        self.camera.get_rgb = lambda: None
        self.p.observe()
        self.assertFalse(self.executor.jobs)

    def test_missing_timestamp_never_reconfirms_frozen_frame(self):
        self.camera.get_frame_token = lambda: None
        self.p.observe()
        self.assertFalse(self.executor.jobs)
        self.assertIn('timestamped', self.p.status)

    def test_camera_token_changed_during_capture_is_rejected(self):
        tokens = iter([1, 2])
        self.camera.get_frame_token = lambda: next(tokens)
        self.p.observe()
        self.assertFalse(self.executor.jobs)

    def test_unknown_geometry_is_not_graspable_even_if_class_known(self):
        self.frame, instances = scene([Solid('cube', center=(.45, 0, .2))])
        self.camera.push(self.frame)
        self.client.instances = instances
        obs = self.confirm()
        self.assertEqual(obs[0].class_name, 'cube')
        self.assertFalse(obs[0].graspable)

    def test_lazy_point_cloud_uses_last_accepted_frame(self):
        obj = self.confirm()[0]
        cloud = self.p.get_point_cloud(obj.object_id)
        self.assertGreater(len(cloud), 30)
        self.assertLess(abs(cloud.mean(0)[0]-.45), .03)
        self.p.forget(obj.object_id)
        self.assertIsNone(self.p.get_point_cloud(obj.object_id))

    def test_capture_and_replay_no_service_needed(self):
        self.confirm()
        with tempfile.TemporaryDirectory() as directory:
            path = save_capture(self.p, Path(directory)/'v2.npz')
            report = replay_capture(path)
            self.assertEqual(report['candidates'], self.p.diagnostics)
            self.assertIn('saved masks', report['mode'])
            self.assertEqual(report['source'], 'rgbd_yolo_v2')

    def test_worker_model_change_requires_reinitialization(self):
        self.confirm()
        self.client.predict = lambda frame_id, rgb: Segmentation(frame_id, self.client.instances,
                                                                 {**INFO, 'sha256': '1'*64})
        self.now += .25
        self.camera.token += 1
        self.complete()
        self.assertIn('weights changed', self.p.last_error)
        self.assertEqual(self.p.tracker.tracks()[0].hits, 3)

    def test_real_executor_runs_inference_off_the_calling_thread(self):
        thread_ids = []
        client = FixtureClient(self.client.instances)
        original = client.predict
        def predict(*args):
            thread_ids.append(threading.get_ident())
            return original(*args)
        client.predict = predict
        p = YoloPerceptionProvider(self.camera, self.p.processor, build_tracker(self.cfg), client)
        self.addCleanup(p.close)
        main_id = threading.get_ident()
        p.observe()
        deadline = time.monotonic()+8
        while not p.processed_frames and time.monotonic() < deadline:
            p.observe()
            time.sleep(.01)
        self.assertEqual(p.processed_frames, 1, p.last_error)
        self.assertEqual(len(thread_ids), 1)
        self.assertNotEqual(thread_ids[0], main_id)

    def test_stop_closes_worker_even_before_callback_started(self):
        updater = WorldModelUpdater(WorldModel(), self.p)
        updater.stop(SimpleNamespace())
        self.assertTrue(self.executor.stopped)
        self.assertEqual(self.p.observe(), [])
        updater.stop(SimpleNamespace())

    def test_close_cancels_pending_job(self):
        self.p.observe()
        f = self.executor.jobs[0][0]
        self.p.close()
        self.assertTrue(f.cancelled())


class V2TransportTests(unittest.TestCase):
    def test_request_round_trip_and_rgb_order(self):
        rgb = np.array([[[255, 3, 11]]], np.uint8)
        data = wire.encode_request(42, rgb, V2Config().inference_options())
        frame_id, found, options = wire.decode_request(data)
        self.assertEqual(frame_id, 42)
        np.testing.assert_array_equal(found, rgb)
        self.assertEqual(options['confidence'], .35)

    def test_empty_response_is_valid(self):
        response = Segmentation(9, [], INFO)
        result = wire.decode_response(wire.encode_response(response, (480, 640)), 9, (480, 640))
        self.assertEqual(result.instances, [])

    def test_different_frame_rejected(self):
        data = wire.encode_response(Segmentation(8, [], INFO), (10, 10))
        with self.assertRaises(ValueError):
            wire.decode_response(data, 9, (10, 10))

    def test_mask_round_trip_and_native_resolution_check(self):
        mask = np.eye(12, dtype=bool)
        data = wire.encode_response(Segmentation(2, [Instance(mask, 'cube', .8)], INFO), (12, 12))
        result = wire.decode_response(data, 2, (12, 12))
        np.testing.assert_array_equal(mask, result.instances[0].mask)
        with self.assertRaises(ValueError):
            wire.decode_response(data, 2, (6, 6))

    def test_pickle_arrays_rejected(self):
        data = wire.pack(rgb=np.array([object()], dtype=object), metadata='{}')
        with self.assertRaises(ValueError):
            wire.decode_request(data)

    def test_unknown_model_classes_rejected(self):
        data = wire.encode_response(Segmentation(2, [], {'classes': ['person']}), (10, 10))
        with self.assertRaises(ValueError):
            wire.decode_response(data, 2, (10, 10))

    def test_invalid_mask_dtype_and_score(self):
        for mask, score in ((np.zeros((4, 4), np.uint8), .9), (np.zeros((4, 4), bool), float('nan'))):
            with self.assertRaises(ValueError):
                Instance(mask, 'cube', score)

    def test_http_round_trip_with_real_socket_test_double_model(self):
        from tools.perception_v2.serve import make_server
        model = SimpleNamespace(info=INFO, predict=lambda frame_id, rgb, **k: Segmentation(frame_id, [Instance(np.ones(rgb.shape[:2], bool), 'sphere', .9)], INFO))
        server = make_server(model, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = VisionClient(V2Config(endpoint=f'http://127.0.0.1:{server.server_port}'))
            self.assertEqual(client.health()['sha256'], INFO['sha256'])
            result = client.predict(10, np.zeros((9, 15, 3), np.uint8))
            self.assertEqual(result.instances[0].mask.shape, (9, 15))
            self.assertEqual(result.instances[0].label, 'sphere')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_connection_failure_has_actionable_message(self):
        client = VisionClient(V2Config(endpoint='http://127.0.0.1:1', timeout_s=.1))
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            client.health()

    def test_yolo_adapter_converts_rgb_to_bgr_requests_native_masks(self):
        class Tensor:
            def __init__(self, a): self.a = np.asarray(a)
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.a
        class Boxes:
            cls, conf = Tensor([1]), Tensor([.9])
            def __len__(self): return 1
        saved = {}
        def predict(image, **kwargs):
            saved.update(image=image, **kwargs)
            return [SimpleNamespace(boxes=Boxes(), masks=SimpleNamespace(data=Tensor(np.ones((1, 10, 20)))))]
        adapter = YoloSegmenter.__new__(YoloSegmenter)
        adapter.model, adapter.names, adapter.info = SimpleNamespace(predict=predict), {1: 'sphere'}, INFO
        adapter.device, adapter.image_size = 'cpu', 640
        rgb = np.zeros((10, 20, 3), np.uint8)
        rgb[..., 0] = 255
        result = adapter.predict(1, rgb)
        self.assertEqual(saved['image'][0, 0].tolist(), [0, 0, 255])
        self.assertTrue(saved['retina_masks'])
        self.assertFalse(saved['half'])
        self.assertEqual(result.instances[0].label, 'sphere')

    def test_checkpoint_task_and_class_validation(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), set_num_threads=lambda n: None,
                                __version__='test')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            path.write_bytes(b'test double checkpoint, not a real model')
            for task, names, valid in [('detect', {0: 'cube', 1: 'sphere', 2: 'cylinder'}, False),
                                       ('segment', {0: 'person'}, False),
                                       ('segment', {0: 'cylinder', 1: 'cube', 2: 'sphere'}, True)]:
                model = SimpleNamespace(task=task, names=names)
                ultra = SimpleNamespace(__version__='test', YOLO=lambda *a: model)
                with patch.dict('sys.modules', {'torch': torch, 'ultralytics': ultra}):
                    if valid:
                        adapter = YoloSegmenter(path)
                        self.assertEqual(adapter.names[0], 'cylinder')
                    else:
                        with self.assertRaises(ValueError):
                            YoloSegmenter(path)

    def test_http_model_error_propagates(self):
        from tools.perception_v2.serve import make_server
        def fail(*a, **kw): raise RuntimeError('fixture inference failure')
        model = SimpleNamespace(info=INFO, predict=fail)
        server = make_server(model, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = VisionClient(V2Config(endpoint=f'http://127.0.0.1:{server.server_port}'))
            with self.assertRaisesRegex(RuntimeError, 'fixture inference failure'):
                client.predict(1, np.zeros((4, 4, 3), np.uint8))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_missing_checkpoint_never_downloads_fallback_model(self):
        with self.assertRaisesRegex(ValueError, 'trusted trained'):
            YoloSegmenter('/definitely/missing/best.pt')


if __name__ == '__main__':
    unittest.main(verbosity=2)
