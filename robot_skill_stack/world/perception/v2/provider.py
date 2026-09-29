"""Nonblocking, one-in-flight inference. Only the simulation thread touches the tracker."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.primitives import PrimitiveShape
from robot_skill_stack.world.perception.frame import PerceptionFrame
from robot_skill_stack.world.perception.geometry import ObjectGeometryService
from robot_skill_stack.world.perception.localizer import RgbdLocalizer
from robot_skill_stack.world.perception.v2.types import validate_frame
from robot_skill_stack.world.perception.v2 import wire


class YoloPerceptionProvider:
    source = 'rgbd_yolo_v2'

    def __init__(self, camera, processor, tracker, client, *, clock=time.monotonic, executor=None, robot_source=None):
        self.camera, self.processor, self.tracker, self.client = camera, processor, tracker, client
        self.config, self.clock = processor.config, clock
        self.robot_source = robot_source
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix='rgbd-v2')
        self.geometry = ObjectGeometryService(RgbdLocalizer(camera.get_intrinsics()), tracker)
        self._future, self._frame_id, self._last_token = None, 0, None
        self._next_request, self._closed = 0., False
        self.last_error, self.status, self.diagnostics = None, 'waiting for RGB-D', []
        self.model_info, self._last_result = {}, None
        self.processed_frames = self.stale_results = 0
        self._last_completed = None
        self._request_started = None
        self.submitted_frames = 0

    def _work(self, frame, robot_snapshot):
        segmentation = self.client.predict(frame.frame_id, frame.rgb)
        return self.processor.process(frame, segmentation, robot_snapshot)

    def _snapshot(self, now):
        token_fn = getattr(self.camera, 'get_frame_token', None)
        token = None if token_fn is None else token_fn()
        if token is None:
            self.status = 'waiting for a timestamped camera frame'
            return None
        if token == self._last_token:
            return None
        rgb, depth = self.camera.get_rgb(), self.camera.get_depth()
        if rgb is None or depth is None:
            return None
        T, K = self.camera.get_world_from_camera_transform(), self.camera.get_intrinsics()
        if token_fn() != token:
            return None
        robot_snapshot = None
        if self.config.robot_self_filter:
            if self.robot_source is None:
                raise RuntimeError('Robot self-filter is enabled but its capture adapter is missing')
            robot_snapshot = self.robot_source.snapshot(token)
        self._frame_id += 1
        frame = PerceptionFrame(self._frame_id, np.asarray(rgb)[..., :3].copy(), np.asarray(depth).copy(),
                                np.asarray(K).copy(), np.asarray(T).copy(), now)
        validate_frame(frame)
        self._last_token = token
        return frame, robot_snapshot

    def _collect(self, now, context):
        future = self._future
        if future is None or not future.done():
            return
        self._future = None
        try:
            result = future.result()
            frame = result.frame
            self._last_completed = result  # Diagnostic only until it passes the freshness gate.
            if self.model_info and result.segmentation.model.get('sha256') != self.model_info.get('sha256'):
                raise RuntimeError('Worker weights changed. Restart/reinitialize the V2 runtime before continuing.')
            if now-frame.timestamp > self.config.max_result_age_s:
                self.stale_results += 1
                self.last_error = (f'Vision result too old ({now-frame.timestamp:.2f}s); discarded. '
                                   f'Robot filter: {result.self_filter.metadata.get("render_s", 0.):.2f}s; '
                                   f'all geometry: {result.geometry_s:.2f}s. Check load before changing age limits.')
                self.status = 'stale'
                return
            self.tracker.update(result.candidates, result.predictions, frame_id=frame.frame_id,
                                timestamp=frame.timestamp, context=context)
            self.geometry.localizer = RgbdLocalizer(frame.intrinsics)
            self.geometry.update_frame(frame)
            self._last_result, self.diagnostics, self.model_info = result, result.diagnostics, result.segmentation.model
            self.processed_frames += 1
            self.last_error, self.status = None, 'ready'
        except Exception as exc:
            self.last_error = f'{type(exc).__name__}: {exc}'
            self.status = 'worker error'

    def observe(self, context=None):
        if self._closed:
            return []
        now = self.clock()
        self._collect(now, context)
        self.age(now, context)
        if self._future is None and now >= self._next_request:
            try:
                frame = self._snapshot(now)
                if frame is not None:
                    self._future = self.executor.submit(self._work, *frame)
                    self._request_started = now
                    self.submitted_frames += 1
                    self._next_request = now+1/self.config.request_hz
                    if self._last_completed is None and self.last_error is None:
                        self.status = 'processing first frame'
            except Exception as exc:
                self.last_error, self.status = str(exc), 'camera error'
                self._next_request = now+1/self.config.request_hz
        return self._observations()

    def _observations(self):
        result = []
        for track in self.tracker.tracks():
            if not track.confirmed:
                continue
            g = track.geometry
            known = g is not None and g.shape != PrimitiveShape.UNKNOWN
            yaw = None if g is None else g.yaw
            orientation = None if yaw is None else np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
            result.append(ObjectObservation(
                track.object_id, pose=Pose(track.position.copy(), orientation), class_name=track.class_name,
                size=track.size.copy(), geometry=g, graspable=known, visible=track.visible,
                confidence=track.confidence, source=self.source, timestamp=track.last_seen,
                metadata=dict(class_belief_authoritative=True, class_confidence=track.class_confidence,
                              geometry_confidence=0. if g is None else g.confidence,
                              position_source='primitive_fit' if known else 'visible_surface_bounds',
                              track_hits=track.hits, track_misses=track.misses,
                              association_hint_match=track.matched_by_hint,
                              association_hint_reason=track.association_hint_reason,
                              observed_at=track.last_seen, pose_predicted=False,
                              class_validation=None if g is None else g.metadata.get('class_validation')),
            ))
        return result

    def age(self, now=None, context=None):
        now = self.clock() if now is None else now
        if (self.status == 'ready' and self._last_result is not None
                and now-self._last_result.frame.timestamp >= self.config.stale_frame_s):
            self.status, self.last_error = 'stale', 'No fresh accepted RGB-D observation; tracks are aging'
        return self.tracker.age(now, self.config.stale_frame_s, context)

    def forget(self, object_id):
        return self.tracker.forget(object_id)

    def get_mask(self, object_id):
        return self.geometry.get_mask(object_id)

    def get_point_cloud(self, object_id):
        return self.geometry.get_point_cloud(object_id)

    def close(self):
        if not self._closed:
            self._closed = True
            if self._future is not None:
                self._future.cancel()
            self.executor.shutdown(wait=False, cancel_futures=True)
            if self.robot_source is not None:
                self.robot_source.close()

    def diagnostics_snapshot(self):
        result = self._last_completed
        robot = getattr(self.robot_source, 'diagnostics_snapshot', None)
        return dict(status=self.status, error=self.last_error,
                    accepted_frames=self.processed_frames, submitted_frames=self.submitted_frames,
                    stale_results=self.stale_results,
                    in_flight=self._future is not None and not self._future.done(),
                    request_age_s=None if self._future is None or self._request_started is None else
                                  max(0., self.clock()-self._request_started),
                    candidate_count=None if result is None else len(result.candidates),
                    tracks=len(self.tracker.tracks()),
                    confirmed_tracks=sum(t.confirmed for t in self.tracker.tracks()),
                    robot_filter_s=None if result is None or result.self_filter is None else
                                   result.self_filter.metadata.get('render_s'),
                    geometry_s=None if result is None else result.geometry_s,
                    last_frame_token=str(self._last_token),
                    robot_snapshot=None if robot is None else robot())

    def save_capture(self, path):
        result = self._last_completed if self._last_completed is not None else self._last_result
        if result is None:
            raise ValueError('No completed V2 observation: '+str(self.last_error or self.status))
        frame = result.frame
        payload = wire.encode_response(result.segmentation, frame.depth.shape)
        settings = dict(version=2, filter_revision=2, source=self.source, v2=asdict(self.config),
                        observation_accepted=result is self._last_result, provider_status=self.diagnostics_snapshot(),
                        robot_self_filter=result.self_filter.metadata,
                        tracks_at_save=[dict(object_id=t.object_id, class_name=t.class_name, confirmed=t.confirmed,
                                             visible=t.visible, last_seen=t.last_seen, position=t.position.tolist(),
                                             size=t.size.tolist()) for t in self.tracker.tracks()],
                        discovery={k: v.tolist() if isinstance(v, np.ndarray) else v
                                   for k, v in vars(self.processor.discovery).items()}, candidates=result.diagnostics,
                        primitives=dict(classification_threshold=self.processor.estimator.threshold,
                                        ambiguity_margin=self.processor.estimator.margin,
                                        top_band=self.processor.estimator.top_band,
                                        fit_tolerance=self.processor.estimator.tolerance))
        path = Path(path)
        if path.suffix != '.npz':
            raise ValueError('Capture filename must end in .npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, rgb=frame.rgb, depth=frame.depth, intrinsics=frame.intrinsics,
                            world_from_camera=frame.world_from_camera, timestamp=frame.timestamp,
                            frame_id=frame.frame_id, segmentation=np.frombuffer(payload, np.uint8),
                            robot_mask=result.self_filter.mask, robot_model_depth=result.self_filter.model_depth,
                            settings=json.dumps(settings, allow_nan=False))
        return path
