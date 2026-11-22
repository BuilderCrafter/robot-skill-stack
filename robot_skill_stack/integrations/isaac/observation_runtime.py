"""Kit-owned acquisition and publication through SDK-independent observation workers."""
from dataclasses import replace
import threading
import time

import omni.usd

from robot_skill_stack.runtime.worker import BoundedWorker
from robot_skill_stack.world.perception_multiview.shared_frames import SharedFramePool


def semantic_camera_paths(config) -> tuple[str, ...]:
    if config.world.provider == 'multi_view':
        return tuple(config.multi_view.camera_prim_paths)
    if config.world.provider == 'perception':
        return (config.perception.camera_prim_path,)
    return ()


def mapping_settings(config):
    settings = config.spatial_mapping
    if settings.enabled and settings.source == 'rgbd' and not settings.camera_prim_paths:
        paths = semantic_camera_paths(config)
        if not paths:
            raise ValueError('RGB-D mapping with ground-truth semantics requires camera_prim_paths')
        settings = replace(settings, camera_prim_paths=paths)
    return settings


class ObservationRuntime:
    """Coordinate camera observations; GT semantics retain their physics updater."""
    def __init__(self, world, config, updater, *, pool=None, owned_cameras=(),
                 robot_filter=None, spatial_mapping=None, stage=None, clock=time.monotonic,
                 worker_factory=BoundedWorker):
        self.world, self.updater = world, updater
        self.pool, self.robot_filter = pool, robot_filter
        self.spatial_mapping, self.stage, self.clock = spatial_mapping, stage, clock
        self._owned_cameras = tuple(owned_cameras)
        self._owner = threading.get_ident()
        self._subscription = self._timeline_subscription = None
        self._closed, self._playing = False, world.is_playing()
        self._simulation_time = None
        self._filter_needs_resume = False
        self._semantic_paths = semantic_camera_paths(config)
        self._multiview = config.world.provider == 'multi_view'
        self._semantic_skew = config.multi_view.max_skew if self._multiview else .025
        self._mapping_settings = mapping_settings(config)
        self._shared_cameras = (self._mapping_settings.source == 'rgbd' and
            bool(set(self._semantic_paths) & set(self._mapping_settings.camera_prim_paths)))
        self._semantic_due = self._mapping_due = 0.
        self._semantic_period = 1/(config.world.update_hz or 5.)
        self._mapping_period = 1/(self._mapping_settings.update_hz or 2.)
        self._semantic_worker = (worker_factory(updater.provider.processor.compute,
            name='semantic-observations') if self._semantic_paths else None)

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError('Observation runtime must be accessed on its Kit owning thread')

    def start(self, app) -> None:
        self._assert_owner()
        import omni.timeline
        self._subscription = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update, name='robot_skill_stack_observations')
        # Catch Stop and Play even when both occur between Kit updates.
        self._timeline_subscription = omni.timeline.get_timeline_interface().get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline, name='robot_skill_stack_observation_lifecycle')

    def _on_timeline(self, event) -> None:
        if self._closed:
            return
        import omni.timeline
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._transition(False)
        elif event.type == int(omni.timeline.TimelineEventType.PLAY):
            self._transition(True)

    def _transition(self, playing: bool) -> None:
        if playing == self._playing:
            return
        self._playing = playing
        self._simulation_time = None
        self._semantic_due = self._mapping_due = 0.
        if self._semantic_worker is not None:
            self._semantic_worker.invalidate()  # The worker owns the timeline epoch.
        if self.spatial_mapping is not None:
            self.spatial_mapping.reset()
        if self.pool is not None:
            self.pool.reset()
            if playing:
                for camera in self.pool.cameras.values():
                    camera.reset_clock()
        if self.robot_filter is not None:
            self.robot_filter.reset_history()
            self._filter_needs_resume = playing
        self.updater.note_failure('Timeline restarted; waiting for fresh observations' if playing else 'Simulation stopped')

    def _on_update(self, event=None) -> None:
        if self._closed:
            return
        try:
            self.tick()
        except Exception as exc:
            if self._semantic_worker is not None:
                self._semantic_worker.invalidate()
                self.updater.note_failure(exc)
            if self.spatial_mapping is not None:
                self.spatial_mapping.note_source_failure(exc)

    def tick(self) -> None:
        """Acquire due inputs before polling, so source failures invalidate older work."""
        self._assert_owner()
        if self._closed:
            return
        current_stage = omni.usd.get_context().get_stage()
        if self.stage is not None and current_stage != self.stage:
            raise RuntimeError('Active stage changed; reinitialize the runtime')
        self._transition(bool(self.world.is_playing()))
        if not self._playing:
            return
        simulation_time = self.world.current_time
        if self._simulation_time is not None and simulation_time < self._simulation_time:
            self._transition(False)
            self._transition(True)
        self._simulation_time = simulation_time
        now = self.clock()
        self.updater.cleanup(now)
        # Admission errors below install a barrier BEFORE an older completed
        # success is polled, so it cannot clear the newly reported failure.
        if self.pool is not None:
            self.pool.begin_update()
        mapping_ready = self.spatial_mapping is not None
        if mapping_ready and self.robot_filter is not None and not self._filter_needs_resume:
            error = self.robot_filter.source_error
            if error is not None:
                self.spatial_mapping.note_source_failure(error)
                mapping_ready = False
        semantic_due = self._semantic_worker is not None and now >= self._semantic_due
        if semantic_due:
            self._semantic_due = self._deadline(self._semantic_due, now, self._semantic_period)
            self._acquire_semantic()
        # Shared cameras use the same acquisition on a semantic update.
        if (mapping_ready and now >= self._mapping_due and
                (not self._shared_cameras or semantic_due)):
            self._mapping_due = self._deadline(self._mapping_due, now, self._mapping_period)
            self._acquire_mapping()
        self._publish_semantic()
        if mapping_ready:
            self.spatial_mapping.poll(now=self.clock())

    @staticmethod
    def _deadline(previous, now, period):
        # No catch-up burst after a long pause, and no accumulated cadence drift.
        return now+period if previous <= 0 or now-previous >= period else previous+period

    def _acquire_semantic(self) -> None:
        try:
            frames = self.pool.batch('semantic', self._semantic_paths, max_skew=self._semantic_skew,
                                     simulation_time=self._simulation_time)
            if frames is None:
                return
            if self._multiview:
                self.updater.provider.rig.status = dict(self.pool.status['semantic'])
            prepared = self.updater.provider.acquire(frames=frames)
            if prepared is not None:
                revision = self.updater.world_model.observation_context().revision
                self._semantic_worker.submit(prepared, tag=revision)
        except Exception as exc:
            self._semantic_worker.invalidate()
            self.updater.note_failure(exc)

    def _publish_semantic(self) -> None:
        if self._semantic_worker is None:
            return
        completed = self._semantic_worker.poll()
        if completed is None:
            return
        if completed.error is not None:
            self.updater.note_failure(completed.error)
            return
        revision, result = completed.tag, completed.value
        if not self.updater.accepts(result.timestamp, revision):
            return
        # Context admission, association and publication stay on the owner, with no awaits.
        try:
            batch = self.updater.provider.associate(result, context=self.updater.world_model.observation_context())
            self.updater.publish(batch, context_revision=revision)
        except Exception as exc:
            self.updater.note_failure(exc)

    def _acquire_mapping(self) -> None:
        from robot_skill_stack.world.spatial_mapping.runtime import MappingInput
        service = self.spatial_mapping
        try:
            if self._mapping_settings.source == 'ground_truth':
                from .spatial_mapping.ground_truth import prepare_ground_truth
                prepared = prepare_ground_truth(self.stage,
                    excluded_paths=self._mapping_settings.excluded_prim_paths, clock=self.clock)
                payload = MappingInput(generation=service.generation, ground_truth=prepared)
            else:
                frames = self.pool.batch('mapping', self._mapping_settings.camera_prim_paths,
                    max_skew=self._mapping_settings.max_skew, simulation_time=self._simulation_time)
                if frames is None:
                    service.note_no_frame()
                    return
                if self.robot_filter is not None and self._filter_needs_resume:
                    self.robot_filter.rebind_after_play()
                    self._filter_needs_resume = False
                snapshots = None if self.robot_filter is None else self.robot_filter.prepare(frames)
                payload = MappingInput(generation=service.generation, frames=frames,
                                       robot_snapshots=snapshots)
            service.submit(payload)
        except Exception as exc:
            service.note_source_failure(exc)

    def close(self) -> None:
        self._assert_owner()
        if self._closed:
            return
        self._closed = True
        self._subscription = self._timeline_subscription = None
        failure = None
        resources = [self._semantic_worker, self.spatial_mapping, self.robot_filter,
                     self.pool, *reversed(self._owned_cameras)]
        for resource in resources:
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    failure = failure or exc
        if failure is not None:
            raise failure


async def create_observation_runtime(world, config, updater, app):
    """Create missing mapping cameras only; borrow existing semantic sources."""
    settings = mapping_settings(config)
    provider, cameras, owned = updater.provider, {}, []
    semantic = bool(semantic_camera_paths(config))
    if not semantic and not settings.enabled:
        return None
    pool = robot_filter = service = None
    try:
        from robot_skill_stack.runtime.process_worker import ProcessWorker
        if semantic:
            cameras.update(provider.rig.cameras if config.world.provider == 'multi_view'
                           else {config.perception.camera_prim_path: provider.camera})
        if settings.enabled and settings.source == 'rgbd':
            from .world.multiview import IsaacMultiViewCamera
            for path in settings.camera_prim_paths:
                if path in cameras:
                    continue
                camera = IsaacMultiViewCamera(path, name=f'mapping_camera_{len(owned)}')
                owned.append(camera)
                camera.initialize(semantic_segmentation=False)
                cameras[path] = camera
            if owned:
                for _ in range(60):
                    await app.next_update_async()
        if cameras:
            from .world.multiview import IsaacMultiViewCamera
            pool = SharedFramePool(cameras, token_reader=IsaacMultiViewCamera.get_frame_token)
        stage = omni.usd.get_context().get_stage()
        if settings.enabled:
            from robot_skill_stack.world.spatial_mapping.runtime import MappingService
            motion, options = config.motion, {}
            if motion.strategy == 'global' and motion.global_obstacle_source == 'frozen_rgbd':
                from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractionSettings
                options['obstacle_settings'] = ObstacleExtractionSettings(max_obstacles=motion.max_obstacles,
                    support_plane_z=config.scene.support_plane_z)
                if motion.global_rgbd_representation == 'reconstructed':
                    options['reconstruction_scene'] = config.scene
            service = MappingService(settings, worker_factory=ProcessWorker, **options)
            if settings.source == 'rgbd' and settings.automatic_robot_filter:
                from .spatial_mapping.robot_filter import IsaacRobotSelfFilter
                robot_filter = IsaacRobotSelfFilter(stage, config.robot.prim_path,
                    pose_tolerance=settings.robot_pose_tolerance)
        return ObservationRuntime(world, config, updater, pool=pool, owned_cameras=owned,
            robot_filter=robot_filter, spatial_mapping=service, stage=stage,
            worker_factory=ProcessWorker)
    except BaseException as exc:
        for resource in [service, robot_filter, pool, *reversed(owned)]:
            if resource is not None:
                try:
                    resource.close()
                except Exception as cleanup:
                    exc.add_note(f'Observation resource cleanup also failed: {cleanup}')
        raise
