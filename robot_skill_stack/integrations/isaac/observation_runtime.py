"""Kit-owned acquisition/publication with bounded SDK-free computation.

No geometry processing, worker wait, or dense map read is performed by this
subscription. Ground-truth semantic providers retain their lightweight updater.
"""
from dataclasses import replace
import threading
import time

import omni.usd

from robot_skill_stack.runtime.worker import BoundedWorker
from robot_skill_stack.world.perception_multiview.shared_frames import SharedFramePool


def semantic_camera_paths(config):
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
    def __init__(self, world, config, updater, *, pool=None, owned_cameras=(),
                 robot_filter=None, spatial_mapping=None, stage=None, clock=time.monotonic,
                 worker_factory=None):
        self.world, self.config, self.updater = world, config, updater
        self.provider, self.pool = updater.provider, pool
        self._owned_cameras, self.robot_filter = tuple(owned_cameras), robot_filter
        self.spatial_mapping, self.stage, self.clock = spatial_mapping, stage, clock
        self._owner = threading.get_ident()
        self._subscription, self._timeline_subscription = None, None
        self._closed, self._playing, self._generation = False, bool(world.is_playing()), 0
        self._semantic_enabled = callable(getattr(self.provider, 'acquire', None))
        self._mapping_enabled = spatial_mapping is not None
        self._semantic_worker = ((worker_factory or BoundedWorker)(self.provider.processor.compute,
            name='semantic-observations') if self._semantic_enabled else None)
        self._semantic_due = self._mapping_due = 0.
        self._semantic_period = 1/(config.world.update_hz or 5.)
        self._mapping_settings = mapping_settings(config)
        self._mapping_period = 1/(self._mapping_settings.update_hz or 2.)
        self._semantic = dict(attempts=0, published=0, no_frame=0, rejected=0,
                              source_failures=0, worker_failures=0, last=None)
        self._state, self._error = 'running', None
        self._last_simulation_time = None
        self._simulation_time = None
        self._filter_needs_resume = False
        self._mapping_invalidation_error = None

    @property
    def diagnostics(self):
        semantic = dict(self._semantic)
        semantic['worker'] = None if self._semantic_worker is None else self._semantic_worker.health
        return dict(state=self._state, generation=self._generation, error=self._error,
            semantic=semantic, mapping=None if self.spatial_mapping is None else dict(self.spatial_mapping.health),
            cameras=None if self.pool is None else dict(acquisitions=self.pool.acquisitions,
                shared_reads=self.pool.shared_reads, failures=self.pool.failures,
                consumers=dict(self.pool.status)))

    def _assert_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('Observation runtime must be accessed on its Kit owning thread')

    def set_enabled(self, *, semantic, mapping):
        """Diagnostic scheduling gate; retains cameras and render products."""
        self._assert_owner()
        if self._semantic_enabled != bool(semantic) and self._semantic_worker is not None:
            self._semantic_worker.invalidate()
        if self._mapping_enabled != bool(mapping) and self.spatial_mapping is not None:
            self.spatial_mapping.reset()
        self._semantic_enabled = bool(semantic) and self._semantic_worker is not None
        self._mapping_enabled = bool(mapping) and self.spatial_mapping is not None
        self._semantic_due = self._mapping_due = 0.
        self._mapping_invalidation_error = None

    def start(self, app):
        self._assert_owner()
        self._subscription = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update, name='robot_skill_stack_observations')
        # Stop and Play may both occur between two Kit updates. The timeline
        # barrier catches that case, while update-time polling supports tests
        # and hosts without a timeline event stream.
        try:
            import omni.timeline
            self._timeline_subscription = omni.timeline.get_timeline_interface().get_timeline_event_stream().create_subscription_to_pop(
                self._on_timeline, name='robot_skill_stack_observation_lifecycle')
        except ImportError:
            pass

    def _on_timeline(self, event):
        if self._closed:
            return
        import omni.timeline
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._transition(False)
        elif event.type == int(omni.timeline.TimelineEventType.PLAY):
            self._transition(True)

    def _transition(self, playing):
        if playing == self._playing:
            return
        self._playing = playing
        self._generation += 1
        self._last_simulation_time = None
        self._semantic_due = self._mapping_due = 0.
        self._mapping_invalidation_error = None
        if self._semantic_worker is not None:
            self._semantic_worker.invalidate()
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
        self._state = 'running' if playing else 'stopped'

    def _on_update(self, event=None):
        if self._closed:
            return
        try:
            self.tick()
        except Exception as exc:
            self._state, self._error = 'source_failure', f'{type(exc).__name__}: {exc}'
            if self._semantic_worker is not None:
                self._semantic_worker.invalidate()
                self.updater.note_failure(exc)
            if self.spatial_mapping is not None:
                self.spatial_mapping.note_source_failure(exc)

    def tick(self):
        self._assert_owner()
        if self._closed:
            return
        current_stage = omni.usd.get_context().get_stage()
        if self.stage is not None and current_stage != self.stage:
            raise RuntimeError('Active stage changed; reinitialize the runtime')
        self._transition(bool(self.world.is_playing()))
        if not self._playing:
            return
        simulation_time = getattr(self.world, 'current_time', None)
        self._simulation_time = simulation_time if isinstance(simulation_time, (int, float)) else None
        if isinstance(simulation_time, (int, float)):
            if self._last_simulation_time is not None and simulation_time < self._last_simulation_time:
                self._transition(False)
                self._transition(True)
            self._last_simulation_time = simulation_time
        now = self.clock()
        self.updater.cleanup(now)
        # Admission errors below install a barrier BEFORE an older completed
        # success is polled, so it cannot clear the newly reported failure.
        if self.pool is not None:
            self.pool.begin_update()
        mapping_source_ready = True
        if self._mapping_enabled and self.robot_filter is not None and not self._filter_needs_resume:
            error = self.robot_filter.source_error
            if error is not None:
                mapping_source_ready = False
                if error != self._mapping_invalidation_error:
                    self.spatial_mapping.note_source_failure(error)
                    self._mapping_invalidation_error = error
            else:
                self._mapping_invalidation_error = None
        semantic_due = self._semantic_enabled and now >= self._semantic_due
        if semantic_due:
            self._semantic_due = self._deadline(self._semantic_due, now, self._semantic_period)
            self._acquire_semantic()
        overlap = (self._semantic_enabled and self._mapping_settings.source == 'rgbd' and
                   bool(set(semantic_camera_paths(self.config)) &
                        set(self._mapping_settings.camera_prim_paths)))
        # A due map sharing semantic cameras uses the next semantic acquisition
        # (at most one semantic period later), so common cameras retain the exact
        # same buffers/exposure. Independent cameras and rates stay independent.
        if (self._mapping_enabled and mapping_source_ready and now >= self._mapping_due and
                (not overlap or semantic_due)):
            self._mapping_due = self._deadline(self._mapping_due, now, self._mapping_period)
            self._acquire_mapping()
        self._publish_semantic()
        if self._mapping_enabled and mapping_source_ready:
            self.spatial_mapping.poll(now=self.clock())

    @staticmethod
    def _deadline(previous, now, period):
        # No catch-up burst after a long pause, and no accumulated cadence drift.
        return now+period if previous <= 0 or now-previous >= period else previous+period

    def _acquire_semantic(self):
        self._semantic['attempts'] += 1
        try:
            paths = semantic_camera_paths(self.config)
            skew = self.config.multi_view.max_skew if self.config.world.provider == 'multi_view' else .025
            frames = self.pool.batch('semantic', paths, max_skew=skew,
                                     simulation_time=self._simulation_time)
            if frames is None:
                self._semantic['no_frame'] += 1
                return
            if self.config.world.provider == 'multi_view':
                self.provider.rig.status = dict(self.pool.status['semantic'])
            prepared = self.provider.acquire(frames=frames)
            if prepared is None:
                self._semantic['no_frame'] += 1
                return
            revision = self.updater.world_model.observation_context().revision
            self._semantic_worker.submit(prepared, tag=(self._generation, revision))
        except Exception as exc:
            self._semantic['source_failures'] += 1
            self._semantic_worker.invalidate()
            self.updater.note_failure(exc)

    def _publish_semantic(self):
        if not self._semantic_enabled:
            return
        completed = self._semantic_worker.poll()
        if completed is None:
            return
        if completed.error is not None:
            self._semantic['worker_failures'] += 1
            self.updater.note_failure(completed.error)
            return
        generation, revision = completed.tag
        result = completed.value
        compute_seconds = getattr(result, 'processing_ms', None)
        diagnostics = getattr(result, 'diagnostics', {})
        if compute_seconds is None and isinstance(diagnostics, dict):
            compute_seconds = diagnostics.get('processing_ms')
        compute_seconds = None if compute_seconds is None else compute_seconds/1000
        last = dict(timestamp=result.timestamp, input_timestamp=result.timestamp,
            completed_at=completed.completed_at,
            worker_seconds=completed.duration, queue_seconds=completed.started_at-completed.queued_at,
            compute_seconds=compute_seconds,
            transport_seconds=None if compute_seconds is None else max(0., completed.duration-compute_seconds),
            publish_age_seconds=self.clock()-result.timestamp, status='rejected')
        self._semantic['last'] = last
        # No awaits between the context guard, tracker association and publish.
        if generation != self._generation or not self.updater.accepts(result.timestamp, revision):
            self._semantic['rejected'] += 1
            return
        before = self.updater.update_count
        try:
            batch = self.provider.associate(result, context=self.updater.world_model.observation_context())
            self.updater.publish(batch, context_revision=revision)
        except Exception as exc:
            self.updater.note_failure(exc)
            self._semantic['worker_failures'] += 1
            return
        if self.updater.update_count > before:
            self._semantic['published'] += 1
            last['status'] = 'published'

    def _acquire_mapping(self):
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

    def close(self):
        self._assert_owner()
        if self._closed:
            return
        self._closed, self._state = True, 'closed'
        self._subscription = self._timeline_subscription = None
        errors = []
        resources = [self._semantic_worker, self.spatial_mapping, self.robot_filter,
                     self.pool, *reversed(self._owned_cameras)]
        for resource in resources:
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise RuntimeError('Observation cleanup failed: '+ '; '.join(map(str, errors))) from errors[0]


async def create_observation_runtime(world, config, updater, app):
    """Create missing mapping cameras only; borrow existing semantic sources."""
    settings = mapping_settings(config)
    provider, cameras, owned = updater.provider, {}, []
    semantic = callable(getattr(provider, 'acquire', None))
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
            service = MappingService(settings, worker_factory=ProcessWorker)
            if settings.source == 'rgbd' and settings.automatic_robot_filter:
                from .spatial_mapping.robot_filter import IsaacRobotSelfFilter
                robot_filter = IsaacRobotSelfFilter(stage, config.robot.prim_path,
                    stride=settings.stride, depth_tolerance=settings.robot_depth_tolerance,
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
