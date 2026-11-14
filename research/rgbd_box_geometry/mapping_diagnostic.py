"""One partial-scene mapping diagnosis; no reconstruction, planning or GOAL."""
import argparse
from collections import deque
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from scripts import benchmark_global_rgbd_v5 as bench
from . import rrt_bridge as bridge


class DiagnosticStop(Exception):
    """Intentional exit before the benchmark's selected motion request."""


class Timeline:
    def __init__(self, clock=time.monotonic):
        self.clock, self.runtime, self.phase = clock, None, 'bootstrap'
        self.state = {}
        self.samples, self.events = deque(maxlen=128), deque(maxlen=64)
        self.ticks, self.sampled, self.event_count, self.next_sample = 0, 0, 0, 0.
        self.outcome = 'not_reached'

    def record(self, event, *, force=False, **extra):
        try:
            self._record(event, force=force, **extra)
        except Exception as exc:
            self.event_count += 1
            self.events.append(dict(event='instrumentation_error', phase=self.phase,
                                    observed_event=event, error=repr(exc)))

    def _record(self, event, *, force=False, **extra):
        now = self.clock()
        if not force and now < self.next_sample:
            return
        phase = self.state.get('acquisition_phase', 'setup') if self.phase == 'bootstrap' else self.phase
        row = dict(event=event, phase=phase, monotonic=now, ticks=self.ticks,
                   thread=threading.get_ident(), **extra)
        runtime = self.runtime
        if runtime is not None and threading.get_ident() != runtime._owner:
            row['owner_thread'] = False
        elif runtime is not None:
            filt, source = runtime.robot_filter, runtime.spatial_mapping
            error = None if filt is None else filt.source_error
            paths = (runtime.config.multi_view.camera_prim_paths
                     if runtime.config.world.provider == 'multi_view' else
                     (runtime.config.perception.camera_prim_path,)
                     if runtime.config.world.provider == 'perception' else ())
            overlap = (runtime._semantic_enabled and runtime._mapping_settings.source == 'rgbd'
                       and bool(set(paths) & set(runtime._mapping_settings.camera_prim_paths)))
            source_ready = not (runtime._mapping_enabled and filt is not None
                                and not runtime._filter_needs_resume and error is not None)
            row.update(simulation_time=runtime.world.current_time,
                physics_step=runtime.world.current_time_step_index,
                owner_thread=threading.get_ident() == runtime._owner,
                runtime_clock=runtime.clock(), world_playing=bool(runtime.world.is_playing()),
                playing=runtime._playing, closed=runtime._closed, state=runtime._state,
                runtime_error=runtime._error, generation=runtime._generation,
                semantic_enabled=runtime._semantic_enabled, mapping_enabled=runtime._mapping_enabled,
                semantic_deadline=runtime._semantic_due, mapping_deadline=runtime._mapping_due,
                semantic_due=runtime._semantic_enabled and now >= runtime._semantic_due,
                mapping_due=now >= runtime._mapping_due, overlap=overlap,
                mapping_source_ready=source_ready, filter_needs_resume=runtime._filter_needs_resume,
                mapping_invalidation_error=runtime._mapping_invalidation_error,
                robot_filter=None if filt is None else dict(source_error=error,
                    model_error=filt._model_error, capture_error=filt._capture_error,
                    closed=filt._closed, history_generation=filt._generation,
                    history_count=len(filt._history), pose_capture_calls=filt._pose_capture_calls,
                    diagnostics=dict(filt.diagnostics)),
                mapping_health=None if source is None else dict(source.health),
                semantic=dict(runtime._semantic))
        if force:
            self.event_count += 1
            self.events.append(row)
        else:
            self.sampled += 1
            self.samples.append(row)
            self.next_sample = now + .25

    @contextmanager
    def installed(self, runtime_type, filter_type):
        tick, changed = runtime_type.tick, filter_type._model_changed
        observer, physical = bench.controlled.make_observer, bench.acceptance.fresh_physical_capture
        camera_timing = bench.install_camera_timing

        def cameras(pool, state):
            self.state = state
            return camera_timing(pool, state)

        def traced_tick(runtime):
            self.runtime = runtime
            self.ticks += 1
            try:
                return tick(runtime)
            finally:
                self.record('observation_tick')

        def traced_change(filt, notice, sender):
            before = filt.source_error
            try:
                return changed(filt, notice, sender)
            finally:
                if filt.source_error != before:
                    self.record('robot_filter_notice', force=True, source_error=filt.source_error,
                        resynced=[str(p) for p in notice.GetResyncedPaths()][:20],
                        changed_info=[str(p) for p in notice.GetChangedInfoOnlyPaths()][:20])

        def step(name, function, bundle):
            self.runtime, self.phase = bundle.observation_runtime, name
            self.record('before_'+name, force=True)
            try:
                return function(bundle)
            finally:
                self.record('after_'+name, force=True)

        with ExitStack() as stack:
            for obj, name, replacement in (
                (runtime_type, 'tick', traced_tick), (filter_type, '_model_changed', traced_change),
                (bench, 'install_camera_timing', cameras),
                (bench.controlled, 'make_observer', lambda b:step('observer', observer, b)),
                (bench.acceptance, 'fresh_physical_capture', lambda b:step('physical_capture', physical, b))):
                stack.enter_context(patch.object(obj, name, replacement))
            yield

    async def ready(self, bundle, state):
        import omni.kit.app
        self.runtime, self.phase = bundle.observation_runtime, 'readiness'
        source = bundle.spatial_mapping
        first, started = source.health['published'], self.clock()
        self.record('readiness_start', force=True)
        try:
            while self.clock()-started < 5.:
                await omni.kit.app.get_app().next_update_async()
                if self.clock()-started >= 5.:
                    break
                health = bench.retain_mapping_publication(state, source)
                if health['published'] > first:
                    if health['published'] != first+1 or health.get('fresh') is not True:
                        self.outcome = 'first_publication_invalid'
                        raise DiagnosticStop(self.outcome)
                    view, _ = bridge.associate(source, state['mapping_inputs'])
                    self.outcome = 'fresh_matching_publication'
                    self.record(self.outcome, force=True, publication=bridge.identity(view))
                    raise DiagnosticStop(self.outcome)
                filt = self.runtime.robot_filter
                if (filt is not None and filt.source_error is not None
                        and self.runtime._mapping_invalidation_error == filt.source_error
                        and health['reason'] == 'acquisition_failure'):
                    self.outcome = 'robot_filter_source_barrier'
                    raise DiagnosticStop(self.outcome)
            self.outcome = 'five_second_readiness_expired'
            raise DiagnosticStop(self.outcome)
        finally:
            self.record('readiness_stop', force=True, outcome=self.outcome,
                        readiness_seconds=self.clock()-started)

    def save(self, path):
        payload = dict(outcome=self.outcome, observation_ticks=self.ticks,
            samples_total=self.sampled, events_total=self.event_count,
            sample_limit=128, event_limit=64, sample_interval_s=.25,
            samples=list(self.samples), events=list(self.events), no_selected_goal=True)
        path.write_text(json.dumps(bench.baseline.serial(payload), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    timeline = Timeline()
    try:
        from robot_skill_stack.integrations.isaac.observation_runtime import ObservationRuntime
        from robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter import IsaacRobotSelfFilter
        native = SimpleNamespace(scenario='partial', execution_condition='frozen_rgbd', budget=64,
            headless=True, artifacts_dir=str(args.output), result=str(args.output/'result.json'))
        with timeline.installed(ObservationRuntime, IsaacRobotSelfFilter):
            bench.execute(native, app, research={'research_prepare': timeline.ready})
    finally:
        timeline.save(args.output/'mapping-diagnostic.json')
        app.close()


if __name__ == '__main__':
    main()
