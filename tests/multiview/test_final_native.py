"""Bounded final-scene native validation; actual cameras and production async runtime.

Run with run_isaac_python.sh --headless; no fixture geometry or synthetic frames.
Timing hooks are test-only and never change production data/source checks.
"""
import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRUTH = dict(Square=('cube', (.36, -.16, .0275), (.055, .055, .055)),
             Rectangle=('cube', (.54, .16, .024), (.068, .042, .048)),
             SmallSphere=('sphere', (.36, .16, .022), (.044,)*3),
             LargeSphere=('sphere', (.54, -.16, .03), (.060,)*3))


def percentile(values):
    return None if not values else dict(count=len(values), p50=float(np.percentile(values, 50)),
                                       p95=float(np.percentile(values, 95)), maximum=float(max(values)))


def snapshot(model):
    rows = []
    for obj in model.visible_objects():
        g = obj.geometry
        dims = g.box_size if obj.class_name == 'cube' else np.full(3, 2*g.radius) if obj.class_name == 'sphere' else obj.size
        rows.append(dict(id=obj.object_id, shape=obj.class_name, center=obj.pose.position.tolist(),
            dimensions=dims.tolist(), confidence=obj.confidence, graspable=obj.graspable,
            views=obj.metadata.get('contributing_views'), last_seen=obj.last_seen,
            reason=g.metadata.get('reason'), parameter_reason=g.metadata.get('parameter_reason')))
    return rows


def score(rows):
    # Evaluation only: no scene identities, dimensions or positions enter detection.
    assigned, used = {}, set()
    for distance, name, index in sorted((np.linalg.norm(np.asarray(row['center'])-center), name, i)
            for name, (_, center, _) in TRUTH.items() for i, row in enumerate(rows)):
        if distance <= .06 and name not in assigned and index not in used:
            assigned[name] = index
            used.add(index)
    result = []
    for name, (shape, center, dimensions) in TRUTH.items():
        if name not in assigned:
            result.append(dict(name=name, detected=False, passed=False))
            continue
        row = rows[assigned[name]]
        measured = row['dimensions']
        if shape == row['shape'] == 'cube':
            measured, dimensions = sorted(measured[:2])+measured[2:], sorted(dimensions[:2])+list(dimensions[2:])
        position_error = float(np.linalg.norm(np.asarray(row['center'])-center)*1000)
        dimension_error = float(np.max(abs(np.asarray(measured)-dimensions))*1000)
        result.append(dict(row, name=name, detected=True, position_error_mm=position_error,
            dimension_error_mm=dimension_error,
            passed=row['shape']==shape and position_error<=10 and dimension_error<=10 and row['graspable']))
    return dict(objects=result, extras=len(rows)-len(used), passed=len(rows)==4 and all(r['passed'] for r in result))


class Probe:
    def __init__(self):
        self.phase = 'warmup'
        self.events, self.states = [], []
        self.last_frames = None
        self.bundle = None
        self.next_sample = 0.

    def event(self, kind, **values):
        self.events.append(dict(at=time.monotonic(), phase=self.phase, kind=kind, **values))

    def factory(self, compute, **kwargs):
        from robot_skill_stack.runtime.process_worker import ProcessWorker
        from tests.multiview.native_probe import TimedCompute
        worker = ProcessWorker(TimedCompute(compute), **kwargs)
        original = worker.poll
        def poll():
            completed = original()
            if completed is not None:
                if completed.error:
                    self.event('worker_error', error=completed.error)
                else:
                    result, timings = completed.value
                    self.event('completed', timestamp=result.timestamp, duration=completed.duration,
                               timings=timings, detections=len(result.detections))
                    completed = replace(completed, value=result)
            return completed
        worker.poll = poll
        return worker

    def install(self, bundle):
        self.bundle = bundle
        runtime, updater = bundle.observation_runtime, bundle.world_updater
        pool = runtime.pool
        original_batch, original_frame = pool.batch, pool._frame
        def frame(key, token):
            cached = pool._cache.get(key)
            result = original_frame(key, token)
            if cached is None or cached[0] != token:
                self.event('camera', camera=key, token=token)
            return result
        def batch(*args, **kwargs):
            try:
                result = original_batch(*args, **kwargs)
                if result is not None:
                    self.last_frames = result
                    self.event('batch', tokens=[f.token for f in result],
                               timestamp=min(f.acquired_at for f in result))
                else:
                    self.event('no_batch', state=dict(pool.status))
                return result
            except Exception as exc:
                self.event('batch_error', error=str(exc))
                raise
        pool.batch, pool._frame = batch, frame
        original_accepts, original_publish, original_failure = updater.accepts, updater.publish, updater.note_failure
        def accepts(timestamp, revision=None):
            accepted = original_accepts(timestamp, revision)
            if not accepted:
                self.event('rejected', timestamp=timestamp, age=time.monotonic()-timestamp,
                           revision=revision, current_revision=bundle.world_model.context_revision)
            return accepted
        def publish(batch, context_revision=None):
            before, started = updater.update_count, time.perf_counter()
            result = original_publish(batch, context_revision)
            if updater.update_count > before:
                self.event('published', timestamp=batch.timestamp, age=time.monotonic()-batch.timestamp,
                           seconds=time.perf_counter()-started, objects=snapshot(bundle.world_model))
            else:
                self.event('publish_failed', error=updater.last_error)
            return result
        def failure(error):
            self.event('failure', error=str(error))
            return original_failure(error)
        updater.accepts, updater.publish, updater.note_failure = accepts, publish, failure

    def sample(self):
        now = time.monotonic()
        if now >= self.next_sample:
            b = self.bundle
            moving = self.phase!='lifecycle' and b.world.is_playing()
            last = b.world_updater.health.last_frame_at
            self.states.append(dict(at=now, simulation_time=b.world.current_time, phase=self.phase,
                health=b.world_updater.health.status, error=b.world_updater.last_error,
                frame_age=None if last is None else now-last,
                objects=snapshot(b.world_model), ee=b.backend.get_end_effector_pose().position.tolist() if moving else None,
                joints=np.asarray(b.backend.robot.get_joint_positions()).tolist() if moving else None))
            self.next_sample = now+.2

    def drive(self, app, seconds, *, simulation_seconds=None, until=None):
        start, sim = time.monotonic(), self.bundle.world.current_time
        while time.monotonic()-start < seconds:
            app.update()
            self.sample()
            if until is not None and until():
                break
            if simulation_seconds is not None and self.bundle.world.current_time-sim >= simulation_seconds:
                break
        return dict(wall_seconds=time.monotonic()-start,
                    simulation_seconds=self.bundle.world.current_time-sim)

    def images(self, output, label):
        from tests.support.results import save_ppm, save_pgm
        for i, frame in enumerate(self.last_frames or ()):
            if frame.rgb is not None:
                save_ppm(output/f'{label}-{i}.ppm', frame.rgb)
            save_pgm(output/f'{label}-{i}-depth.pgm', np.nan_to_num(frame.depth, nan=0, posinf=0).clip(0, 2)*127)
            np.savez_compressed(output/f'{label}-{i}.npz', depth=frame.depth, intrinsics=frame.intrinsics,
                                world_from_camera=frame.world_from_camera, token=frame.token, acquired_at=frame.acquired_at)


def run(args, app, metrics):
    import omni.kit.app
    import omni.usd
    from isaacsim.core.utils.stage import open_stage, is_stage_loading
    from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
    from robot_skill_stack.integrations.isaac import observation_runtime
    from tests.support.isaac_async import run_kit_coroutine
    from omni.kit.async_engine import run_coroutine
    output = Path(args.artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    scene, config = ROOT/'scenes/multiview_perception_final.usda', ROOT/'config/scenes/multiview_perception_final.toml'
    metrics['source_hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [scene, config, ROOT/'scenes/multiview_playground.usda', ROOT/'scenes/playground.usd',
                  ROOT/'robot_skill_stack/world/perception/multiview/segmentation.py']}
    metrics['environment'] = dict(python=platform.python_version(), numpy=np.__version__, numpy_path=np.__file__,
        headless=args.headless, scene=str(scene), config=str(config), sdk=Path('/isaac-sim/VERSION').read_text().strip()
        if Path('/isaac-sim/VERSION').exists() else 'see launcher SDK VERSION')
    if not open_stage(str(scene)):
        raise RuntimeError('Could not open exact final scene')
    for _ in range(600):
        if not is_stage_loading():
            break
        app.update()
    else:
        raise RuntimeError('Final scene loading timed out')
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    probe = Probe()
    # Hook construction, retaining the actual production ProcessWorker and lifecycle.
    real_init = observation_runtime.ObservationRuntime.__init__
    def init(instance, *positional, **keywords):
        keywords['worker_factory'] = probe.factory
        real_init(instance, *positional, **keywords)
    bundle = None
    try:
        with patch.object(observation_runtime.ObservationRuntime, '__init__', init):
            bundle = run_kit_coroutine(build_runtime(config), app)
        probe.install(bundle)
        worker = bundle.observation_runtime._semantic_worker
        probe.drive(app, 15., until=lambda: bundle.world_updater.update_count >= 6 and len(bundle.world_model.visible_objects())==4)
        metrics['cameras'] = {key: dict(intrinsics=c.get_intrinsics().tolist(), transform=c.get_world_from_camera_transform().tolist(),
            token=c.get_frame_token(), resolution=list(c.camera.get_resolution())) for key,c in bundle.world_updater.provider.rig.cameras.items()}
        probe.phase = 'static'
        probe.drive(app, 3.)
        rows = [e for e in probe.events if e['phase']=='static' and e['kind']=='published']
        evaluation = score(rows[-1]['objects'] if rows else snapshot(bundle.world_model))
        evaluation['publication_count'] = len(rows)
        evaluation['stable_ids'] = bool(len(rows)>=2 and all(
            [r.get('id') for r in score(row['objects'])['objects']] == [r.get('id') for r in evaluation['objects']]
            for row in rows))
        static_states = [s for s in probe.states if s['phase']=='static']
        evaluation['visibility_samples'] = len(static_states)
        evaluation['empty_visibility_samples'] = sum(not s['objects'] for s in static_states)
        evaluation['all_published_geometry_passed'] = bool(rows and all(score(r['objects'])['passed'] for r in rows))
        evaluation['status'] = 'PASS' if evaluation['all_published_geometry_passed'] and evaluation['stable_ids'] and not evaluation['empty_visibility_samples'] else 'FAIL'
        metrics['phase_a'] = evaluation
        probe.images(output, 'static')
        print('PHASE A', json.dumps(evaluation), flush=True)

        probe.phase = 'throughput'
        before = worker.health
        elapsed = probe.drive(app, 90., simulation_seconds=12.)
        after = worker.health
        window = [e for e in probe.events if e['phase']=='throughput']
        completed = [e for e in window if e['kind']=='completed']
        published = [e for e in window if e['kind']=='published']
        metrics['phase_b'] = dict(elapsed, configured_hz=bundle.config.world.update_hz,
            worker={k: after[k]-before[k] for k in ('submitted','started','completed','replaced','discarded','errors','delivered')},
            counts={kind:sum(e['kind']==kind for e in window) for kind in ('camera','batch','no_batch','batch_error','worker_error','published','rejected','failure')},
            published_hz=len(published)/elapsed['wall_seconds'],
            published_per_simulation_second=len(published)/elapsed['simulation_seconds'] if elapsed['simulation_seconds'] else 0,
            worker_seconds=percentile([e['duration'] for e in completed]),
            compute_seconds=percentile([e['timings']['seconds'] for e in completed]),
            transfer_and_io_seconds=percentile([max(0,e['duration']-e['timings']['seconds']) for e in completed]),
            age_seconds=percentile([e['age'] for e in published]),
            stages_seconds={name:percentile([e['timings']['stages'].get(name,0) for e in completed])
                for name in ('projection_voxel','coarse','refinement','metric_graph','fit','parameters','observability')},
            visibility_counts={str(n):sum(len(s['objects'])==n for s in probe.states if s['phase']=='throughput') for n in range(9)})
        metrics['phase_b']['stale_visibility_samples'] = sum(s['health']=='stale' for s in probe.states if s['phase']=='throughput')
        metrics['phase_b']['status'] = 'PASS' if published and not metrics['phase_b']['worker']['errors'] and not any(e['kind']=='rejected' for e in window) and not metrics['phase_b']['stale_visibility_samples'] else 'FAIL'
        print('PHASE B', json.dumps(metrics['phase_b']), flush=True)

        metrics['phase_c'] = dict(status='BLOCKED', reason='Initial four-object detection did not pass')
        if evaluation['all_published_geometry_passed'] and evaluation['stable_ids']:
            probe.drive(app, 5., until=lambda: score(snapshot(bundle.world_model))['passed'])
            target_id = next(r['id'] for r in evaluation['objects'] if r['name']=='LargeSphere')
            planner = bundle.runtime.registry.get('pick').grasp_planner
            planned = planner.plan(bundle.world_model.require(target_id))
            if not planned.ok:
                metrics['phase_c']['reason'] = str(planned)
            else:
                home, pre = bundle.backend.get_end_effector_pose(), planned.plan.pre_grasp
                intermediate = pre.translated([0,0,.12])
                metrics['phase_c'] = dict(status='PASS', target_id=target_id, motions=[], initial=snapshot(bundle.world_model))
                for label, target in (('intermediate',intermediate), ('pregrasp',pre), ('retreat',intermediate), ('home',home)):
                    probe.phase = label
                    async def move():
                        return await asyncio.wait_for(bundle.backend.move_to_pose_async(target), timeout=25.)
                    task = run_coroutine(move())
                    while not task.done():
                        app.update()
                        probe.sample()
                    outcome = task.result()
                    actual = bundle.backend.get_end_effector_pose()
                    row = dict(phase=label, ok=outcome.ok, message=outcome.message, requested=target.position.tolist(),
                        actual=actual.position.tolist(), error_mm=float(np.linalg.norm(actual.position-target.position)*1000))
                    metrics['phase_c']['motions'].append(row)
                    probe.drive(app, 2.)
                    if label in ('pregrasp','home'):
                        probe.images(output, label)
                    if not outcome.ok:
                        metrics['phase_c']['status'] = 'FAIL'
                        break
                timeline = []
                for label in ('intermediate','pregrasp','retreat','home'):
                    states = [s for s in probe.states if s['phase']==label]
                    targets = [next((o for o in s['objects'] if o['id']==target_id), None) for s in states]
                    timeline.append(dict(phase=label, samples=len(states), missing=sum(o is None for o in targets),
                        unknown=sum(o is not None and o['shape']!='sphere' for o in targets),
                        ungraspable=sum(o is not None and not o['graspable'] for o in targets),
                        extras=sum(len(s['objects'])>4 for s in states),
                        center_errors_mm=percentile([float(np.linalg.norm(np.asarray(o['center'])-TRUTH['LargeSphere'][1])*1000)
                                                    for o in targets if o is not None])))
                metrics['phase_c']['timeline'] = timeline
                if any(not r['samples'] or r['missing'] or r['unknown'] or r['ungraspable'] or r['extras']
                       or r['center_errors_mm']['maximum']>10 for r in timeline):
                    metrics['phase_c']['status'] = 'FAIL'

        probe.phase = 'lifecycle'
        runtime, cameras = bundle.observation_runtime, bundle.world_updater.provider.rig.cameras
        for camera in cameras.values():
            camera.camera.pause()
        probe.drive(app, 1.3)
        frozen_last, frozen_count = bundle.world_updater.health.last_frame_at, bundle.world_updater.update_count
        probe.drive(app, .5)
        lifecycle = dict(frozen_exposure_not_republished=bundle.world_updater.health.last_frame_at==frozen_last
                         and bundle.world_updater.update_count==frozen_count,
                         frozen_objects_invisible=not bundle.world_model.visible_objects())
        for camera in cameras.values():
            camera.camera.resume()
        probe.drive(app, 10., until=lambda: bundle.world_updater.update_count>frozen_count and bundle.world_updater.health.status=='fresh')
        lifecycle['resume_fresh'] = bundle.world_updater.update_count>frozen_count and bundle.world_updater.health.status=='fresh'
        # Yield real Kit renders between camera reads, with runtime ticks detached.
        runtime._subscription = None
        runtime.pool.reset()
        app.update()
        camera = next(iter(cameras.values()))
        original_depth = camera.get_depth
        def advancing_depth():
            old, depth = camera.get_frame_token(), original_depth()
            for _ in range(4):
                app.update()
                if camera.get_frame_token()!=old:
                    break
            if camera.get_frame_token()==old:
                raise RuntimeError('Native exposure did not advance during in-flight probe')
            return depth
        try:
            runtime.pool.begin_update()
            with patch.object(camera, 'get_depth', advancing_depth):
                runtime.pool.batch('inflight-probe', tuple(cameras), simulation_time=bundle.world.current_time)
        except RuntimeError as exc:
            lifecycle['inflight_reason'] = str(exc)
            lifecycle['inflight_mixed_batch_rejected'] = 'updated during acquisition' in str(exc)
        else:
            lifecycle['inflight_mixed_batch_rejected'] = False
        runtime.start(omni.kit.app.get_app())
        probe.drive(app, 3., until=lambda: worker.health['running'])
        lifecycle['work_inflight_at_stop'] = worker.health['running'] or worker.health['pending']
        epoch = worker.health['epoch']
        run_kit_coroutine(bundle.world.stop_async(), app)
        app.update()
        lifecycle['stop_invalidated_epoch'] = worker.health['epoch']>epoch
        stopped_count = bundle.world_updater.update_count
        probe.drive(app,.4)
        lifecycle['no_stopped_publication'] = bundle.world_updater.update_count==stopped_count
        restarted = time.monotonic()
        run_kit_coroutine(bundle.world.play_async(), app)
        probe.drive(app, 10., until=lambda: bundle.world_updater.update_count>stopped_count and bundle.world_updater.health.status=='fresh')
        lifecycle['restart_fresh'] = bundle.world_updater.update_count>stopped_count and bundle.world_updater.health.last_frame_at>=restarted
        metrics['phase_d'] = lifecycle
    finally:
        if bundle is not None:
            worker = bundle.observation_runtime._semantic_worker
            cameras = tuple(bundle.world_updater.provider.rig.cameras.values())
            bundle.close()
            worker._thread.join(timeout=3.)
            metrics['cleanup'] = dict(worker_closed=worker.health['closed'], thread_stopped=not worker._thread.is_alive(),
                child_exited=worker._child is None or worker._child.poll() is not None,
                cameras_closed=all(c._closed and c.camera._render_product is None and
                    c.camera._acquisition_callback is None and c.camera._fabric_time_annotator is None for c in cameras))
        (output/'events.json').write_text(json.dumps(probe.events, indent=2)+'\n')
        (output/'states.json').write_text(json.dumps(probe.states, indent=2)+'\n')
        metrics['files_unchanged'] = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in metrics['source_hashes'].items())
    if 'phase_d' in metrics:
        metrics['phase_d']['status'] = 'PASS' if all(v is True for k,v in metrics['phase_d'].items() if k!='inflight_reason') and all(metrics['cleanup'].values()) else 'FAIL'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args()
    if np.__version__!='1.26.4':
        raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless':args.headless})
    from tests.support.results import write_result
    metrics, error = {}, None
    try:
        run(args, app, metrics)
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        passed = error is None and all(metrics.get(f'phase_{p}',{}).get('status')=='PASS' for p in 'abcd')
        write_result(args.result, status='PASS' if passed else 'FAIL', metrics=metrics, error=error)
        app.close()
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
