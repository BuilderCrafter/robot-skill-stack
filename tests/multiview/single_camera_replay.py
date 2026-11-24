"""Experimental native one/two-camera replay of immutable original captures.

Reuses original exposures, numerical profiling and the final-scene evaluator.
Run after other tests finish; processor timings exclude acquisition/publication.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.common.camera import FrameSynchronizer
from robot_skill_stack.world.common.shared_frames import SharedFramePool
from robot_skill_stack.world.model import ObservationBatch, WorldModel, WorldModelUpdater
from robot_skill_stack.world.perception import build_perception_provider
from robot_skill_stack.world.perception.multiview.provider import MultiViewInput
from tests.multiview.performance_ab import digest, environment, geometry, native_frames, profiled_compute
from tests.multiview.test_final_native import TRUTH, score, snapshot
from tests.unit.test_multiview_provider import Camera


class ArchivedCamera(Camera):
    def get_frame_token(self):
        return self.frame.token


def captures(data):
    manifest = json.loads(data.read_text())
    cases = dict(manifest['cases'])
    home = []
    for index, row in enumerate(cases['static']['frames']):
        path = Path(row['path']).with_name(f'home-{index}.npz')
        home.append(dict(path=str(path), sha256=digest(path), camera_id=row['camera_id']))
    cases['home'] = dict(frames=home)
    frames = {name: native_frames(case) for name, case in cases.items()}
    for batch in frames.values():
        for reference, frame in zip(frames['static'], batch):
            assert reference.camera_id == frame.camera_id
            np.testing.assert_array_equal(reference.intrinsics, frame.intrinsics)
            np.testing.assert_array_equal(reference.world_from_camera, frame.world_from_camera)
    return manifest, cases, frames


def detection_rows(detections):
    return [dict(geometry(d.fit, finalized=True), views=list(d.views), points=len(d.points),
                 observable=d.fit.geometry.metadata.get('parameter_observable', False),
                 parameter_reason=d.fit.geometry.metadata.get('parameter_reason'),
                 scores=d.fit.geometry.scores) for d in detections]


def evaluate(rows):
    result = score(rows)  # Existing 60 mm assignment and 10 mm center/dimension limits.
    for row in result['objects']:
        if not row['detected']:
            row['outcome'] = 'not_detected'
            continue
        expected_shape, _, dimensions = TRUTH[row['name']]
        accurate = (row['shape'] == expected_shape and
                    max(row['position_error_mm'], row['dimension_error_mm']) <= 10)
        row['outcome'] = ('unknown' if row['shape'] == 'unknown' else
                          'incorrect_geometry' if not accurate else
                          'qualified' if row['graspable'] else 'unqualified')
        row['radius_error_mm'] = (abs(row['radius']-dimensions[0]/2)*1000
                                  if row['shape'] == expected_shape == 'sphere' else None)
        yaw = {'Square': 25., 'Rectangle': -33.}.get(row['name'])
        row['yaw_error_deg'] = (abs((np.rad2deg(row['yaw'])-yaw+45) % 90-45)
                               if yaw is not None and row['yaw'] is not None else None)
    return result


def benchmark(processor, frames, repeats):
    prepared = MultiViewInput(frames, min(f.acquired_at for f in frames))
    reference = processor.compute(prepared)  # One untimed warm-up, also retains qualification metadata.
    runs = [profiled_compute(processor, prepared) for _ in range(repeats)]
    expected = [dict(geometry(d.fit, finalized=True), views=list(d.views)) for d in reference.detections]
    stable = lambda row: {k: v for k, v in row.items() if k not in ('seconds', 'stages')}
    for row in runs:
        assert row['detections'] == expected and stable(row) == stable(runs[0])
        assert row['fitting_calls'] <= 24
        assert row['input_timestamp'] == row['result_timestamp'] == prepared.timestamp
    return dict(detected=len(expected), evaluation=evaluate(detection_rows(reference.detections)), runs=runs,
                camera_ids=[f.camera_id for f in frames], tokens=[f.token for f in frames],
                acquired_at=[f.acquired_at for f in frames])


def provider_contract(config, batches, mode):
    cameras = {f.camera_id: ArchivedCamera(f) for f in batches['static']}
    rig = FrameSynchronizer(cameras, max_skew=config.multi_view.max_skew)
    pool = SharedFramePool(cameras, clock=lambda: time.monotonic())
    provider = build_perception_provider('multi_view', rig, scene=config.scene, settings=config.multi_view)
    model = WorldModel()
    updater = WorldModelUpdater(model, provider)
    history = []
    try:
        for name in ('static', 'pregrasp', 'home'):
            frames = batches[name]
            stamp = min(f.acquired_at for f in frames)
            assert all(f.acquired_at == stamp for f in frames)
            for frame in frames:
                cameras[frame.camera_id].frame = frame
            context = model.observation_context()
            started = time.perf_counter()
            with patch('time.monotonic', return_value=stamp), \
                 patch.object(provider.processor.fitter, 'fit', wraps=provider.processor.fitter.fit) as fitted:
                if mode == 'observe':
                    batch = provider.observe(context)
                    assert provider.acquire() is None
                    assert provider.observe(context).timestamp is None
                else:
                    pool.begin_update()
                    shared = pool.batch('semantic', tuple(cameras))
                    borrowed = pool.batch('mapping', tuple(cameras))
                    assert pool.batch('semantic', tuple(cameras)) is None
                    for original, copied, reused in zip(frames, shared, borrowed):
                        assert copied is reused and copied.token == original.token
                        assert copied.acquired_at == original.acquired_at
                        for field in ('depth', 'intrinsics', 'world_from_camera'):
                            np.testing.assert_array_equal(getattr(original, field), getattr(copied, field))
                            assert not getattr(copied, field).flags.writeable
                    prepared = provider.acquire(frames=shared)
                    computed = provider.processor.compute(prepared)
            duration = time.perf_counter()-started
            with patch('time.monotonic', return_value=stamp+duration):
                if mode == 'split':
                    assert updater.accepts(computed.timestamp, context.revision)
                    batch = provider.associate(computed, context)
                assert isinstance(batch, ObservationBatch) and batch.timestamp == stamp
                observations = updater.publish(batch, context_revision=context.revision)
                assert updater.health.last_frame_at == stamp
                assert not updater.accepts(stamp, context.revision)
                assert fitted.call_count <= 24
                for obj in model.visible_objects():
                    assert obj.last_seen == stamp
                active = [track for track in provider.tracker.tracks.values() if track.visible]
                tracks = [dict(row, id=track.object_id, hits=track.hits)
                          for row, track in zip(detection_rows([t.detection for t in active]), active)]
                history.append(dict(capture=name, timestamp=stamp, context_revision=context.revision,
                    published=[obj.object_id for obj in observations], tracks=evaluate(tracks),
                    visible=snapshot(model), compute_and_acquire_seconds=duration))
        forgotten = next(iter(provider.tracker.tracks))
        provider.forget(forgotten)
        assert forgotten not in provider.tracker.tracks
        assert provider.get_point_cloud(forgotten) is None
        return dict(mode=mode, history=history, final_evaluation=score(snapshot(model)),
                    forget_verified=True, close_verified=True)
    finally:
        pool.close()  # Borrowed cameras remain owned by the rig/provider.
        provider.close()
        provider.close()
        assert provider.acquire() is None and provider.observe().timestamp is None
        assert all(camera.closes == 1 for camera in cameras.values())


def run(data, repeats):
    assert repeats >= 1
    assert subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'robot_skill_stack', 'config']).returncode == 0
    manifest, cases, frames = captures(data)
    config_path = Path('config/scenes/multiview_perception_final.toml')
    assert digest(config_path) == manifest['config_sha256']
    config = load_scene_config(config_path)
    selections = dict(Main=(0,), Side=(1,), Both=(0, 1))
    records = {}
    before = environment()
    for label, indices in selections.items():
        selected = {name: tuple(batch[i] for i in indices) for name, batch in frames.items()}
        rig = FrameSynchronizer({f.camera_id: ArchivedCamera(f) for f in selected['static']})
        provider = build_perception_provider('multi_view', rig, scene=config.scene, settings=config.multi_view)
        try:
            measured = {name: benchmark(provider.processor, selected[name], repeats)
                        for name in ('static', 'pregrasp')}
        finally:
            provider.close()
        records[label] = dict(measured=measured,
            contracts=[provider_contract(config, selected, mode) for mode in ('split', 'observe')])
    assert all(digest(row['path']) == row['sha256'] for case in cases.values() for row in case['frames'])
    sources = subprocess.check_output(['git', 'ls-files', 'robot_skill_stack/world'], text=True).splitlines()
    return dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        harness_sha256=digest(__file__), input_manifest_sha256=digest(data), captures=cases,
        configuration_sha256=digest(config_path), settings=manifest['settings'],
        sources={path: digest(path) for path in sources}, environment_before=before,
        environment_after=environment(), warmup=1, repeats=repeats, results=records,
        clock='original acquisition time + measured work; no live scheduling/availability claim')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve existing experiment outputs'
    result = run(args.data, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for label, row in result['results'].items():
        print(label, {name: (value['detected'], value['evaluation']['passed'])
                      for name, value in row['measured'].items()},
              'published', [[len(step['published']) for step in c['history']] for c in row['contracts']])
