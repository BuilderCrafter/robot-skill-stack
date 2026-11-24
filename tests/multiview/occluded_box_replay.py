"""Replay three actual archived exposures through compute, association and WorldModel.

Original timestamps are preserved. An offline clock uses each archived capture
time plus measured compute duration; this does not validate live freshness/queues.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.perception.multiview.provider import MultiViewProvider
from tests.multiview.performance_ab import digest, native_frames
from tests.multiview.test_final_native import score, snapshot
from tests.support.results import fail_result, write_result


def run(data):
    manifest = json.loads(data.read_text())
    cases = dict(manifest['cases'])
    home = []
    for index, row in enumerate(cases['static']['frames']):
        path = Path(row['path']).with_name(f'home-{index}.npz')
        with np.load(path, allow_pickle=False) as captured, np.load(row['path'], allow_pickle=False) as static:
            assert np.array_equal(captured['intrinsics'], static['intrinsics'])
            assert np.array_equal(captured['world_from_camera'], static['world_from_camera'])
        home.append(dict(path=str(path),sha256=digest(path),camera_id=row['camera_id']))
    cases['home'] = dict(frames=home)
    scene = load_scene_config('config/scenes/multiview_perception_final.toml').scene
    provider = MultiViewProvider(SimpleNamespace(close=lambda: None), scene)
    model = WorldModel()
    updater = WorldModelUpdater(model, provider)
    rows, ids = [], None
    previous_timestamp = -np.inf
    try:
        for name in ('static','pregrasp','home'):
            frames = native_frames(cases[name])
            prepared = provider.acquire(frames=frames)
            assert prepared.timestamp>previous_timestamp
            previous_timestamp = prepared.timestamp
            context = model.observation_context()
            with patch.object(provider.processor.fitter, 'fit', wraps=provider.processor.fitter.fit) as fitted:
                started = time.perf_counter()
                result = provider.processor.compute(prepared)
                duration = time.perf_counter()-started
                calls = fitted.call_count
            assert calls<=24
            assert result.timestamp==prepared.timestamp
            assert sorted(d.fit.geometry.shape.value for d in result.detections)==['cube','cube','sphere','sphere']
            qualified = [dict(shape=d.fit.geometry.shape.value,center=d.fit.center.tolist(),
                dimensions=d.fit.size.tolist(),
                parameter_observable=d.fit.geometry.metadata['parameter_observable']) for d in result.detections]
            assert all(row['parameter_observable'] for row in qualified)
            # Replay the original clock domain, including actual compute cost.
            with patch('time.monotonic',return_value=prepared.timestamp+duration):
                assert updater.accepts(result.timestamp, context.revision)
                batch = provider.associate(result, context)
                updater.publish(batch, context_revision=context.revision)
                current_ids = set(provider.tracker.tracks)
                if ids is None:
                    ids = current_ids
                assert current_ids==ids and len(ids)==4
                objects = snapshot(model)
                rows.append(dict(scene=name,timestamp=result.timestamp,tokens=[f.token for f in frames],
                    compute_seconds=duration,fitting_calls=calls,published=len(batch.observations),
                    detections=qualified,visible=objects,evaluation=score(objects)))
        assert len(model.visible_objects())==4
        assert rows[-1]['evaluation']['passed']
        assert [r['published'] for r in rows]==[0,0,4]
        assert all(d.fit.geometry.metadata['parameter_observable'] for d in result.detections)
        unchanged = all(digest(row['path'])==row['sha256'] for case in cases.values() for row in case['frames'])
        assert unchanged
        return dict(clock='archived acquisition time + measured compute; no live scheduling/queue',
                    input_manifest_sha256=digest(data),inputs_unchanged=unchanged,observations=rows,
                    ids=sorted(ids),final_objects=4)
    finally:
        updater.stop()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    args = parser.parse_args()
    try:
        metrics = run(args.data)
        write_result(args.result,status='PASS',metrics=metrics)
        print(json.dumps(dict(status='PASS',objects=metrics['final_objects'],ids=metrics['ids'])))
    except Exception as exc:
        fail_result(args.result,exc)
        raise SystemExit(1)
