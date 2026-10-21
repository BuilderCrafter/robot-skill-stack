"""Standalone GT/RGB-D mapping demo on the validated scene; no robot commands."""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False})

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import is_stage_loading, open_stage

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.spatial_mapping import (
    SpatialMappingConfig, SharedFrames, create_spatial_mapping)
from robot_skill_stack.integrations.isaac.world.factory import create_source
from robot_skill_stack.world.spatial_mapping import OccupancyState
from tests.support.isaac_async import run_kit_coroutine
from tests.support.results import fail_result, save_ppm, write_result

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT/'scenes/multiview_perception_final.usda'
PROFILE = ROOT/'config/scenes/multiview_perception_final.toml'


def main(args):
    output = Path(args.artifacts_dir) if args.artifacts_dir else Path(args.result).parent/'rgbd'
    output.mkdir(parents=True, exist_ok=True)
    protected = (SCENE, PROFILE, ROOT/'scenes/multiview_playground.usda', ROOT/'scenes/playground.usd')
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    if not open_stage(str(SCENE)):
        raise RuntimeError('Could not open validated scene')
    for _ in range(600):
        if not is_stage_loading():
            break
        simulation_app.update()
    else:
        raise RuntimeError('Stage loading timed out')
    world = World(stage_units_in_meters=1.)
    world.reset()
    world.play()
    cfg = load_scene_config(PROFILE)
    source = run_kit_coroutine(create_source(cfg), simulation_app)
    tap = mapping = truth = None
    try:
        tap = SharedFrames(source.rig)
        settings = SpatialMappingConfig(enabled=True, source='rgbd',
            bounds_min=(.12, -.4, -.02), bounds_max=(.8, .4, .3), voxel_size=.01,
            camera_prim_paths=cfg.multi_view.camera_prim_paths, stride=8, max_rays=1024,
            max_frame_age=2., stale_after=2., evidence_ttl=30.)
        mapping = create_spatial_mapping(settings, frame_source=tap)
        truth = create_spatial_mapping(replace(settings, source='ground_truth'),
                                       stage=omni.usd.get_context().get_stage())
    except BaseException:
        for resource in (truth, mapping, tap, source):
            if resource is not None:
                resource.close()
        raise
    timings, captures, queries = [], [], []

    def observe():
        for _ in range(8):
            world.step(render=True)
        batch = source.observe()  # Semantic owner performs the only acquisition.
        frames = tap.capture()
        assert frames is tap.latest and batch.timestamp == frames[0].acquired_at
        before = time.perf_counter()
        assert mapping.update(), dict(mapping.map.health)
        timings.append((time.perf_counter()-before)*1000)
        assert mapping.map.timestamp == min(frame.acquired_at for frame in frames)
        assert mapping.map.health['fresh'], dict(mapping.map.health)
        assert all(view['rays'] > 0 and view['occupied_voxels'] > 0
                   for view in mapping.source.diagnostics['cameras'].values())
        captures.append(dict(timestamp=mapping.map.timestamp,
            tokens={frame.camera_id: frame.token for frame in frames},
            diagnostics=mapping.source.diagnostics))

    try:
        for _ in range(4):
            observe()
        assert truth.update()
        gt, rgbd = truth.map.snapshot(), mapping.map.snapshot()
        assert gt.spec.bounds_min != tuple(cfg.scene.workspace_min)
        assert gt.spec.voxel_size != cfg.multi_view.voxel_size
        centers = gt.voxel_centers()
        gt_state, rgbd_state = gt.states.ravel(), rgbd.states.ravel()

        def compare(label, point, expected_gt, expected_rgbd):
            a, b = gt.query(point), rgbd.query(point)
            row = dict(label=label, position=np.asarray(point).tolist(),
                       ground_truth=a.state.name, rgbd=b.state.name,
                       rgbd_probability=b.probability, rgbd_observed=b.observed)
            queries.append(row)
            assert (a.state, b.state) == (expected_gt, expected_rgbd), row

        visible = ((gt_state == OccupancyState.OCCUPIED) &
                   (rgbd_state == OccupancyState.OCCUPIED) & (centers[:, 2] > .015))
        free = ((gt_state == OccupancyState.FREE) &
                (rgbd_state == OccupancyState.FREE) & (centers[:, 2] > .10))
        assert visible.sum() > 10 and free.sum() > 100, (visible.sum(), free.sum())
        compare('measured_surface', centers[np.flatnonzero(visible)[0]],
                OccupancyState.OCCUPIED, OccupancyState.OCCUPIED)
        compare('visible_empty_space', centers[np.flatnonzero(free)[len(np.flatnonzero(free))//2]],
                OccupancyState.FREE, OccupancyState.FREE)
        compare('occluded_cube_interior', (.36, -.16, .0275),
                OccupancyState.OCCUPIED, OccupancyState.UNKNOWN)
        assert not rgbd.query((.36, -.16, .0275)).observed
        assert rgbd.query((2., 0., 0.)).state == OccupancyState.OUT_OF_BOUNDS
        for name, snapshot in (('ground_truth', gt), ('rgbd', rgbd)):
            np.savez_compressed(output/f'{name}.npz', states=snapshot.states,
                probabilities=snapshot.probabilities, observed=snapshot.observed,
                last_observed=snapshot.last_observed, bounds_min=snapshot.spec.bounds_min,
                bounds_max=snapshot.spec.bounds_max, voxel_size=snapshot.spec.voxel_size)
            # XY slice through the cube interior: unknown gray, free white, occupied red.
            z = int(snapshot.spec.world_to_indices((.36, -.16, .0275))[0, 2])
            palette = np.array([[90, 90, 90], [245, 245, 245], [220, 45, 35]], dtype=np.uint8)
            save_ppm(output/f'{name}_xy.ppm', palette[snapshot.states[:, :, z].T[::-1]])
        (output/'queries.json').write_text(json.dumps(queries, indent=2)+'\n')

        previous = (mapping.map.timestamp, mapping.map.revision)
        assert source.observe().timestamp is None  # No render step: exact old buffers.
        assert not mapping.update()
        assert (mapping.map.timestamp, mapping.map.revision) == previous
        assert not mapping.map.health['fresh']
        frozen = source.rig.cameras[cfg.multi_view.camera_prim_paths[-1]]
        frozen.camera.pause()
        for _ in range(8):
            world.step(render=True)
        try:
            source.observe()
        except RuntimeError as exc:
            assert 'frozen' in str(exc).lower(), str(exc)
        else:
            raise AssertionError('Frozen camera accepted by acquisition')
        try:
            mapping.update()
        except RuntimeError as exc:
            assert 'acquisition failed' in str(exc).lower(), str(exc)
        else:
            raise AssertionError('Failed acquisition accepted by mapping')
        assert (mapping.map.timestamp, mapping.map.revision) == previous
        assert mapping.map.health['status'] == 'error' and not mapping.map.health['fresh']
        frozen.camera.resume()
        observe()

        mapping.close()
        mapping.close()
        tap.close()
        tap.close()
        assert not source.rig._closed and all(not c._closed for c in source.rig.cameras.values())
        # A replacement mapping consumer borrows the same already initialized rig.
        tap = SharedFrames(source.rig)
        mapping = create_spatial_mapping(settings, frame_source=tap)
        observe()
        assert hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
        (output/'captures.json').write_text(json.dumps(captures, indent=2)+'\n')
        metrics = dict(dimensions=gt.spec.shape, voxel_size=gt.spec.voxel_size,
            ground_truth_counts=gt.counts(), rgbd_counts=rgbd.counts(),
            ground_truth_health=dict(gt.health), rgbd_health=dict(rgbd.health),
            ground_truth_update_ms=truth.source.diagnostics['update_seconds']*1000,
            rgbd_update_ms=timings, camera_paths=settings.camera_prim_paths,
            matching_visible_surface_cells=int(visible.sum()), matching_free_cells=int(free.sum()),
            queries=queries, shared_exact_snapshots=True, duplicate_buffers_rejected=True,
            frozen_camera_not_fresh=True, recovered=True, reinitialized_without_new_cameras=True,
            saved_scenes_and_config_unchanged=True, robot_commands=0,
            automatic_robot_masking=False)
    finally:
        mapping.close()
        truth.close()
        tap.close()
        source.close()
        source.close()
        for camera in source.rig.cameras.values():
            assert camera._closed and camera.camera._render_product is None
            assert camera.camera._fabric_time_annotator is None
    metrics['all_cameras_closed'] = True
    write_result(args.result, status='PASS', metrics=metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir')
    args = parser.parse_args()
    code = 0
    try:
        main(args)
    except Exception as exc:
        fail_result(args.result, exc)
        code = 1
    finally:
        simulation_app.close()
    raise SystemExit(code)
