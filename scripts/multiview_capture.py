"""Checkpoint A: capture two real cameras without commanding the robot."""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False})

import argparse
import json
from pathlib import Path
import time

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.world.multiview import create_cameras
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, fuse_clouds, project_depth, save_ply
from tests.support.results import fail_result, save_ppm, write_result
from tests.support.multiview_native import CAMERAS, ROOT, open_experiment


def nearest_errors(query, reference):
    query = query[np.linspace(0, len(query)-1, min(64, len(query)), dtype=int)]
    best = np.full(len(query), np.inf)
    for start in range(0, len(reference), 10000):
        delta = query[:, None]-reference[None, start:start+10000]
        best = np.minimum(best, (delta*delta).sum(2).min(1))
    return np.sqrt(best)


def main(args):
    output = Path(args.artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    world = open_experiment(simulation_app)
    rig = create_cameras(CAMERAS)
    try:
        before = time.perf_counter()
        for _ in range(90):
            world.step(render=True)
        render_seconds = time.perf_counter()-before
        frames = rig.capture()
        scene = load_scene_config(ROOT/'config/scenes/playground.toml').scene
        clouds, metrics = [], {'render_warmup_seconds': render_seconds, 'cameras': {}}
        for index, frame in enumerate(frames):
            camera = rig.cameras[frame.camera_id]
            points, pixels = project_depth(frame)
            z = frame.depth[pixels[:, 0], pixels[:, 1]]
            reference = camera.image_points_to_world(pixels[:, ::-1]+.5, z)
            projection_error = np.linalg.norm(points-reference, axis=1)
            sdk = np.asarray(camera.camera.get_pointcloud(world_frame=True))
            sdk = sdk[np.isfinite(sdk).all(1)]
            unordered_error = nearest_errors(points, sdk)
            cloud = crop_workspace(points, scene)
            clouds.append(cloud)
            save_ply(output/f'camera_{index}.ply', cloud)
            np.savez_compressed(output/f'camera_{index}.npz', depth=frame.depth,
                                intrinsics=frame.intrinsics, world_from_camera=frame.world_from_camera,
                                render_token=frame.token, acquired_at=frame.acquired_at)
            if frame.rgb is not None:
                save_ppm(output/f'camera_{index}.ppm', frame.rgb)
            metrics['cameras'][frame.camera_id] = {
                'world_from_camera_ros': frame.world_from_camera.tolist(),
                'intrinsics': frame.intrinsics.tolist(), 'render_token': frame.token,
                'workspace_points': len(cloud),
                'sdk_projection_max_m': float(projection_error.max()),
                'sdk_pointcloud_unordered_nn_max_m': float(unordered_error.max())}
            assert len(cloud) > 100, f'Camera does not cover objects: {frame.camera_id}'
            assert projection_error.max() < 1e-5
            assert unordered_error.max() < 1e-5
        save_ply(output/'fused.ply', fuse_clouds(clouds))
        metrics['synchronization'] = rig.status
        assert rig.capture() is None, 'Unchanged frames must not be fresh'
        print(json.dumps(metrics, indent=2), flush=True)
        write_result(args.result, status='PASS', metrics=metrics)
    finally:
        rig.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args()
    exit_code = 0
    try:
        main(args)
    except Exception as exc:
        fail_result(args.result, exc)
        exit_code = 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
