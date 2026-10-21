"""Offline latency/accuracy tradeoff on calibrated native capture, no publishing."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, fuse_clouds, project_depth
from robot_skill_stack.world.perception_multiview.fitting import SurfaceFitter
from robot_skill_stack.world.perception_multiview.segmentation import cluster_points
from tests.support.multiview_research import ResearchObject, metric_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    frames = []
    for index in range(2):
        with np.load(args.capture/f'camera_{index}.npz') as data:
            frames.append(DepthFrame(str(index), tuple(data['render_token']), data['depth'],
                data['intrinsics'], data['world_from_camera'], float(data['acquired_at'])))
    scene = load_scene_config('config/scenes/multiview_playground.toml').scene
    truth = [ResearchObject('box', 'cube', np.array([.38, -.18, .025]), np.array([.05, .04, .05]), np.deg2rad(25)),
             ResearchObject('sphere', 'sphere', np.array([.48, 0, .027]), np.full(3, .054), radius=.027),
             ResearchObject('upright', 'cylinder', np.array([.38, .18, .045]), np.array([.036, .036, .09]),
                            radius=.018, length=.09),
             ResearchObject('lying', 'cylinder', np.array([.58, -.18, .018]), np.array([.09, .036, .036]),
                            radius=.018, length=.09, axis=(np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20)), 0.))]
    rows = []
    for stride, voxel, limit in ((1, .002, 1200), (2, .002, 1200), (3, .002, 1200), (2, .003, 600)):
        elapsed = []
        fitter = SurfaceFitter(max_points=limit)
        for _ in range(3):
            started = time.perf_counter()
            clouds = [crop_workspace(project_depth(frame, stride=stride)[0], scene) for frame in frames]
            fused = fuse_clouds(clouds, voxel=voxel, max_points=12000)
            clusters = cluster_points(fused)
            fits = [fitter.fit(cloud, support_plane_z=scene.support_plane_z) for cloud in clusters]
            elapsed.append((time.perf_counter()-started)*1000)
        metrics = [metric_row(obj, min(fits, key=lambda fit: np.linalg.norm(fit.center-obj.center))) for obj in truth]
        rows.append(dict(stride=stride, voxel_size=voxel, fit_point_limit=limit,
                         processing_ms=float(np.mean(elapsed)), points=len(fused),
                         clusters=len(clusters), correct=sum(r['correct'] for r in metrics),
                         center_error_mm=float(np.mean([r['center_error_mm'] for r in metrics])),
                         objects=metrics))
    result = dict(kind='offline_replay_sampling_probe', rows=rows,
        limitation='One previously captured four-object scene; excludes live acquisition/rendering and does not prove native update rate')
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2))
    print(json.dumps([{key: value for key, value in row.items() if key != 'objects'} for row in rows], indent=2))


if __name__ == '__main__':
    main()
