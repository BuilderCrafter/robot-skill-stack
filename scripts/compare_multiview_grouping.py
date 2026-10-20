"""Research-only grouping probe; not a second production perception pipeline."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception.discovery import DepthObjectDiscoverer
from robot_skill_stack.world.perception.frame import PerceptionFrame
from robot_skill_stack.world.perception.localizer import RgbdLocalizer
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, fuse_clouds, project_depth
from robot_skill_stack.world.perception_multiview.segmentation import cluster_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path, help='Directory containing camera_0.npz and camera_1.npz')
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    scene = load_scene_config('config/scenes/multiview_playground.toml').scene
    frames = []
    for index in range(2):
        with np.load(args.capture/f'camera_{index}.npz') as data:
            frames.append(DepthFrame(str(index), tuple(data['render_token']), data['depth'],
                data['intrinsics'], data['world_from_camera'], float(data['acquired_at'])))
    started = time.perf_counter()
    candidates = []
    for frame in frames:
        detector = DepthObjectDiscoverer(RgbdLocalizer(frame.intrinsics),
            support_plane_z=scene.support_plane_z, workspace_min=scene.workspace_min,
            workspace_max=scene.workspace_max)
        rgb = np.zeros((*frame.depth.shape, 3), np.uint8)
        candidates.append(detector.discover(PerceptionFrame(1, rgb, frame.depth, frame.intrinsics,
                                                           frame.world_from_camera, frame.acquired_at)))
    edges = sorted((float(np.linalg.norm(a.position-b.position)), ai, bi)
                   for ai, a in enumerate(candidates[0]) for bi, b in enumerate(candidates[1])
                   if np.linalg.norm(a.position-b.position) < .08)
    used_a, used_b, pairs = set(), set(), []
    for distance, ai, bi in edges:
        if ai not in used_a and bi not in used_b:
            used_a.add(ai)
            used_b.add(bi)
            pairs.append([ai, bi, distance])
    association_ms = (time.perf_counter()-started)*1000
    started = time.perf_counter()
    clouds = [crop_workspace(project_depth(frame)[0], scene) for frame in frames]
    fused = fuse_clouds(clouds)
    clusters = cluster_points(fused)
    global_ms = (time.perf_counter()-started)*1000
    result = dict(kind='offline_grouping_probe', status='COMPLETED',
        strategy_a=dict(candidates_per_view=list(map(len, candidates)), associations=pairs,
            unmatched=[len(candidates[0])-len(used_a), len(candidates[1])-len(used_b)],
            ambiguous_a=sum(sum(ai == edge[1] for edge in edges) > 1 for ai in range(len(candidates[0]))),
            ms=association_ms, note='V1 regions used as research candidates only; no primitive classification'),
        strategy_b=dict(clusters=len(clusters), cluster_points=list(map(len, clusters)), ms=global_ms),
        limitation='One captured four-object scene cannot establish clutter robustness or fair cross-pipeline latency')
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
