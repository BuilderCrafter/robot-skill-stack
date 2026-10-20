"""Native recovery and conservative rejection at the old narrow-gap boundary."""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False})

import argparse
from dataclasses import replace
import json
from pathlib import Path

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.world.multiview import create_cameras
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, fuse_clouds, project_depth, save_ply
from tests.support.multiview_iteration2 import score_observations
from tests.support.multiview_native import CAMERAS, ROOT, open_experiment, populate, research_objects, set_viewpoint
from tests.support.multiview_research import ResearchObject
from tests.support.results import fail_result, write_result


class BorrowedRig:
    def __init__(self, rig): self.rig, self.frames = rig, None
    def capture(self):
        frames = self.rig.capture()
        if frames is not None:
            self.frames = frames
        return frames
    @property
    def status(self): return self.rig.status
    def close(self): pass


def main(args):
    output = Path(args.artifacts_dir) if args.artifacts_dir else Path(args.result).parent/'close_object_recovery'
    output.mkdir(parents=True, exist_ok=True)
    world = open_experiment(simulation_app)
    cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
    rig = create_cameras(CAMERAS)
    base = research_objects()
    layout = ((.34, .10), (.50, -.18), (.35, -.18), (.53, .20))
    originals = [replace(obj, center=np.r_[xy, obj.center[2]]) for obj, xy in
                 zip((base[1], base[2], base[4], base[7]), layout)]
    rows, counts, recovered = [], [], 0
    try:
        for _ in range(60):
            world.step(render=True)
        for arrangement in range(2):
            if arrangement:
                set_viewpoint(rig.cameras[CAMERAS[1]], (.25, -.70, .55))
            for repeat in range(2):
                rng = np.random.default_rng(1000*arrangement+300+repeat)
                shift = np.r_[rng.uniform(-.004, .004, 2), 0.]
                objects = [replace(obj, center=obj.center+shift) for obj in originals]
                occluder = ResearchObject('Occluder', 'cube', np.array([.32, -.14, .10]),
                                          np.array([.15, .035, .20]))
                populate(objects+[occluder])
                source = MultiViewProvider(BorrowedRig(rig), cfg.scene)
                previous = {}
                try:
                    for index in range(5):
                        for _ in range(8):
                            world.step(render=True)
                        batch = source.observe()
                        if index < 2:
                            assert not batch.observations, 'Three distinct frames must confirm objects'
                            continue
                        observations = batch.observations
                        clouds = [source.get_point_cloud(obs.object_id) for obs in observations]
                        selected_rows, selected_counts, identities = score_observations(objects, observations, clouds,
                            method='improved', trial=f'{arrangement}_{repeat}', frame_index=index,
                            previous_ids=previous)
                        rows.extend(selected_rows)
                        counts.append(selected_counts)
                        (output/'rows.json').write_text(json.dumps(rows, indent=2))
                        (output/'counts.json').write_text(json.dumps(counts, indent=2))
                        # The frozen research method recovered three old misses.
                        # The last boundary case may remain conservatively absent;
                        # it must never become a duplicate or invented known shape.
                        required = 3 if (arrangement, repeat) == (1, 1) else 4
                        assert len(observations) >= required, source.diagnostics
                        assert all(row['correct'] and row['center_error_mm'] < 10
                                   for row in selected_rows if row['detected']), selected_rows
                        assert all((arrangement, repeat) == (1, 1) and row['object'] == originals[2].name
                                   for row in selected_rows if not row['detected']), selected_rows
                        assert selected_counts['missing'] <= 4-required, selected_counts
                        assert not any(selected_counts[key] for key in
                            ('false_positives', 'mixed_accepted_clouds', 'false_splits',
                             'duplicate_ids', 'identity_switches')), selected_counts
                        previous = identities
                        assert all(obs.timestamp == batch.timestamp for obs in observations)
                        repeated = source.observe()
                        assert not repeated.observations and repeated.timestamp is None
                    recovered += next(row['detected'] for row in selected_rows if row['object'] == originals[2].name)
                    directory = output/f'{arrangement}_{repeat}'
                    directory.mkdir(parents=True, exist_ok=True)
                    clouds = []
                    for index, frame in enumerate(source.rig.frames):
                        np.savez_compressed(directory/f'camera_{index}.npz', depth=frame.depth,
                            intrinsics=frame.intrinsics, world_from_camera=frame.world_from_camera,
                            acquired_at=frame.acquired_at, token=frame.token, rgb=frame.rgb)
                        points = crop_workspace(project_depth(frame)[0], cfg.scene)
                        save_ply(directory/f'camera_{index}.ply', points)
                        clouds.append(points)
                    save_ply(directory/'fused.ply', fuse_clouds(clouds, max_points=12000))
                    (directory/'diagnostics.json').write_text(json.dumps(source.diagnostics, indent=2))
                finally:
                    source.close()
        write_result(args.result, status='PASS', metrics=dict(scenes=4, confirmed_samples=len(counts),
            expected_per_sample=4, recovered_original_misses=recovered,
            retained_boundary_misses=4-recovered, duplicate_or_mixed_observations=0,
            identity_switches=0, three_frame_confirmation=True, unchanged_frames_publish_nothing=True,
            maximum_center_error_mm=max(row['center_error_mm'] for row in rows if row['detected']),
            robot_commands=0))
    finally:
        rig.close()


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
