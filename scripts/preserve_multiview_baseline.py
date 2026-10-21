"""Export an isolated iteration-one package and instrument its unchanged scenes.

Run with the established Isaac Python wrapper. Generated files stay in outputs/;
only import names and after-timing diagnostic exports differ from the Git source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'robot_skill_stack.world.perception_multiview'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def substitute(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected exactly one instrumentation anchor: {old!r}')
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', default='42e3258')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/multiview-iteration2/baseline_snapshot')
    args = parser.parse_args()
    commit = git('rev-parse', '--verify', args.ref+'^{commit}').decode().strip()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package = 'robot_skill_stack/world/perception_multiview/'
    paths = git('ls-tree', '-r', '--name-only', commit, '--', package).decode().splitlines()
    records = []
    for path in paths:
        original = git('show', f'{commit}:{path}')
        exported = original.decode().replace(PREFIX, 'baseline_multiview').encode()
        target = output/'baseline_multiview'/Path(path).relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(exported)
        records.append(dict(path=path, original_sha256=hashlib.sha256(original).hexdigest(),
                            exported_sha256=hashlib.sha256(exported).hexdigest()))
    benchmark_path = 'scripts/benchmark_multiview_detection.py'
    original = git('show', f'{commit}:{benchmark_path}')
    source = original.decode()
    source = substitute(source, 'from robot_skill_stack.integrations.isaac.config import load_scene_config',
        f'import sys\nsys.path.insert(0, {str(output)!r})\n'
        'from robot_skill_stack.integrations.isaac.config import load_scene_config')
    source = source.replace(PREFIX, 'baseline_multiview')
    source = substitute(source, '    rig = create_cameras(CAMERAS)', '''    rig = create_cameras(CAMERAS)
    original_capture = rig.capture
    latest_frames = [None]
    def capture_with_diagnostics():
        frames = original_capture()
        if frames is not None:
            latest_frames[0] = frames
        return frames
    rig.capture = capture_with_diagnostics''')
    source = substitute(source, '                    if repeat == 0:', '''                    if repeat == 0:
                        # Disk/projection diagnostics happen AFTER all timed samples.
                        from baseline_multiview.clouds import project_depth, crop_workspace, voxelize, fuse_clouds
                        clouds = []
                        for index, frame in enumerate(latest_frames[0]):
                            destination = output/'full_clouds'
                            destination.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(destination/f'{trial}_camera_{index}.npz',
                                depth=frame.depth, intrinsics=frame.intrinsics,
                                world_from_camera=frame.world_from_camera,
                                acquired_at=frame.acquired_at, token=str(frame.token), rgb=frame.rgb)
                            points, _ = project_depth(frame)
                            points = voxelize(crop_workspace(points, cfg.scene), max_points=12000)
                            clouds.append(points)
                            save_ply(destination/f'{trial}_camera_{index}.ply', points)
                        save_ply(destination/f'{trial}_fused.ply', fuse_clouds(clouds, max_points=12000))''')
    (output/'benchmark_original.py').write_text(source)
    for path in ('config/scenes/multiview_playground.toml', 'docs/multiview_rgbd.md'):
        (output/Path(path).name).write_bytes(git('show', f'{commit}:{path}'))
    manifest = dict(commit=commit, tree=git('rev-parse', commit+'^{tree}').decode().strip(),
        package_files=records, benchmark_original_sha256=hashlib.sha256(original).hexdigest(),
        benchmark_exported_sha256=hashlib.sha256(source.encode()).hexdigest(),
        instrumentation='Import isolation; retain latest fresh frames; full-cloud export after timing only.',
        previous_detection_artifacts='outputs/multiview-detection-01',
        previous_geometry_artifacts='outputs/multiview-geometry-final')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(commit=commit, output=str(output), files=len(records))))


if __name__ == '__main__':
    main()
