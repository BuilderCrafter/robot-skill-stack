"""Author the selected cameras into a separate, manually editable USD layer.

This research authoring utility never starts Isaac or changes runtime camera
poses. Original and historical experimental scenes are inherited unchanged.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from tests.support.multiview_final_research import camera_layouts, fixture_geometry_errors
from tests.support.multiview_iteration2 import box, sphere, object_payload


def camera_matrix(layout, index):
    existing = None if layout.world_from_cameras is None else layout.world_from_cameras[index]
    if existing is not None:
        return np.asarray(existing, float)
    eye, target = np.asarray(layout.eyes[index]), np.asarray(layout.targets[index])
    forward = target-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((right, -up, forward))
    transform[:3, 3] = eye
    return transform


def tuple_text(values):
    return '('+', '.join(f'{float(value):.12g}' for value in values)+')'


def create(layout_name, destination, config_path, metadata_path):
    layout = next(value for value in camera_layouts() if value.name == layout_name)
    if any(path.exists() for path in (destination, config_path, metadata_path)):
        raise FileExistsError('Use new paths; existing demo assets are preserved')
    matrices = [camera_matrix(layout, index) for index in range(2)]
    objects = (
        box('Square', (.36, -.16), (.055, .055, .055), np.deg2rad(25)),
        box('Rectangle', (.54, .16), (.068, .042, .048), np.deg2rad(-33)),
        sphere('SmallSphere', (.36, .16), .022),
        sphere('LargeSphere', (.54, -.16), .030),
    )
    if fixture_geometry_errors(objects):
        raise ValueError(fixture_geometry_errors(objects))
    parent_scene = Path(__file__).resolve().parents[1]/'scenes/multiview_playground.usda'
    parent_reference = os.path.relpath(parent_scene, destination.resolve().parent)
    lines = ['#usda 1.0', '(', f'    subLayers = [@{parent_reference}@]',
        '    metersPerUnit = 1', '    upAxis = "Z"', ')', '', 'over "World"', '{']
    for name, transform in zip(('PerceptionCamera', 'PerceptionCameraSide'), matrices):
        usd = transform @ np.diag([1., -1., -1., 1.])
        matrix = '('+', '.join(tuple_text(row) for row in usd.T)+')'
        lines += [f'    over "{name}"', '    {', f'        matrix4d xformOp:transform = {matrix}',
            '        uniform token[] xformOpOrder = ["xformOp:transform"]', '    }']
    lines += ['    over "MultiViewObjects" (active = false) {}',
        '    def Xform "FinalPerceptionObjects"', '    {']
    colors = ((.85,.30,.15), (.20,.45,.90), (.20,.72,.35), (.90,.68,.15))
    for obj, color in zip(objects, colors):
        schema = 'Cube' if obj.shape == 'cube' else 'Sphere'
        lines += [f'        def {schema} "{obj.name}"', '        {']
        if obj.shape == 'cube':
            lines += ['            double size = 1', f'            double3 xformOp:translate = {tuple_text(obj.center)}',
                f'            float xformOp:rotateZ = {np.rad2deg(obj.yaw):.12g}',
                f'            float3 xformOp:scale = {tuple_text(obj.size)}',
                '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]']
        else:
            lines += [f'            double radius = {obj.radius:.12g}',
                f'            double3 xformOp:translate = {tuple_text(obj.center)}',
                '            uniform token[] xformOpOrder = ["xformOp:translate"]']
        lines += [f'            color3f[] primvars:displayColor = [{tuple_text(color)}]', '        }']
    lines += ['    }', '}', '']
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(lines))
    original_config = Path(__file__).resolve().parents[1]/'config/scenes/multiview_playground.toml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(f'# Opt-in cube/sphere demo; open {destination} first.\n'
                           +original_config.read_text())
    record = {**asdict(layout), 'scene': str(destination), 'config': str(config_path),
        'world_from_cameras': [matrix.tolist() for matrix in matrices],
        'objects': [object_payload(obj) for obj in objects],
        'camera_prims': ['/World/PerceptionCamera', '/World/PerceptionCameraSide'],
        'authorship': 'Ordinary inherited USD Camera prims; only this experimental layer overrides poses.',
        'intrinsics': 'Inherited unchanged from historical experimental scene; native audit records actual K.',
        'runtime': 'Actual saved transforms read every capture; this utility is not called by runtime startup.'}
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(record, indent=2)+'\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', required=True)
    parser.add_argument('--scene', type=Path, default=Path('scenes/multiview_perception_final.usda'))
    parser.add_argument('--config', type=Path, default=Path('config/scenes/multiview_perception_final.toml'))
    parser.add_argument('--metadata', type=Path, default=Path('docs/assets/multiview_final/selected-layout.json'))
    args = parser.parse_args()
    record = create(args.layout, args.scene, args.config, args.metadata)
    print(json.dumps(dict(layout=record['name'], eyes=record['eyes'], scene=record['scene'])))
