"""Session-only camera workload profiles; importing this module needs no Isaac.

The shipped scene/profile stay authoritative and are never written. The native
runner applies the third camera's individual resolution during construction;
the original two cameras retain the shipped 640x480 setting.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'scenes/multiview_perception_final.usda'
PROFILE = ROOT / 'config/scenes/multiview_perception_final.toml'
ORIGINAL_CAMERAS = ('/World/PerceptionCamera', '/World/PerceptionCameraSide')
THIRD = '/World/CameraWorkloadThird'
THIRD_EYE = (.35, 1.10, .95)
THIRD_TARGET = (.15, 0., .45)
MODES = ('A', 'B', 'C', 'D')
THIRD_RESOLUTIONS = ((320, 240), (640, 480))
MAX_RAYS = 1024


def validate_scenario(mode, third_resolution):
    """Validate explicit experiment choices without silently changing them."""
    if mode not in MODES:
        raise ValueError(f'Unknown scenario {mode!r}; expected one of {MODES}')
    resolution = tuple(third_resolution)
    if (any(type(value) is not int for value in resolution)
            or resolution not in THIRD_RESOLUTIONS):
        raise ValueError(f'Third camera resolution must be one of {THIRD_RESOLUTIONS}')
    return resolution


def scenario_camera_paths(mode):
    """Return independent semantic and mapping camera selections."""
    if mode not in MODES:
        raise ValueError(f'Unknown scenario {mode!r}; expected one of {MODES}')
    three = (*ORIGINAL_CAMERAS, THIRD)
    return (three if mode == 'D' else ORIGINAL_CAMERAS,
            three if mode in ('C', 'D') else ORIGINAL_CAMERAS)


def _set_table_values(text, section, values):
    """Replace only requested simple TOML keys, retaining all other settings."""
    pattern = re.compile(r'^\[' + re.escape(section) + r'\][^\n]*\n'
                         r'.*?(?=^\[|\Z)', re.MULTILINE | re.DOTALL)
    match = pattern.search(text)
    table = match.group(0) if match else f'[{section}]\n'
    for key, value in values.items():
        rendered = json.dumps(value)
        line = f'{key} = {rendered}'
        key_pattern = re.compile(r'^' + re.escape(key) + r'\s*=.*$', re.MULTILINE)
        if key_pattern.search(table):
            table = key_pattern.sub(lambda _: line, table)
        else:
            table = table.rstrip() + '\n' + line + '\n'
    table = table.rstrip() + '\n\n'
    return (text[:match.start()] + table + text[match.end():]
            if match else text.rstrip() + '\n\n' + table)


def write_profile(output_dir, mode, third_resolution=(320, 240)):
    """Write a generated native-run profile under the supplied artifact dir.

    ``max_rays`` is a GLOBAL budget shared fairly among mapping cameras, so
    A/B and C/D use the same 1024 selected-ray cap, not 1024 per camera.
    """
    resolution = validate_scenario(mode, third_resolution)
    semantic, mapping = scenario_camera_paths(mode)
    text = PROFILE.read_text()
    original = tomllib.loads(text)
    if (original['world']['provider'] != 'multi_view'
            or tuple(original['multi_view']['camera_prim_paths']) != ORIGINAL_CAMERAS
            or tuple(original['multi_view']['resolution']) != (640, 480)):
        raise ValueError('Shipped two-camera profile no longer matches this benchmark baseline')
    text = _set_table_values(text, 'world', {'update_hz': 5.0})
    text = _set_table_values(text, 'multi_view', {'camera_prim_paths': semantic})
    text = _set_table_values(text, 'motion', {'strategy': 'direct'})
    text = _set_table_values(text, 'spatial_mapping', dict(
        enabled=True, source='rgbd', bounds_min=[-.3, -.6, -.02],
        bounds_max=[.85, .6, 1.12], voxel_size=.01,
        camera_prim_paths=mapping, stride=8, max_rays=MAX_RAYS,
        automatic_robot_filter=True, update_hz=2.0, max_frame_age=1.0,
        stale_after=1.0, evidence_ttl=5.0))
    text = (f'# Generated camera workload scenario {mode}; third resolution '
            f'{resolution[0]}x{resolution[1]} is applied by the native runner.\n' + text)
    tomllib.loads(text)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'runtime_profile.toml'
    if path.resolve() == PROFILE.resolve():
        raise ValueError('Generated profile must not overwrite the shipped profile')
    path.write_text(text)
    return path
