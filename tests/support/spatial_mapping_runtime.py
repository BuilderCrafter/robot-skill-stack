"""Shared fixture setup and diagnostics for native automatic mapping tests."""
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT/'scenes/multiview_perception_final.usda'
PROFILE = ROOT/'config/scenes/multiview_perception_final.toml'
EYES = ((1.2, -.85, .95), (-.65, .85, .85))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return serial(asdict(value))
    if hasattr(value, 'items'):
        return {str(key): serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    return value


def hashes():
    files = [SCENE, PROFILE, Path(__file__),
        ROOT/'scenes/multiview_playground.usda', ROOT/'scenes/playground.usd']
    files += [ROOT/path for path in (
        'robot_skill_stack/integrations/isaac/bootstrap.py',
        'robot_skill_stack/world/model/updater.py',
        'robot_skill_stack/world/perception_multiview/provider.py',
        'robot_skill_stack/integrations/isaac/manipulation/franka_backend.py',
        'robot_skill_stack/world/spatial_mapping/rgbd.py',
        'robot_skill_stack/world/spatial_mapping/occupancy.py',
        'robot_skill_stack/integrations/isaac/spatial_mapping/robot_filter.py')]
    files += [ROOT/path for path in (
        'robot_skill_stack/integrations/isaac/observation_runtime.py',
        'robot_skill_stack/runtime/worker.py',
        'robot_skill_stack/runtime/process_worker.py',
        'robot_skill_stack/runtime/worker_process.py',
        'robot_skill_stack/world/spatial_mapping/runtime.py',
        'robot_skill_stack/world/spatial_mapping/geometry.py',
        'robot_skill_stack/world/perception_multiview/shared_frames.py') if (ROOT/path).exists()]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def runtime_profile(output):
    """Session-only diagnostic configuration; never rewrite the shipped TOML."""
    path = Path(output)/'runtime_profile.toml'
    path.write_text(PROFILE.read_text()+'''\n[spatial_mapping]
enabled = true
source = "rgbd"
bounds_min = [-0.3, -0.6, -0.02]
bounds_max = [0.85, 0.6, 1.12]
voxel_size = 0.01
camera_prim_paths = ["/World/PerceptionCamera", "/World/PerceptionCameraSide"]
stride = 8
max_rays = 1024
automatic_robot_filter = true
update_hz = 2.0
max_frame_age = 1.0
stale_after = 1.0
evidence_ttl = 5.0
''')
    return path
