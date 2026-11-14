"""Portable readers of externally supplied, hash-bound trusted sensor archives."""
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import tomllib

from . import reference as ref

PACKAGE=Path(__file__).resolve().parent
CASES={'static':'static/frozen_rgbd','v43':'v43/static'}
SENSOR_FILES=('accepted_mapping_input.pickle','camera_frames.npz','camera_metadata.json',
              'runtime_profile.toml','occupancy.json','occupancy.npz')


def expected_inputs():
    return json.loads((PACKAGE/'external-inputs.json').read_text())['files']


def load_capture(root,case):
    root=Path(root)
    expected=expected_inputs()
    folder=root/CASES[case]
    hashes={}
    for name in SENSOR_FILES:
        relative=f'{CASES[case]}/{name}'
        actual=ref.digest(folder/name)
        if actual!=expected[relative]:
            raise ValueError(f'Archived input SHA256 mismatch: {relative}')
        hashes[relative]=actual
    # Inputs are trusted historical local pickles, verified above, not uploads.
    payload=pickle.loads((folder/'accepted_mapping_input.pickle').read_bytes())
    if payload.ground_truth is not None or len(payload.frames)!=2:
        raise ValueError('Expected a camera-only synchronized pair')
    if any(f.depth.shape!=(480,640) for f in payload.frames):
        raise ValueError('Unexpected camera resolution')
    config=tomllib.loads((folder/'runtime_profile.toml').read_text())['scene']
    scene=SimpleNamespace(**{k:config[k] for k in ('workspace_min','workspace_max','support_plane_z')})
    return payload,scene,hashes
