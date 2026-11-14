"""Replay one trusted external capture; compare frozen original/current encoders.

Run from the repository root with NumPy 1.26.4. Refuses existing output paths.
The archived camera input is verified against the previously published manifest.
"""
import argparse
from functools import partial
import gc
import hashlib
import io
import json
from pathlib import Path
import pickle
import time
import tracemalloc

import numpy as np

from robot_skill_stack.runtime.process_worker import _TransportPickler, _restore_readonly_array
from robot_skill_stack.world.spatial_mapping.runtime import MappingProcessor
from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractionSettings
from robot_skill_stack.integrations.isaac.config import load_scene_config


class OriginalPickler(_TransportPickler):
    def reducer_override(self, value):
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            if value.dtype.hasobject:
                raise TypeError('Observation transport requires numeric, non-object arrays')
            return _restore_readonly_array, (value.tobytes(), value.dtype, value.shape)
        return super().reducer_override(value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Refusing existing output')
    folder=Path('outputs/motion-planning-v5/autonomous-admission-20261114/admitted-0/attempt-1')
    manifest=json.loads(Path('research/rgbd_box_geometry/evidence/autonomous_admission/raw-manifest.json').read_text())['files']
    raw=(folder/'accepted_mapping_input.pickle').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest['admitted-0/attempt-1/accepted_mapping_input.pickle']
    payload=pickle.loads(raw);cfg=load_scene_config(folder/'runtime_profile.toml')
    processor=MappingProcessor(cfg.spatial_mapping,clock=partial(float,payload.frames[0].acquired_at+.1),
        obstacle_settings=ObstacleExtractionSettings(64,0.),reconstruction_scene=cfg.scene)
    result=processor.compute(payload)
    assert result.accepted and result.inferred.status=='qualified'
    def encode(cls,value):
        stream=io.BytesIO();cls(stream,protocol=5).dump(value);return stream.getvalue()
    rows=[]
    for kind,value in [('input',payload),('completion',result)]:
        for trial in range(8):
            wires=[]
            for label,cls in [('original',OriginalPickler),('optimized',_TransportPickler)]:
                gc.collect();started=time.perf_counter();data=encode(cls,value)
                rows.append(dict(kind=kind,variant=label,trial=trial,seconds=time.perf_counter()-started,
                    bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
                wires.append(data)
            assert wires[0]==wires[1]
        for label,cls in [('original',OriginalPickler),('optimized',_TransportPickler)]:
            gc.collect();tracemalloc.start();encode(cls,value)
            _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            rows.append(dict(kind=kind,variant=label,peak_traced_bytes=peak))
    args.output.write_text(json.dumps(dict(input_sha256=hashlib.sha256(raw).hexdigest(),rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
