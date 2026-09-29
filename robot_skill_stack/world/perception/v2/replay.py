"""Replay the exact saved V2 mask result through geometry, without an ML worker."""
from types import SimpleNamespace
import numpy as np
from robot_skill_stack.world.perception.frame import PerceptionFrame
from robot_skill_stack.world.perception.v2.settings import V2Config
from robot_skill_stack.world.perception.v2.pipeline import FrameProcessor
from robot_skill_stack.world.perception.v2.wire import decode_response


def replay_saved(data, settings, path):
    frame = PerceptionFrame(int(data['frame_id']), data['rgb'], data['depth'], data['intrinsics'],
                            data['world_from_camera'], float(data['timestamp']))
    segmentation = decode_response(data['segmentation'].tobytes(), frame.frame_id, frame.depth.shape)
    d = dict(settings['discovery'])
    for key in ('workspace_min', 'workspace_max'):
        d[key] = np.array(d[key])
    processor = FrameProcessor(SimpleNamespace(**d), SimpleNamespace(**settings['primitives']), V2Config(**settings['v2']))
    result = processor.process(frame, segmentation)
    return dict(capture=str(path), source='rgbd_yolo_v2', mode='saved masks; no model inference rerun',
                model=segmentation.model, candidates=result.diagnostics)
