"""Replay the exact saved V2 mask result through geometry, without an ML worker."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from robot_skill_stack.world.perception.frame import PerceptionFrame
from robot_skill_stack.world.perception.v2.settings import V2Config
from robot_skill_stack.world.perception.v2.pipeline import FrameProcessor
from robot_skill_stack.world.perception.v2.self_filter import SelfFilterResult
from robot_skill_stack.world.perception.v2.wire import decode_response


def replay_saved(data, settings, path):
    frame = PerceptionFrame(int(data['frame_id']), data['rgb'], data['depth'], data['intrinsics'],
                            data['world_from_camera'], float(data['timestamp']))
    segmentation = decode_response(data['segmentation'].tobytes(), frame.frame_id, frame.depth.shape)
    d = dict(settings['discovery'])
    for key in ('workspace_min', 'workspace_max'):
        d[key] = np.array(d[key])
    config = V2Config(**settings['v2'])
    robot = None
    if settings.get('filter_revision') == 1:
        robot = SelfFilterResult(data['robot_mask'], data['robot_model_depth'], settings['robot_self_filter'])
    else:
        config = replace(config, robot_self_filter=False)
    processor = FrameProcessor(SimpleNamespace(**d), SimpleNamespace(**settings['primitives']), config)
    result = processor.process(frame, segmentation, robot)
    return dict(capture=str(path), source='rgbd_yolo_v2', mode='saved masks; no model inference rerun',
                model=segmentation.model, candidates=result.diagnostics, robot_self_filter=result.self_filter.metadata,
                note='Legacy captures cannot replay robot removal' if robot is None else 'Saved robot mask; mesh rendering not rerun')
