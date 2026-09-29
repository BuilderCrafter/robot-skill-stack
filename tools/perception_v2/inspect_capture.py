#!/usr/bin/env python3
"""Inspect the exact robot mask and cleaned V2 candidates saved by Capture. No ML/Isaac imports."""
import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from robot_skill_stack.world.perception.v2.pipeline import FrameProcessor
from robot_skill_stack.world.perception.v2.settings import V2Config
from robot_skill_stack.world.perception.v2.self_filter import SelfFilterResult
from robot_skill_stack.world.perception.v2.wire import decode_response
from robot_skill_stack.world.perception.frame import PerceptionFrame
from tools.perception_v2_data.common import write_png
from tools.perception_v2.benchmark import overlay
from types import SimpleNamespace


def inspect(path, out):
    path, out = Path(path), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new/empty inspection output directory')
    with np.load(path, allow_pickle=False) as data:
        settings = json.loads(str(data['settings']))
        if settings.get('source') != 'rgbd_yolo_v2' or settings.get('filter_revision') != 1:
            raise ValueError('Use a V2 Capture saved after applying the robot-filter patch')
        frame = PerceptionFrame(int(data['frame_id']), data['rgb'], data['depth'], data['intrinsics'],
                                data['world_from_camera'], float(data['timestamp']))
        segmentation = decode_response(data['segmentation'].tobytes(), frame.frame_id, frame.depth.shape)
        robot = SelfFilterResult(data['robot_mask'], data['robot_model_depth'], settings['robot_self_filter'])
        d = dict(settings['discovery'])
        d['workspace_min'], d['workspace_max'] = np.array(d['workspace_min']), np.array(d['workspace_max'])
        result = FrameProcessor(SimpleNamespace(**d), SimpleNamespace(**settings['primitives']),
                                V2Config(**settings['v2'])).process(frame, segmentation, robot)
    out.mkdir(parents=True, exist_ok=True)
    marked = frame.rgb.copy()
    marked[robot.mask] = (.35*marked[robot.mask]+.65*np.array([255, 50, 70])).astype(np.uint8)
    filtered = frame.rgb.copy(); filtered[robot.mask] = [35, 35, 35]
    candidates = [dict(mask=c.mask, label=p.label) for c, p in zip(result.candidates, result.predictions)]
    raw = [dict(mask=i.mask, label=i.label) for i in segmentation.instances]
    images = {'rgb': frame.rgb, 'robot_removed_red': marked, 'rgb_filtered_debug_only': filtered,
              'raw_yolo': overlay(frame.rgb, raw), 'accepted': overlay(frame.rgb, candidates),
              'robot_mask': np.repeat((robot.mask.astype(np.uint8)*255)[..., None], 3, axis=2)}
    for name, image in images.items():
        write_png(out/f'{name}.png', image)
    report = dict(capture=str(path.resolve()), robot_self_filter=robot.metadata, candidates=result.diagnostics,
                  tracks_at_save=settings.get('tracks_at_save', []),
                  note='Saved robot mask + saved neural results. No new model inference or robot rendering.')
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    titles = [('rgb', 'Original RGB'), ('robot_removed_red', 'Removed robot surface pixels (red)'),
              ('raw_yolo', 'Raw YOLO masks'), ('accepted', 'Accepted object masks')]
    cards = ''.join(f'<div><h2>{title}</h2><img src="{name}.png"></div>' for name, title in titles)
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>V2 filter review</title>'
                                  '<style>body{background:#18212b;color:#eee;font:16px sans-serif}'
                                  'main{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{max-width:100%}</style>'
                                  '<h1>V2 same-frame filter review</h1><p>Blackened RGB is debug only; YOLO receives original RGB. '
                                  'See report.json for rejection reasons and timing.</p><main>'+cards+'</main>')
    print('Review:', out/'review.html')
    print('Self-filter:', robot.metadata)
    for c in result.diagnostics:
        print(c['candidate_index'], c['source'], c['label'], '->', c.get('final_label'), c['rejected'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        inspect(args.capture, args.out)
    except Exception as exc:
        print(f'Inspection failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
