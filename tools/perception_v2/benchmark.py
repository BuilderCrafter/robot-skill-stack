#!/usr/bin/env python3
"""Compare native V1 and V2 on independent RAW SDG frames. Requires depth, not Isaac."""
from pathlib import Path
import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import html
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # Do not inject .deps into an arbitrary interpreter.
import numpy as np
from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception.factory import build_perception_provider
from robot_skill_stack.world.perception.v2.client import VisionClient
from robot_skill_stack.world.perception.v2.pipeline import FrameProcessor
from robot_skill_stack.world.model.primitives import PrimitiveShape
from tools.perception_v2.dataset import list_samples, load_sample, sha256
from tools.perception_v2.metrics import Metrics
from tools.perception_v2_data.common import write_png


class FrameCamera:
    def __init__(self, frame):
        self.frame = frame

    def get_intrinsics(self):
        return self.frame.intrinsics


def candidates_output(candidates, labels):
    return [dict(mask=c.mask, label=label, position=c.position, size=c.size,
                 geometry_known=c.metadata.get('geometry') is not None and c.metadata['geometry'].shape != PrimitiveShape.UNKNOWN)
            for c, label in zip(candidates, labels)]


def run_v1(frame, config):
    config = replace(config, perception=replace(config.perception, type='geometry_v1'))
    provider = build_perception_provider(FrameCamera(frame), config)
    candidates, labels = [], []
    for c in provider.discoverer.discover(frame):
        g = provider.primitive_estimator.estimate(c)
        c.metadata['geometry'] = g
        if 'center' in g.metadata:
            c.position = np.asarray(g.metadata['center'], float)
            c.size = np.asarray(g.metadata['size_world'], float)
            c.spawnable = provider.discoverer.is_spawnable(c.size)
        if c.spawnable:
            candidates.append(c)
            labels.append(None if g.shape == PrimitiveShape.UNKNOWN else g.shape.value)
    return candidates_output(candidates, labels)


def overlay(rgb, predictions):
    image = rgb.copy()
    colors = {'cube': [255, 85, 65], 'sphere': [50, 170, 255], 'cylinder': [60, 230, 110], None: [255, 220, 70]}
    for pred in predictions:
        m = pred['mask']
        edge = m.copy()
        edge[1:-1, 1:-1] &= ~(m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
        image[m] = (.65*image[m]+.35*np.array(colors[pred['label']])).astype(np.uint8)
        image[edge] = colors[pred['label']]
    return image


def write_reports(out, report, images):
    out = Path(out)
    (out/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    fields = ['mode', 'class', 'gt', 'predicted', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1']
    with (out/'per_class.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, values in report['metrics'].items():
            for label, row in values['per_class'].items():
                writer.writerow(dict(mode=name, **{'class': label}, **row))
    def fmt(value):
        return 'n/a' if value is None else f'{value:.3f}'
    lines = ['# Perception comparison', '', f"Split: **{report['split']}**; frames: **{report['frames']}**; mask IoU threshold: **{report['iou']}**.", '',
             '| Mode | Target precision | Target recall | Target F1 | Localization recall | Class accuracy (localized) | Center error mean (mm) | Mean time (ms) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode, r in report['metrics'].items():
        vals = [r[k] for k in ('named_target_precision', 'named_target_recall', 'named_target_f1', 'localization_recall', 'classification_accuracy_on_localized')]
        vals += [r['center_error_mm']['mean'], r['latency_ms']['mean']]
        lines.append('| '+mode+' | '+' | '.join(map(fmt, vals))+' |')
    lines += ['', *report['notes'], '', 'Per-class TP/FP/FN: `per_class.csv`; confusion/counts/settings: `metrics.json`.',
              'Color key: red=cube, blue=sphere, green=cylinder, yellow=unknown. Open `review.html` for overlays.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    page = '<!doctype html><meta charset="utf-8"><title>Perception comparison</title><style>body{background:#18212b;color:#eee;font:16px sans-serif}img{max-width:100%}td{vertical-align:top}</style><h1>Same-frame comparison</h1><p>Red cube / blue sphere / green cylinder / yellow unknown. Independent scenes, not tracking.</p>'
    for name, stages in images:
        page += '<h2>'+html.escape(name)+'</h2><table><tr>'
        for stage, relative in stages:
            page += f'<td>{html.escape(stage)}<br><img src="{html.escape(relative)}"></td>'
        page += '</tr></table>'
    (out/'review.html').write_text(page, encoding='utf-8')


def evaluate(data, profile, split='test', out=None, limit=None, iou=.5, v1_only=False, previews=12, client=None):
    data, profile = Path(data).resolve(), Path(profile).resolve()
    config = load_scene_config(profile)
    if split == "train":
        print("WARNING: train split is in-sample, not generalization evidence.", file=sys.stderr)
    samples, manifest = list_samples(data, split)
    if abs(manifest['settings']['support_z']-config.perception.discovery.support_plane_z) > 1e-6:
        raise ValueError('Profile support plane differs from the dataset; use the matching camera/workspace profile')
    samples = samples if limit is None else samples[:limit]
    if not samples:
        raise ValueError('No samples selected')
    # Health failure aborts the comparison, never silently substitutes GT or V1 for V2.
    client = None if v1_only else client or VisionClient(config.perception.v2)
    worker = None if client is None else client.health()
    # Original SDG bundles do not contain acquisition-time robot poses/meshes.
    # Explicitly report the ablation rather than inventing a robot mask from GT.
    offline_v2 = replace(config.perception.v2, robot_self_filter=False)
    if not v1_only and config.perception.v2.robot_self_filter:
        print('NOTE: SDG benchmark has no robot snapshots. Robot self-filter NOT evaluated; geometry/duplicate gates ARE evaluated.', flush=True)
    processor = FrameProcessor(config.perception.discovery, config.perception.primitives, offline_v2)
    metrics = {'v1': Metrics(iou)}
    if client is not None:
        metrics.update(v2_rgb=Metrics(iou), v2=Metrics(iou))
    out = Path(out or ROOT/'outputs'/'perception_comparison'/datetime.now().strftime('%Y%m%d_%H%M%S'))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new benchmark output directory')
    out.mkdir(parents=True, exist_ok=True)
    images, rows = [], []
    for index, path in enumerate(samples):
        frame, truth, meta = load_sample(data, path)
        t = time.perf_counter()
        v1 = run_v1(frame, config)
        dt1 = time.perf_counter()-t
        stages = [('ground_truth', truth), ('v1', v1)]
        metrics['v1'].add(truth, v1, dt1)
        row = dict(frame_id=frame.frame_id, split=split, gt=len(truth), v1=len(v1), v1_ms=1000*dt1)
        if client is not None:
            t = time.perf_counter()
            segmentation = client.predict(frame.frame_id, frame.rgb)
            if segmentation.model.get('sha256') != worker.get('sha256'):
                raise RuntimeError('Worker model changed during the comparison')
            rgb_time = time.perf_counter()-t
            result = processor.process(frame, segmentation)
            rgb_predictions = [dict(mask=i.mask, label=i.label) for i in segmentation.instances]
            v2 = candidates_output(result.candidates, [p.label for p in result.predictions])
            metrics['v2_rgb'].add(truth, rgb_predictions, rgb_time)
            metrics['v2'].add(truth, v2, rgb_time+result.geometry_s)
            stages += [('v2_rgb', rgb_predictions), ('v2', v2)]
            row.update(v2_rgb=len(rgb_predictions), v2=len(v2), v2_ms=1000*(rgb_time+result.geometry_s),
                       v2_rejected=sum(d.get('rejected') is not None for d in result.diagnostics))
        rows.append(row)
        if index < previews:
            saved = []
            for stage, predictions in stages:
                relative = f'previews/{path.stem}_{stage}.png'
                write_png(out/relative, overlay(frame.rgb, predictions))
                saved.append((stage, relative))
            images.append((path.stem, saved))
        if index % 10 == 0 or index == len(samples)-1:
            print(f'Compared {index+1}/{len(samples)}', flush=True)
    notes = [
        'Robot self-filter is NOT applied: these original SDG bundles lack synchronized robot poses/meshes. Use live captures for self-filter validation.',
        'Named-target metrics require BOTH class and one-to-one mask IoU match. Unknown predictions cannot be a correct named target.',
        'Localization metrics ignore class, include unknown candidates, and count unmatched candidates as non-target predictions (even genuine unlabeled distractors).',
        'Class accuracy is conditional on a spatial match; unknown is incorrect. Always read it together with target recall.',
        'V1 uses its current native discovery/size filters and primitive fitting. V2 uses its current mask/depth gates; rejected/missed objects remain in the denominators.',
        'v2_rgb = model masks/classes BEFORE depth filtering; v2 = localizable candidates supplied to the runtime tracker, including configured unknown-depth fallback.',
        'Ground truth is used ONLY for scoring. RGB/depth/calibration go into perception; GT masks, object positions, dimensions, class IDs never do.',
        'Each frame is an independent randomized scene. Track confirmation is bypassed for BOTH pipelines; these scores do NOT measure ID stability, occlusion recovery, live latency, or grasp success.',
        'Center/size errors are conditional on localized matches, including approximate visible-surface bounds. Counts and elevated/supported breakdowns are in metrics.json.',
        'Times include cold V1 geometry and V2 HTTP/mask processing; model startup is excluded (worker warmed). This is not a simulation FPS or COCO mAP benchmark.',
        'Use val for tuning; use test after choosing settings. Validate on actual camera/manipulation episodes before switching away from V1.',
    ]
    report = dict(split=split, frames=len(samples), iou=iou, dataset=str(data), profile=str(profile),
                  profile_sha256=sha256(profile), manifest_sha256=sha256(data/'manifest.json'),
                  v2_settings=asdict(offline_v2), v2_settings_requested=asdict(config.perception.v2), robot_filter_applied=False, worker=worker, python=sys.version, numpy=np.__version__,
                  metrics={k: v.report() for k, v in metrics.items()}, notes=notes)
    write_reports(out, report, images)
    with (out/'frames.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print((out/'summary.md').read_text(), flush=True)
    print('Report:', out.resolve(), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='RAW primitive_1200_v2, not converted *_yolo')
    parser.add_argument('--profile', type=Path, default=ROOT/'config/scenes/playground_v2.toml')
    parser.add_argument('--split', choices=('val', 'test', 'train'), default='test')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--previews', type=int, default=12)
    parser.add_argument('--v1-only', action='store_true', help='Baseline only while weights are still training')
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or args.previews < 0:
        parser.error('limit must be positive; previews must be nonnegative')
    try:
        evaluate(**vars(args))
    except Exception as exc:
        print(f'Benchmark failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
