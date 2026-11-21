"""Shared scene paths and distribution summaries for retained motion diagnostics."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'scenes/multiview_perception_final.usda'
PROFILE = ROOT / 'config/scenes/multiview_perception_final.toml'
EYES = ((1.2, -.85, .95), (-.65, .85, .85))


def distribution(values, scale=1.):
    data = np.asarray(values, float) * scale
    data = data[np.isfinite(data)]
    if not len(data):
        return dict(count=0)
    return dict(count=len(data), mean=float(data.mean()),
        **{key: float(np.percentile(data, percentile))
           for key, percentile in [('p50', 50), ('p95', 95), ('p99', 99)]},
        worst=float(data.max()))
