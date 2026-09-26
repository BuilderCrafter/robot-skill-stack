from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np


def write_result(path, *, status, metrics=None, error=None):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "metrics": metrics or {},
        "error": error,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fail_result(path, exc):
    write_result(
        path,
        status="FAIL",
        error=f"{type(exc).__name__}: {exc}",
    )
    traceback.print_exc()


def save_ppm(path, image):
    image = np.asarray(image)[..., :3].astype(np.uint8)
    h, w = image.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(image.tobytes())


def save_pgm(path, image):
    image = np.asarray(image).astype(np.uint8)
    h, w = image.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(image.tobytes())
