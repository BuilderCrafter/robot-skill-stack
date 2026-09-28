"""PC-side conversion: exact instance masks -> validated YOLO segmentation polygons."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import CLASSES, FORMAT, SPLITS, dump_json, sha256


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.


def raster(poly, shape):
    result = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(result, [np.asarray(poly, np.int32)], 1)
    return result.astype(bool)


def resample(poly, n=1000):
    # Match the pinned Ultralytics 8.3.161 loader: retain every original vertex.
    if len(poly) == n:
        return np.asarray(poly, np.float32).copy()
    closed = np.concatenate((poly, poly[:1]), axis=0)
    knots = np.arange(len(closed))
    if len(closed) < n:
        grid = np.linspace(0., float(knots[-1]), n-len(closed))
        grid = np.insert(grid, np.searchsorted(grid, knots), knots)
    else:
        grid = np.linspace(0., float(knots[-1]), n)
    return np.column_stack([np.interp(grid, knots, closed[:, axis]) for axis in (0, 1)]).astype(np.float32)


def mask_polygon(mask, min_pixels=20):
    """One row per INSTANCE, including disconnected visibility via zero-width bridges.

    Exact renderer masks are retained; bad polygon approximations reject the whole
    image instead of silently turning part of a target into background.
    """
    m = np.asarray(mask, np.uint8)
    if int(m.sum()) < min_pixels:
        raise ValueError("visible target too small for a reliable training polygon")
    contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rings = [c.reshape(-1, 2).astype(np.float32) for c in contours if len(c) >= 3]
    if not rings:
        raise ValueError("no polygonal contour")
    rings.sort(key=lambda p: abs(cv2.contourArea(p)), reverse=True)
    poly = rings.pop(0)
    for ring in rings:
        distance = ((poly[:, None, :] - ring[None, :, :])**2).sum(axis=2)
        i, j = np.unravel_index(int(distance.argmin()), distance.shape)
        loop = np.concatenate((ring[j:], ring[:j+1]))
        poly = np.concatenate((poly[:i+1], loop, poly[i:i+1], poly[i+1:]))
    if len(poly) > 900:
        raise ValueError("mask too complex for this initial single-polygon export")
    score = iou(m, raster(poly, m.shape))
    height, width = m.shape
    normalized = poly / np.array([width, height], dtype=np.float32)
    serialized = np.array([float(f"{v:.8f}") for v in normalized.ravel()], np.float32).reshape(-1, 2)
    loaded = resample(serialized) * np.array([width, height], dtype=np.float32)
    training_score = iou(m, raster(loaded, m.shape))
    if score < .97 or training_score < .90:
        raise ValueError(f"mask polygon fidelity too low: raw={score:.3f}, resampled={training_score:.3f}")
    return normalized, score, training_score


def convert_frame(raw, meta_path, out):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    split, stem = meta["split"], meta_path.stem
    if split not in SPLITS or meta_path.parent.name != split:
        raise ValueError("split mismatch")
    rgb = raw / "images" / split / f"{stem}.png"
    mask_path = raw / "masks" / split / f"{stem}.png"
    if sha256(rgb) != meta["rgb_sha256"] or sha256(mask_path) != meta["mask_sha256"]:
        raise ValueError("capture checksum mismatch")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    image = cv2.imread(str(rgb), cv2.IMREAD_COLOR)
    if mask is None or image is None or mask.ndim != 2 or mask.dtype != np.uint16:
        raise ValueError("invalid RGB or 16-bit instance mask")
    if image.shape[:2] != mask.shape or list(reversed(mask.shape)) != meta["resolution"]:
        raise ValueError("resolution mismatch")
    expected_ids = {int(obj["instance_id"]) for obj in meta["objects"]}
    if len(expected_ids) != len(meta["objects"]) or not set(np.unique(mask)).issubset(expected_ids | {0}):
        raise ValueError("unmapped or duplicate instance IDs")
    lines, classes, scores, display_polys = [], [], [], []
    for obj in meta["objects"]:
        cid = int(obj["class_id"])
        if not 0 <= cid < len(CLASSES) or CLASSES[cid] != obj["shape"]:
            raise ValueError("invalid class mapping")
        binary = mask == obj["instance_id"]
        pixels = int(binary.sum())
        if pixels != obj["visible_pixels"]:
            raise ValueError("instance pixel count mismatch")
        if not pixels:
            continue
        poly, score, training_score = mask_polygon(binary)
        coords = " ".join(f"{v:.8f}" for v in poly.ravel())
        lines.append(f"{cid} {coords}")
        classes.append(cid)
        scores.append([score, training_score])
        display_polys.append((cid, (poly * np.array([mask.shape[1], mask.shape[0]])).round().astype(np.int32)))
    if meta["negative"] != (not meta["objects"]):
        raise ValueError("inconsistent intended-negative metadata")
    if not lines and not meta["negative"]:
        raise ValueError("unintended empty positive scene")
    (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    shutil.copy2(rgb, out / "images" / split / rgb.name)
    return split, classes, scores, image, display_polys


def contact_sheet(tiles, path):
    thumbs = []
    colors = [(65, 85, 255), (255, 170, 50), (110, 230, 60)]  # BGR for OpenCV
    for name, image, polygons in tiles:
        canvas = image.copy()
        for cid, poly in polygons:
            cv2.polylines(canvas, [poly], True, colors[cid], 2)
            xy = poly.min(axis=0)
            cv2.putText(canvas, CLASSES[cid], (int(xy[0]), max(14, int(xy[1])-3)), cv2.FONT_HERSHEY_SIMPLEX, .4, colors[cid], 1, cv2.LINE_AA)
        canvas = cv2.resize(canvas, (320, 240))
        cv2.putText(canvas, name, (5, 232), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(canvas)
    if not thumbs:
        return
    while len(thumbs) % 4:
        thumbs.append(np.zeros_like(thumbs[0]))
    grid = np.vstack([np.hstack(thumbs[i:i+4]) for i in range(0, len(thumbs), 4)])
    if not cv2.imwrite(str(path), grid):
        raise OSError(f"Could not write {path}")


def prepare(raw, out):
    raw, out = Path(raw).resolve(), Path(out).resolve()
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("classes") != list(CLASSES):
        raise ValueError("Not a compatible raw primitive dataset")
    progress = json.loads((raw / "progress.json").read_text(encoding="utf-8"))
    if not progress.get("complete"):
        raise ValueError("Capture is incomplete. Resume capture or use a separately completed smaller run.")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output is not empty. Choose a new output directory; existing data will not be deleted.")
    for split in SPLITS:
        for kind in ("images", "labels"):
            (out / kind / split).mkdir(parents=True, exist_ok=True)
    stats = {s: {"images": 0, "negative_images": 0, "instances": [0, 0, 0]} for s in SPLITS}
    rejected, tiles, scores, hashes = [], [], [], {}
    metas = sorted((raw / "meta").glob("*/*.json"))
    if len(metas) != manifest["settings"]["frames"]:
        raise ValueError("Missing or extra committed frame metadata")
    for path in metas:
        try:
            split, classes, fidelity, image, polys = convert_frame(raw, path, out)
            digest = sha256(out / "images" / split / f"{path.stem}.png")
            if classes and digest in hashes and hashes[digest] != split:
                raise ValueError("identical positive RGB image occurs in different splits")
            if classes:
                hashes[digest] = split
            stats[split]["images"] += 1
            stats[split]["negative_images"] += not classes
            for cid, n in Counter(classes).items():
                stats[split]["instances"][cid] += n
            scores.extend(fidelity)
            if len(tiles) < 24:
                tiles.append((f"{split}/{path.stem}", image, polys))
        except (ValueError, OSError, KeyError) as exc:
            rejected.append({"frame": str(path.relative_to(raw)), "reason": str(exc)})
            for kind, extension in (("images", "png"), ("labels", "txt")):
                (out / kind / path.parent.name / f"{path.stem}.{extension}").unlink(missing_ok=True)
    dump_json(out / "conversion_report.json", {"splits": stats, "rejected": rejected,
              "polygon_iou_min": min((x[0] for x in scores), default=None),
              "resampled_iou_min": min((x[1] for x in scores), default=None), "raw_manifest_sha256": sha256(raw / "manifest.json")})
    contact_sheet(tiles, out / "review.jpg")
    if len(rejected) > max(2, .15 * len(metas)) or any(not all(stats[s]["instances"]) for s in SPLITS):
        raise ValueError("Dataset quality gate failed: excessive rejected frames or a missing class in a split. Inspect conversion_report.json; no training YAML was created.")
    data = {"path": out.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test", "names": dict(enumerate(CLASSES))}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"READY: {out / 'data.yaml'}\nInspect {out / 'review.jpg'} BEFORE training. Rejected frames: {len(rejected)}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.raw, args.out)
