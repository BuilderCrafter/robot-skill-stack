"""Simulator-independent data format, sampling, and lossless PNG I/O."""
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np

CLASSES = ("cube", "sphere", "cylinder")
SPLITS = ("train", "val", "test")
FORMAT = "primitive-sdg-v1"


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_png(path, array):
    """RGB uint8 or instance-ID uint16 PNG. No Pillow/OpenCV needed in Isaac."""
    a = np.asarray(array)
    if a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3:
        bit_depth, color_type, raw = 8, 2, np.ascontiguousarray(a)
    elif a.dtype == np.uint16 and a.ndim == 2:
        bit_depth, color_type, raw = 16, 0, np.ascontiguousarray(a.astype(">u2"))
    else:
        raise ValueError(f"Expected HxWx3 uint8 or HxW uint16, got {a.shape} {a.dtype}")
    height, width = a.shape[:2]
    if min(height, width) < 1:
        raise ValueError("Empty PNG")
    rows = raw.view(np.uint8).reshape(height, -1)
    payload = b"".join(b"\0" + row.tobytes() for row in rows)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    blob = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0))
    blob += chunk(b"IDAT", zlib.compress(payload, 3)) + chunk(b"IEND", b"")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(path)


def split_schedule(frames, seed):
    if frames < 12:
        raise ValueError("Use at least 12 frames so all three splits have examples")
    nval = ntest = max(3, frames // 10)
    splits = ["train"] * (frames - nval - ntest) + ["val"] * nval + ["test"] * ntest
    np.random.default_rng(seed).shuffle(splits)
    return splits


def camera_matrix(eye, target):
    """Column-vector world_from_camera, USD camera: X right, Y up, -Z forward."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0., 1., 0.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.column_stack((x, y, z)), eye
    return T


def project(point, T_world_usd_camera, K):
    p = np.linalg.inv(T_world_usd_camera) @ np.r_[point, 1.]
    p = p[:3] * [1., -1., -1.]
    if p[2] <= 0:
        return None
    uv = K @ p
    return uv[:2] / uv[2]


def rotation(yaw, lying=False):
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    Ry = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]) if lying else np.eye(3)
    return Rz @ Ry


def world_extents(shape, dimensions, R):
    d, R = np.asarray(dimensions, float), np.asarray(R, float)
    if shape == "sphere":
        return d.copy()
    if shape == "cylinder":
        axis = np.abs(R[:, 2])
        return d[2] * axis + d[0] * np.sqrt(np.maximum(0., 1.-axis**2))
    return np.abs(R) @ d


def random_color(rng):
    if rng.random() < .20:
        return [float(rng.uniform(.18, .85))] * 3
    return rng.uniform(.10, .90, 3).tolist()


def sample_objects(rng, count, required_class, bounds, support, camera, K, resolution):
    """Analytically supported poses; no expensive dynamics or overlapping solids."""
    objects = []
    width, height = resolution
    for slot in range(count):
        shape = required_class if slot == 0 else CLASSES[int(rng.integers(3))]
        diameter = float(rng.uniform(.028, .066))
        length = float(rng.uniform(.04, .12)) if shape == "cylinder" else diameter
        dims = np.array([diameter, diameter, length])
        lying = shape == "cylinder" and rng.random() < .5
        yaw = float(rng.uniform(-np.pi, np.pi))
        R = rotation(yaw, lying)
        world_size = world_extents(shape, dims, R)
        # A conservative footprint radius prevents interpenetration, including long lying cylinders.
        footprint = float(np.linalg.norm(world_size[:2]) / 2)
        elevated = rng.random() < .20
        z = support + world_size[2] / 2 + (float(rng.uniform(.04, .13)) if elevated else 0.)
        for _ in range(300):
            xy = rng.uniform(bounds[:2], bounds[2:])
            if any(np.linalg.norm(xy - np.asarray(o["position"][:2])) < footprint + o["footprint"] + .004 for o in objects):
                continue
            p = np.r_[xy, z]
            uv = project(p, camera, K)
            if uv is None or not (12 <= uv[0] < width - 12 and 12 <= uv[1] < height - 12):
                continue
            objects.append({"slot": slot, "shape": shape, "class_id": CLASSES.index(shape),
                            "position": p.tolist(), "dimensions_local": dims.tolist(),
                            "rotation_world": R.tolist(), "size_world_aabb": world_size.tolist(),
                            "axis_world": R[:, 2].tolist() if shape == "cylinder" else None,
                            "radius": diameter / 2 if shape != "cube" else None,
                            "length": length if shape == "cylinder" else None,
                            "yaw": yaw if shape == "cube" else None, "elevated": bool(elevated),
                            "footprint": footprint, "color": random_color(rng),
                            "roughness": float(rng.uniform(.35, .95)), "metallic": float(rng.uniform(0., .15))})
            break
        else:
            if slot == 0:
                raise RuntimeError("No target fits the camera view. Check --bounds, --support-z, and the scene camera.")
            break
    return objects


def decode_instances(data, objects):
    """Resolve renderer IDs from THIS frame to unique object paths, never class IDs."""
    if not isinstance(data, dict) or "data" not in data:
        raise RuntimeError("Instance annotator did not return data + info")
    raw = np.squeeze(np.asarray(data["data"]))
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
        raise RuntimeError(f"Expected a non-colorized instance ID image, got {raw.shape} {raw.dtype}")
    labels = data.get("info", {}).get("idToLabels", {})
    if not labels:
        raise RuntimeError("Instance annotator has no idToLabels map; refuse to create empty labels")
    mask = np.zeros(raw.shape, dtype=np.uint16)
    entries = []
    for obj in objects:
        ids = []
        for key, value in labels.items():
            if isinstance(value, dict):
                value = value.get("primPath", value.get("prim_path", value.get("path", "")))
            if isinstance(value, str) and (value == obj["prim_path"] or value.startswith(obj["prim_path"] + "/")):
                ids.append(int(key))
        instance_id = int(obj["slot"]) + 1
        hit = np.isin(raw, ids) if ids else np.zeros(raw.shape, bool)
        mask[hit] = instance_id
        entries.append({**obj, "instance_id": instance_id, "renderer_ids": ids, "visible_pixels": int(hit.sum())})
    return mask, entries


def preview(rgb, mask, entries):
    image = np.asarray(rgb).copy()
    colors = np.array([[255, 85, 65], [50, 170, 255], [60, 230, 110]], dtype=np.uint8)
    for obj in entries:
        m = mask == obj["instance_id"]
        if not m.any():
            continue
        color = colors[obj["class_id"]]
        image[m] = (.65 * image[m] + .35 * color).astype(np.uint8)
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        image[y0:y0+2, x0:x1+1] = color
        image[max(y0, y1-1):y1+1, x0:x1+1] = color
        image[y0:y1+1, x0:x0+2] = color
        image[y0:y1+1, max(x0, x1-1):x1+1] = color
    return image
