"""Offline SDG appearance policy and inspection images; NumPy + stdlib only."""
from __future__ import annotations

import colorsys
import numpy as np

REVISION = "contrast-v2.1"
# Display-space colors. USD material inputs are converted to scene-linear RGB.
GROUNDS = np.array([[.43, .47, .51], [.53, .51, .47], [.35, .39, .38],
                    [.62, .64, .65], [.28, .31, .35], [.55, .57, .52]])


def srgb_to_linear(rgb):
    a = np.clip(np.asarray(rgb, dtype=float), 0., 1.)
    return np.where(a <= .04045, a / 12.92, ((a + .055) / 1.055) ** 2.4)


def luminance(rgb):
    return float(np.asarray(rgb) @ [.2126, .7152, .0722])


def color_gap(a, b):
    """Display-space sampling heuristic, not a guarantee of rendered contrast."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def sample_look(rng, light_scale=1.):
    if not np.isfinite(light_scale) or light_scale <= 0:
        raise ValueError("light_scale must be positive and finite")
    mode = str(rng.choice(["clear", "varied", "hard"], p=[.7, .2, .1]))
    if mode == "hard":
        floor = np.full(3, float(rng.uniform(.84, .94)))
    else:
        floor = np.clip(GROUNDS[int(rng.integers(len(GROUNDS)))] + rng.uniform(-.025, .025, 3), .15, .80)
        if mode == "varied" and rng.random() < .5:
            floor = np.full(3, float(rng.uniform(.70, .88)))
    return {"mode": mode, "floor_srgb": floor.tolist(),
            "floor_linear": srgb_to_linear(floor).tolist(),
            "fill_intensity": float(rng.uniform(120., 230.) * light_scale),
            "key_intensity": float(rng.uniform(700., 1250.) * light_scale),
            "key_rotation_deg": [float(rng.uniform(-45., -25.)), float(rng.uniform(-25., 25.)),
                                 float(rng.uniform(-180., 180.))],
            "key_angle_deg": float(rng.uniform(4., 10.)),
            "light_color": [1., float(rng.uniform(.94, 1.)), float(rng.uniform(.90, 1.))]}


def sample_surface(rng, look):
    """Same distribution for every target AND cone distractor; no class input."""
    floor = np.asarray(look["floor_srgb"])
    mode = look["mode"]
    for _ in range(256):
        hue = float(rng.random())
        if mode == "hard":
            color = np.asarray(colorsys.hsv_to_rgb(hue, float(rng.uniform(0., .25)), float(rng.uniform(.76, .96))))
        else:
            saturation = float(rng.uniform(.72, .98) if mode == "clear" else rng.uniform(.3, .92))
            color = np.asarray(colorsys.hsv_to_rgb(hue, saturation, float(rng.uniform(.35, .86))))
            if rng.random() < .15:
                color[:] = float(rng.uniform(.08, .24))
        gap = color_gap(color, floor)
        if mode != "clear" or (gap >= .24 and (abs(luminance(color)-luminance(floor)) >= .15 or gap >= .34)):
            break
    else:
        color = np.array([.10, .16, .26]) if luminance(floor) > .4 else np.array([.90, .65, .10])
        gap = color_gap(color, floor)
    return {"color_srgb": color.tolist(), "color": srgb_to_linear(color).tolist(),
            "roughness": float(rng.uniform(.75, .95)), "metallic": 0.,
            "sampled_color_gap": gap}


def dilate(mask, radius):
    """Square dilation used only for preview boundaries and local diagnostics."""
    m = np.asarray(mask, bool)
    if m.ndim != 2 or radius < 0:
        raise ValueError("Expected a 2D mask and nonnegative radius")
    h, w = m.shape
    p = np.pad(m, radius)
    result = np.zeros_like(m)
    for y in range(2*radius + 1):
        for x in range(2*radius + 1):
            result |= p[y:y+h, x:x+w]
    return result


def colorize_instances(mask):
    """VIEW ONLY. Never replace the uint16 training label map with these colors."""
    m = np.asarray(mask)
    if m.ndim != 2 or not np.issubdtype(m.dtype, np.integer) or np.any(m < 0):
        raise ValueError("Expected a nonnegative 2D integer instance map")
    image = np.zeros((*m.shape, 3), np.uint8)
    for identity in np.unique(m):
        if identity:
            rgb = colorsys.hsv_to_rgb((int(identity) * .61803398875) % 1., .80, 1.)
            image[m == identity] = np.rint(np.asarray(rgb)*255).astype(np.uint8)
    return image


def overlay(rgb, mask, entries):
    image = np.asarray(rgb).copy()
    colors = np.array([[255, 85, 65], [50, 170, 255], [60, 230, 110]], np.uint8)
    for obj in entries:
        hit = mask == obj["instance_id"]
        if not hit.any():
            continue
        color = colors[obj["class_id"]]
        image[hit] = (.7 * image[hit] + .3 * color).astype(np.uint8)
        image[dilate(hit, 1) & ~hit] = color
    return image


def frame_quality(rgb, mask, entries):
    """Diagnostics only: hard examples are not silently discarded or relabeled."""
    a = np.asarray(rgb, float) / 255.
    m = np.asarray(mask)
    if a.shape != (*m.shape, 3):
        raise ValueError("RGB/mask shape mismatch")
    result = []
    for obj in entries:
        hit = m == obj["instance_id"]
        pixels = int(hit.sum())
        item = {"instance_id": obj["instance_id"], "shape": obj["shape"], "visible_pixels": pixels, "flags": []}
        if not pixels:
            item["flags"].append("not_visible")
            result.append(item)
            continue
        yy, xx = np.nonzero(hit)
        x0, x1, y0, y1 = max(0, xx.min()-6), min(m.shape[1], xx.max()+7), max(0, yy.min()-6), min(m.shape[0], yy.max()+7)
        local_mask, local_rgb = m[y0:y1, x0:x1], a[y0:y1, x0:x1]
        local_hit = local_mask == obj["instance_id"]
        ring = dilate(local_hit, 5) & (local_mask == 0)
        inside = np.median(local_rgb[local_hit], axis=0)
        item["bbox_xyxy"] = [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1]
        item["median_rgb"] = np.round(inside*255, 1).tolist()
        item["white_fraction"] = float(np.mean(np.all(local_rgb[local_hit] >= 245/255., axis=1)))
        item["black_fraction"] = float(np.mean(np.all(local_rgb[local_hit] <= 5/255., axis=1)))
        if pixels < 100 or min(xx.max()-xx.min()+1, yy.max()-yy.min()+1) < 8:
            item["flags"].append("small_instance")
        if item["white_fraction"] > .35:
            item["flags"].append("near_white")
        if item["black_fraction"] > .35:
            item["flags"].append("near_black")
        if ring.any():
            outside = np.median(local_rgb[ring], axis=0)
            item["local_rgb_gap"] = color_gap(inside, outside)
            item["local_luma_gap"] = abs(luminance(inside)-luminance(outside))
            if item["local_rgb_gap"] < .10:
                item["flags"].append("low_local_contrast")
        else:
            item["flags"].append("no_background_ring")
        result.append(item)
    return {"visible_instances": sum(i["visible_pixels"] > 0 for i in result),
            "objects": result, "flagged_instances": sum(bool(i["flags"]) for i in result)}
