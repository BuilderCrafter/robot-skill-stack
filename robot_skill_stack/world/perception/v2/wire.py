"""Bounded NumPy wire format; RGB only goes to the model. No pickle or filesystem RPC."""
import io
import json
import zipfile
import numpy as np
from robot_skill_stack.world.perception.v2.types import CLASSES, Instance, Segmentation

MAX_BYTES = 48 * 1024 * 1024
MAX_PIXELS = 1920 * 1080
VERSION = 1


def pack(**arrays):
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    data = out.getvalue()
    if len(data) > MAX_BYTES:
        raise ValueError('Payload too large')
    return data


def unpack(data):
    if len(data) > MAX_BYTES:
        raise ValueError('Payload too large')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        members = z.infolist()
        if len(members) > 8 or sum(m.file_size for m in members) > MAX_BYTES:
            raise ValueError('Unpacked payload too large')
        if len({m.filename for m in members}) != len(members):
            raise ValueError('Duplicate fields')
    with np.load(io.BytesIO(data), allow_pickle=False) as a:
        return {key: a[key] for key in a.files}


def encode_request(frame_id, rgb, options):
    return pack(rgb=rgb, metadata=json.dumps(dict(version=VERSION, frame_id=int(frame_id), **options)))


def decode_request(data):
    a = unpack(data)
    if set(a) != {'rgb', 'metadata'}:
        raise ValueError('Invalid request fields')
    meta = json.loads(str(a['metadata']))
    rgb = a['rgb']
    if (meta.get('version') != VERSION or not isinstance(meta.get('frame_id'), int)
            or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 1 or np.prod(rgb.shape[:2]) > MAX_PIXELS):
        raise ValueError('Invalid request version, frame ID, or RGB image')
    allowed = {'version', 'frame_id', 'confidence', 'nms_iou', 'max_detections'}
    if set(meta) != allowed:
        raise ValueError('Unexpected request metadata')
    from robot_skill_stack.world.perception.v2.settings import V2Config
    options = {k: meta[k] for k in ('confidence', 'nms_iou', 'max_detections')}
    V2Config(**options)
    return meta['frame_id'], rgb, options


def encode_response(result, shape):
    masks = np.stack([i.mask for i in result.instances]) if result.instances else np.empty((0, *shape), bool)
    meta = dict(version=VERSION, frame_id=result.frame_id, model=result.model, inference_s=result.inference_s,
                labels=[i.label for i in result.instances], scores=[i.confidence for i in result.instances])
    return pack(masks=masks, metadata=json.dumps(meta, allow_nan=False))


def decode_response(data, frame_id, shape):
    a = unpack(data)
    if set(a) != {'masks', 'metadata'}:
        raise ValueError('Invalid response fields')
    meta, masks = json.loads(str(a['metadata'])), a['masks']
    if meta.get('version') != VERSION or meta.get('frame_id') != frame_id:
        raise ValueError('Response is for a different frame / protocol version')
    if masks.dtype != np.bool_ or masks.ndim != 3 or masks.shape[1:] != tuple(shape) or len(masks) > 64:
        raise ValueError('Masks do not match original RGB dimensions')
    labels, scores = meta.get('labels', []), meta.get('scores', [])
    if len(labels) != len(masks) or len(scores) != len(masks):
        raise ValueError('Mask/class count mismatch')
    model = meta.get('model', {})
    if model.get('classes') != list(CLASSES) or not isinstance(model.get('sha256'), str):
        raise ValueError('Worker is not a cube/sphere/cylinder segmentation model')
    elapsed = float(meta.get('inference_s', 0.))
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid inference duration')
    return Segmentation(frame_id, [Instance(m, c, float(s)) for m, c, s in zip(masks, labels, scores)], model, elapsed)
