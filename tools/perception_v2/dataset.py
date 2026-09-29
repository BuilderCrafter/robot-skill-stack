"""Read the raw SDG dataset without OpenCV/Pillow or training-label conversion."""
from pathlib import Path
import hashlib
import json
import struct
import zlib
import numpy as np
from robot_skill_stack.world.perception.frame import PerceptionFrame
from robot_skill_stack.world.perception.v2.types import CLASSES, validate_frame


def read_png(path):
    data = Path(path).read_bytes()
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'Not a PNG: {path}')
    offset, chunks, header = 8, [], None
    while offset < len(data):
        if offset+12 > len(data):
            raise ValueError('Truncated PNG chunk')
        length = struct.unpack('>I', data[offset:offset+4])[0]
        if offset+12+length > len(data):
            raise ValueError('Truncated PNG chunk payload')
        kind, body = data[offset+4:offset+8], data[offset+8:offset+8+length]
        crc = struct.unpack('>I', data[offset+8+length:offset+12+length])[0]
        if zlib.crc32(kind+body) & 0xffffffff != crc:
            raise ValueError(f'PNG CRC mismatch: {path}')
        if kind == b'IHDR':
            header = struct.unpack('>IIBBBBB', body)
        elif kind == b'IDAT':
            chunks.append(body)
        offset += 12+length
        if kind == b'IEND':
            break
    if header is None:
        raise ValueError('Missing PNG header')
    w, h, bits, kind, compression, filtering, interlace = header
    if (bits, kind) not in ((8, 2), (16, 0)) or compression or filtering or interlace or not 0 < w*h <= 1920*1080:
        raise ValueError('Expected generator RGB8 / instance-gray16 noninterlaced PNG')
    bpp = 3 if kind == 2 else 2
    raw = zlib.decompress(b''.join(chunks))
    if len(raw) != h*(w*bpp+1):
        raise ValueError('PNG decompressed size mismatch')
    rows = np.frombuffer(raw, np.uint8).reshape(h, w*bpp+1)
    out = np.zeros((h, w*bpp), np.uint8)
    for y, row in enumerate(rows):
        method, values = int(row[0]), row[1:]
        if method == 0:
            out[y] = values
            continue
        if method not in (1, 2, 3, 4):
            raise ValueError('Unsupported PNG row filter')
        for x, value in enumerate(values):
            a, b, c = int(out[y, x-bpp]) if x >= bpp else 0, int(out[y-1, x]) if y else 0, int(out[y-1, x-bpp]) if y and x >= bpp else 0
            p = a+b-c
            nearest = min(((abs(p-a), a), (abs(p-b), b), (abs(p-c), c)), key=lambda pair: pair[0])[1]
            pred = {1: a, 2: b, 3: (a+b)//2, 4: nearest}[method]
            out[y, x] = (int(value)+pred) & 255
    return out.reshape(h, w, 3) if kind == 2 else out.copy().view('>u2').reshape(h, w).astype(np.uint16)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def list_samples(root, split):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    progress = json.loads((root/'progress.json').read_text())
    if manifest.get('format') != 'primitive-sdg-v1' or manifest.get('classes') != list(CLASSES):
        raise ValueError('Expected RAW primitive-sdg-v1 dataset, not the *_yolo directory')
    if not progress.get('complete'):
        raise ValueError('Capture is not complete')
    samples = sorted((root/'meta'/split).glob('*.json'))
    if not samples or len(samples) != progress.get('splits', {}).get(split):
        raise ValueError('Missing or incomplete split')
    return samples, manifest


def load_sample(root, meta_path):
    root, meta_path = Path(root), Path(meta_path)
    meta = json.loads(meta_path.read_text())
    split, name = meta_path.parent.name, meta_path.stem
    rgb_path, mask_path = root/'images'/split/f'{name}.png', root/'masks'/split/f'{name}.png'
    for path, key in ((rgb_path, 'rgb_sha256'), (mask_path, 'mask_sha256')):
        if sha256(path) != meta.get(key):
            raise ValueError(f'Capture integrity check failed: {path}')
    depth_path = root/'depth'/split/f'{name}.npz'
    if not depth_path.is_file():
        raise ValueError(f'Missing {depth_path}; the V1 comparison requires capture with --save-depth')
    with np.load(depth_path, allow_pickle=False) as a:
        depth = a['depth']
    rgb, masks = read_png(rgb_path), read_png(mask_path)
    if masks.shape != depth.shape or list(meta['resolution']) != [rgb.shape[1], rgb.shape[0]]:
        raise ValueError('RGB/mask/depth sizes differ')
    if meta.get('depth_type') != 'distance_to_image_plane_m':
        raise ValueError('Depth must be distance to image plane in meters')
    frame = PerceptionFrame(meta['frame_id'], rgb, depth, np.array(meta['K']),
                            np.array(meta['T_world_camera_opencv']), float(meta['frame_id']))
    validate_frame(frame)
    truth, used = [], set()
    for obj in meta['objects']:
        identity, label = obj['instance_id'], obj['shape']
        if not isinstance(identity, int) or identity <= 0 or identity in used or label not in CLASSES:
            raise ValueError('Invalid/duplicate ground-truth instance ID or class')
        used.add(identity)
        if CLASSES[obj['class_id']] != label:
            raise ValueError('Inconsistent ground-truth class index')
        mask = masks == identity
        if int(mask.sum()) != obj['visible_pixels']:
            raise ValueError('Metadata visible-pixel count differs from mask')
        if mask.any():
            truth.append(dict(mask=mask, label=label, position=np.array(obj['position']),
                              size=np.array(obj['size_world_aabb']), elevated=bool(obj['elevated'])))
    if set(np.unique(masks)) - {0, *used}:
        raise ValueError('Unmapped ground-truth pixels')
    # Only frame (RGB, depth, calibration) goes into either perception pipeline.
    return frame, truth, meta
