from __future__ import annotations

import ast
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appearance import REVISION, color_gap, colorize_instances, dilate, frame_quality, overlay, sample_look, sample_surface, srgb_to_linear
from common import CLASSES, camera_matrix, sha256, write_png
import generate_dataset


def read_own_png(path):
    data = Path(path).read_bytes()
    width, height, bits, kind = struct.unpack('>IIBB', data[16:26])
    offset, compressed = 8, b''
    while offset < len(data):
        length = struct.unpack('>I', data[offset:offset+4])[0]
        if data[offset+4:offset+8] == b'IDAT':
            compressed += data[offset+8:offset+8+length]
        offset += 12 + length
    channels = 3 if kind == 2 else 1
    rows = np.frombuffer(zlib.decompress(compressed), np.uint8).reshape(height, width*channels*bits//8+1)
    assert np.all(rows[:, 0] == 0)
    pixels = rows[:, 1:].copy()
    if bits == 16:
        return pixels.view('>u2').reshape(height, width).astype(np.uint16)
    return pixels.reshape(height, width, 3)


class AppearanceTests(unittest.TestCase):
    def test_srgb_reference_values(self):
        np.testing.assert_allclose(srgb_to_linear([0., .5, 1.]), [0., .214041140482, 1.], atol=1e-10)

    def test_srgb_clamps(self):
        np.testing.assert_equal(srgb_to_linear([-1., 2.]), [0., 1.])

    def test_reproducible_independent_sampling(self):
        a, b = np.random.default_rng(73), np.random.default_rng(73)
        for _ in range(30):
            la, lb = sample_look(a), sample_look(b)
            self.assertEqual(la, lb)
            self.assertEqual(sample_surface(a, la), sample_surface(b, lb))

    def test_modes_distribution(self):
        rng = np.random.default_rng(81)
        counts = Counter(sample_look(rng)['mode'] for _ in range(2000))
        for mode, target in [('clear', .7), ('varied', .2), ('hard', .1)]:
            self.assertLess(abs(counts[mode]/2000-target), .035)

    def test_clear_object_floor_color_separation_and_matte(self):
        rng = np.random.default_rng(94)
        for _ in range(500):
            look = sample_look(rng); look['mode'] = 'clear'
            surface = sample_surface(rng, look)
            self.assertGreaterEqual(color_gap(surface['color_srgb'], look['floor_srgb']), .24)
            self.assertEqual(surface['metallic'], 0.)
            self.assertGreaterEqual(surface['roughness'], .75)
            np.testing.assert_allclose(surface['color'], srgb_to_linear(surface['color_srgb']))

    def test_hard_examples_include_pale_objects(self):
        rng = np.random.default_rng(28)
        look = {'mode': 'hard', 'floor_srgb': [.9, .9, .9]}
        for _ in range(20):
            surface = sample_surface(rng, look)
            self.assertGreater(max(surface['color_srgb']), .75)

    def test_palette_is_not_class_conditioned(self):
        import inspect
        self.assertEqual(list(inspect.signature(sample_surface).parameters), ['rng', 'look'])
        text = inspect.getsource(sample_surface)
        self.assertNotIn('class_id', text)
        self.assertNotIn('shape', text)

    def test_light_scale_only_changes_intensities(self):
        a, b = sample_look(np.random.default_rng(9)), sample_look(np.random.default_rng(9), .5)
        for k in a:
            self.assertEqual(b[k], a[k]*.5 if k.endswith('_intensity') else a[k])

    def test_invalid_light_scale(self):
        for scale in [-1., 0., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                sample_look(np.random.default_rng(1), scale)

    def test_colorization_does_not_change_label_ids(self):
        mask = np.array([[0, 1, 2], [3, 4, 65535]], np.uint16); saved = mask.copy()
        view = colorize_instances(mask)
        np.testing.assert_array_equal(mask, saved)
        self.assertEqual(view.dtype, np.uint8)
        self.assertTrue((view[0, 0] == 0).all())
        self.assertTrue((view[mask > 0].max(axis=1) == 255).all())
        self.assertEqual(len(np.unique(view.reshape(-1, 3), axis=0)), 6)

    def test_negative_mask_preview_is_black(self):
        self.assertFalse(colorize_instances(np.zeros((8, 8), np.uint16)).any())

    def test_invalid_colorization_input(self):
        for invalid in [np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), float), np.full((2, 2), -1)]:
            with self.assertRaises(ValueError):
                colorize_instances(invalid)

    def test_png_preview_and_raw_roundtrip(self):
        mask = np.array([[0, 1], [400, 65535]], np.uint16)
        with tempfile.TemporaryDirectory() as d:
            raw, view = Path(d)/'raw.png', Path(d)/'view.png'
            write_png(raw, mask); write_png(view, colorize_instances(mask))
            np.testing.assert_equal(read_own_png(raw), mask)
            np.testing.assert_equal(read_own_png(view), colorize_instances(mask))

    def test_overlay_preserves_inputs_and_outline(self):
        rgb = np.full((24, 24, 3), 150, np.uint8); mask = np.zeros((24, 24), np.uint16)
        mask[8:16, 8:16] = 1; before = mask.copy()
        out = overlay(rgb, mask, [{'instance_id': 1, 'class_id': 0}])
        np.testing.assert_equal(mask, before)
        self.assertTrue((rgb == 150).all())
        np.testing.assert_equal(out[7, 8], [255, 85, 65])

    def test_dilation_does_not_wrap_edges(self):
        m = np.zeros((8, 9), bool); m[0, 0] = True
        result = dilate(m, 1)
        self.assertEqual(result.sum(), 4)
        self.assertFalse(result[-1].any())
        np.testing.assert_equal(dilate(m, 0), m)

    def test_low_contrast_diagnostic(self):
        rgb = np.full((40, 40, 3), 225, np.uint8); mask = np.zeros((40, 40), np.uint16)
        mask[10:30, 10:30] = 1; rgb[mask > 0] = 221
        q = frame_quality(rgb, mask, [{'instance_id': 1, 'shape': 'cube'}])
        self.assertIn('low_local_contrast', q['objects'][0]['flags'])
        self.assertEqual(q['visible_instances'], 1)

    def test_high_contrast_diagnostic(self):
        rgb = np.full((40, 40, 3), 140, np.uint8); mask = np.zeros((40, 40), np.uint16)
        mask[10:30, 10:30] = 1; rgb[mask > 0] = [30, 55, 120]
        q = frame_quality(rgb, mask, [{'instance_id': 1, 'shape': 'cube'}])
        self.assertEqual(q['objects'][0]['flags'], [])

    def test_small_and_invisible_diagnostics(self):
        rgb = np.full((20, 20, 3), 100, np.uint8); mask = np.zeros((20, 20), np.uint16); mask[4:7, 4:7] = 1
        q = frame_quality(rgb, mask, [{'instance_id': i, 'shape': 'sphere'} for i in [1, 2]])
        self.assertIn('small_instance', q['objects'][0]['flags'])
        self.assertEqual(q['objects'][1]['flags'], ['not_visible'])

    def test_negative_diagnostic_not_failure(self):
        q = frame_quality(np.full((20, 20, 3), 220, np.uint8), np.zeros((20, 20), np.uint16), [])
        self.assertEqual(q['flagged_instances'], 0)

    def test_json_contains_no_nan(self):
        q = frame_quality(np.zeros((10, 10, 3), np.uint8), np.ones((10, 10), np.uint16), [{'instance_id': 1, 'shape': 'cube'}])
        json.dumps(q, allow_nan=False)
        self.assertIn('no_background_ring', q['objects'][0]['flags'])


class FakeApp:
    def __init__(self, *args, **kwargs): pass
    def is_running(self): return True
    def close(self): pass


class FakeScene:
    calls = 0
    def __init__(self, app, args):
        self.args = args
        self.K = np.array([[100., 0., 64.], [0., 100., 64.], [0., 0., 1.]])
        self.appearance = types.SimpleNamespace(current={'mode': 'clear'}, setup_info=lambda: {'ground_prims': ['/Ground']})
    def choose_camera(self, rng): return camera_matrix([.9, -.7, .8], [.45, 0., .04])
    def apply(self, objects, rng, T):
        self.objects = objects
        for obj in objects: obj['prim_path'] = f"/Targets/{obj['slot']}"
    def capture(self):
        type(self).calls += 1
        mask = np.zeros((128, 128), np.uint32); rgb = np.full((128, 128, 4), 150, np.uint8)
        labels = {'0': 'BACKGROUND'}
        for i, obj in enumerate(self.objects):
            mask[25:50, 4+i*24:20+i*24] = i+1
            labels[str(i+1)] = obj['prim_path']
        return {'rgb': rgb, 'instances': {'data': mask, 'info': {'idToLabels': labels}}, 'depth': np.ones((128, 128), np.float32)}
    def close(self): pass


class PipelineTests(unittest.TestCase):
    def run_capture(self, out, extra=()):
        argv = ['generate_dataset.py', '--blank-scene', '--frames', '24', '--width', '128', '--height', '128', '--out', str(out), '--save-depth', *extra]
        fake = {'isaacsim': types.SimpleNamespace(SimulationApp=FakeApp), 'isaac_scene': types.SimpleNamespace(CaptureScene=FakeScene)}
        with patch.dict(sys.modules, fake), patch.object(sys, 'argv', argv), patch.object(sys, 'version_info', (3, 11)), patch.object(np, '__version__', '1.26.4'), contextlib.redirect_stdout(io.StringIO()):
            generate_dataset.main()

    def test_capture_outputs_raw_masks_reviews_metadata_and_resume(self):
        FakeScene.calls = 0
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'pilot'; self.run_capture(out)
            self.assertEqual(len(list(out.glob('mask_preview/*.png'))), 24)
            self.assertEqual(len(list(out.glob('review/*.png'))), 24)
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['generator_revision'], REVISION)
            self.assertEqual(len(json.loads((out/'quality_report.json').read_text())['frames']), 24)
            for path in out.glob('meta/*/*.json'):
                meta = json.loads(path.read_text())
                raw = out/'masks'/path.parent.name/f'{path.stem}.png'
                self.assertEqual(sha256(raw), meta['mask_sha256'])
                mask = read_own_png(raw)
                view = read_own_png(out/'mask_preview'/f'{path.stem}.png')
                np.testing.assert_equal(view, colorize_instances(mask))
                self.assertEqual(read_own_png(out/'review'/f'{path.stem}.png').shape, (128, 384, 3))
            calls = FakeScene.calls; self.run_capture(out, ['--resume'])
            self.assertEqual(FakeScene.calls, calls)
            with self.assertRaises(RuntimeError): self.run_capture(out)
            with self.assertRaises(RuntimeError): self.run_capture(out, ['--resume', '--light-scale', '0.8'])

    def test_old_revision_refused(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'pilot'; self.run_capture(out)
            p = out/'manifest.json'; config = json.loads(p.read_text()); config.pop('generator_revision'); p.write_text(json.dumps(config))
            with self.assertRaises(RuntimeError): self.run_capture(out, ['--resume'])

    def test_cli_help_no_simulator_import(self):
        result = subprocess.run([sys.executable, str(ROOT/'generate_dataset.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--ground-root', result.stdout)
        self.assertIn('--light-scale', result.stdout)

    def test_cli_bad_scale_rejected(self):
        for value in ['nan', '0', '-1', '5']:
            with patch.object(sys, 'argv', ['x', '--blank-scene', '--out', '/tmp/unused', '--light-scale', value]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): generate_dataset.arguments()

    def test_numpy_before_isaac_and_no_new_runtime_dependencies(self):
        text = (ROOT/'generate_dataset.py').read_text()
        self.assertLess(text.index('import numpy as np'), text.index('from isaacsim import SimulationApp'))
        for name in ['generate_dataset.py', 'isaac_scene.py', 'appearance.py', 'usd_appearance.py']:
            code = (ROOT/name).read_text()
            ast.parse(code, feature_version=(3, 11))
            for forbidden in ['--no-ros-env', 'import cv2', 'import torch', 'from PIL', '.Save(']:
                self.assertNotIn(forbidden, code)


if __name__ == '__main__':
    unittest.main(verbosity=2)
