from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import CLASSES, FORMAT, SPLITS, camera_matrix, decode_instances, dump_json, project, rotation, sample_objects, sha256, split_schedule, world_extents, write_png
from prepare_dataset import convert_frame, mask_polygon, prepare
from train_model import resolve_data, training_options


def raw_fixture(root, frames=30):
    dump_json(root / "manifest.json", {"format": FORMAT, "classes": list(CLASSES), "settings": {"frames": frames}})
    dump_json(root / "progress.json", {"complete": True})
    for index, split in enumerate(split_schedule(frames, 9)):
        stem = f"{index:06d}"
        rgb = np.full((100, 160, 3), 30+index*3, np.uint8)
        mask = np.zeros((100, 160), np.uint16)
        objects = []
        for cid, name in enumerate(CLASSES):
            mask[25:65, 8+cid*50:38+cid*50] = cid+1
            objects.append({"instance_id": cid+1, "shape": name, "class_id": cid, "visible_pixels": 1200})
        image = root / "images" / split / f"{stem}.png"
        labels = root / "masks" / split / f"{stem}.png"
        write_png(image, rgb); write_png(labels, mask)
        dump_json(root / "meta" / split / f"{stem}.json", {"split": split, "objects": objects,
                  "resolution": [160, 100], "negative": False,
                  "rgb_sha256": sha256(image), "mask_sha256": sha256(labels)})


class ImageFormatTests(unittest.TestCase):
    def test_rgb_png_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            rgb = np.random.default_rng(5).integers(0, 256, (23, 37, 3), dtype=np.uint8)
            path = Path(d)/"a.png"; write_png(path, rgb)
            np.testing.assert_array_equal(cv2.imread(str(path))[..., ::-1], rgb)

    def test_uint16_png_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            mask = np.array([[0, 1, 255], [256, 40000, 65535]], dtype=np.uint16)
            path = Path(d)/"m.png"; write_png(path, mask)
            np.testing.assert_array_equal(cv2.imread(str(path), -1), mask)

    def test_invalid_png_rejected(self):
        with self.assertRaises(ValueError): write_png("unused.png", np.ones((4, 4)))

    def test_instance_ids_resolved_per_frame(self):
        obj = {"slot": 0, "prim_path": "/Target/item"}
        for rid in (7, 510):
            data = {"data": np.array([[rid, 0], [rid, rid]], np.uint32), "info": {"idToLabels": {str(rid): "/Target/item", "0": "BACKGROUND"}}}
            mask, entries = decode_instances(data, [obj])
            self.assertEqual(mask.dtype, np.uint16); self.assertEqual(entries[0]["visible_pixels"], 3)
            self.assertEqual(mask.max(), 1)

    def test_descendant_prim_mapping(self):
        mask, _ = decode_instances({"data": np.full((2, 2), 9, np.uint32), "info": {"idToLabels": {"9": {"primPath": "/A/item/mesh"}}}}, [{"slot": 3, "prim_path": "/A/item"}])
        self.assertTrue((mask == 4).all())

    def test_class_labels_not_mistaken_for_paths(self):
        mask, _ = decode_instances({"data": np.full((2, 2), 9, np.uint32), "info": {"idToLabels": {"9": {"class": "cube"}}}}, [{"slot": 0, "prim_path": "/A/item"}])
        self.assertFalse(mask.any())

    def test_missing_mapping_fails_closed(self):
        with self.assertRaises(RuntimeError): decode_instances({"data": np.zeros((10, 10), np.uint32)}, [])


class SamplingTests(unittest.TestCase):
    def test_split_sizes_and_reproducibility(self):
        schedule = split_schedule(1200, 31)
        self.assertEqual([schedule.count(s) for s in SPLITS], [960, 120, 120])
        self.assertEqual(schedule, split_schedule(1200, 31))
        self.assertNotEqual(schedule, split_schedule(1200, 32))

    def test_too_few_samples_rejected(self):
        with self.assertRaises(ValueError): split_schedule(5, 0)

    def test_camera_projection_center(self):
        T = camera_matrix([.9, -.7, .8], [.45, 0, .04])
        K = np.array([[480., 0, 320.], [0, 480., 240.], [0, 0, 1.]])
        np.testing.assert_allclose(project([.45, 0, .04], T, K), [320, 240], atol=1e-8)
        self.assertIsNone(project(T[:3, 3] + T[:3, 2], T, K))

    def test_rotation_lying_axis_and_orthogonality(self):
        R = rotation(.6, True)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(R[2, 2], 0.)
        np.testing.assert_allclose(R[:, 2], [np.cos(.6), np.sin(.6), 0.])

    def test_upright_cylinder_extents_ignore_yaw(self):
        for yaw in np.linspace(-3., 3., 17):
            np.testing.assert_allclose(world_extents("cylinder", [.04, .04, .10], rotation(yaw)), [.04, .04, .10], atol=1e-10)

    def test_lying_cylinder_height_equals_diameter(self):
        for yaw in np.linspace(-3., 3., 17):
            extents = world_extents("cylinder", [.04, .04, .10], rotation(yaw, True))
            self.assertAlmostEqual(extents[2], .04)

    def test_noninterpenetrating_visible_balanced_samples(self):
        T = camera_matrix([.9, -.7, .8], [.45, 0., .04])
        K = np.array([[480., 0, 320.], [0, 480., 240.], [0, 0, 1.]])
        types, elevated = set(), 0
        for seed in range(120):
            items = sample_objects(np.random.default_rng(seed), 5, CLASSES[seed % 3], [.3, -.18, .62, .18], 0., T, K, [640, 480])
            self.assertEqual(items[0]["shape"], CLASSES[seed % 3])
            for i, obj in enumerate(items):
                types.add(obj["shape"]); elevated += obj["elevated"]
                self.assertGreaterEqual(obj["position"][2], obj["size_world_aabb"][2]/2-1e-10)
                uv = project(obj["position"], T, K)
                self.assertTrue(12 <= uv[0] < 628 and 12 <= uv[1] < 468)
                for other in items[:i]:
                    self.assertGreaterEqual(np.linalg.norm(np.array(obj["position"][:2])-other["position"][:2]), obj["footprint"]+other["footprint"]+.004-1e-10)
        self.assertEqual(types, set(CLASSES)); self.assertGreater(elevated, 0)


class PolygonTests(unittest.TestCase):
    def check_shape(self, mask):
        poly, original, loaded = mask_polygon(mask)
        self.assertGreaterEqual(original, .97); self.assertGreaterEqual(loaded, .90)
        self.assertTrue(np.isfinite(poly).all() and (poly >= 0).all() and (poly <= 1).all())
        self.assertGreaterEqual(len(poly), 3)

    def test_box_polygon(self):
        m = np.zeros((100, 160), np.uint8); m[20:70, 30:80] = 1; self.check_shape(m)

    def test_sphere_contour(self):
        m = np.zeros((100, 160), np.uint8); cv2.circle(m, (70, 50), 25, 1, -1); self.check_shape(m)

    def test_occluded_disconnected_single_instance(self):
        m = np.zeros((100, 160), np.uint8); m[20:65, 20:45] = 1; m[25:55, 58:75] = 1; self.check_shape(m)

    def test_hole_preserved(self):
        m = np.zeros((100, 160), np.uint8); m[10:80, 20:110] = 1; m[30:55, 45:65] = 0; self.check_shape(m)

    def test_small_visible_fragment_rejected(self):
        m = np.zeros((30, 30), np.uint8); m[3:5, 3:5] = 1
        with self.assertRaises(ValueError): mask_polygon(m)


class DatasetTests(unittest.TestCase):
    def test_end_to_end_export_and_movable_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d)/"raw", Path(d)/"yolo"
            raw_fixture(raw)
            stats = prepare(raw, out)
            self.assertEqual(sum(s["images"] for s in stats.values()), 30)
            self.assertTrue((out/"review.jpg").is_file())
            for path in (out/"labels").glob("*/*.txt"):
                lines = path.read_text().splitlines(); self.assertEqual(len(lines), 3)
                for line in lines:
                    tokens = line.split(); self.assertGreaterEqual(len(tokens), 7); self.assertEqual(len(tokens) % 2, 1)
            cfg = yaml.safe_load((out/"data.yaml").read_text()); cfg["path"] = "/old/computer/path"
            (out/"data.yaml").write_text(yaml.safe_dump(cfg))
            resolved = resolve_data(out/"data.yaml")
            self.assertEqual(yaml.safe_load(Path(resolved).read_text())["path"], out.as_posix())

    def test_reject_whole_bad_image_not_one_positive(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d)/"raw", Path(d)/"out"; raw_fixture(raw)
            bad = next((raw/"meta").glob("*/*.json"))
            meta = json.loads(bad.read_text()); meta["objects"][0]["visible_pixels"] = 2; dump_json(bad, meta)
            stats = prepare(raw, out)
            self.assertEqual(sum(s["images"] for s in stats.values()), 29)
            self.assertFalse((out/"images"/bad.parent.name/f"{bad.stem}.png").exists())
            self.assertEqual(len(json.loads((out/"conversion_report.json").read_text())["rejected"]), 1)

    def test_incomplete_capture_refused(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d)/"raw"; raw_fixture(raw); dump_json(raw/"progress.json", {"complete": False})
            with self.assertRaises(ValueError): prepare(raw, Path(d)/"out")

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d)/"raw", Path(d)/"out"; raw_fixture(raw)
            out.mkdir(); (out/"important.txt").write_text("keep")
            with self.assertRaises(ValueError): prepare(raw, out)
            self.assertEqual((out/"important.txt").read_text(), "keep")

    def test_true_negative_exports_empty_label(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d)/"raw", Path(d)/"out"; raw_fixture(raw)
            path = next((raw/"meta").glob("*/*.json")); meta = json.loads(path.read_text()); split = meta["split"]
            mask_path = raw/"masks"/split/f"{path.stem}.png"; write_png(mask_path, np.zeros((100, 160), np.uint16))
            meta.update(objects=[], negative=True, mask_sha256=sha256(mask_path)); dump_json(path, meta)
            stats = prepare(raw, out)
            self.assertEqual((out/"labels"/split/f"{path.stem}.txt").read_text(), "")
            self.assertEqual(stats[split]["negative_images"], 1)

    def test_checksum_tampering_detected(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d)/"raw", Path(d)/"out"; raw_fixture(raw)
            path = next((raw/"meta").glob("*/*.json")); meta = json.loads(path.read_text()); meta["rgb_sha256"] = "wrong"; dump_json(path, meta)
            with self.assertRaisesRegex(ValueError, "checksum"): convert_frame(raw, path, out)


class LaunchTests(unittest.TestCase):
    def test_help_requires_no_isaac_or_torch(self):
        for script in ("generate_dataset.py", "train_model.py", "check_environment.py"):
            result = subprocess.run([sys.executable, str(ROOT/script), "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_fp32_low_memory_windows_defaults(self):
        args = SimpleNamespace(smoke=False, epochs=60, batch=2, name="test")
        options = training_options(args, "data.yaml")
        self.assertFalse(options["amp"]); self.assertEqual(options["workers"], 0)
        self.assertEqual(options["batch"], 2); self.assertEqual(options["imgsz"], 640)
        self.assertEqual(options["mask_ratio"], 2)

    def test_numpy_import_precedes_simulation_app(self):
        tree = ast.parse((ROOT/"generate_dataset.py").read_text())
        imports = [(node.lineno, node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        np_line = min(line for line, node in imports if isinstance(node, ast.Import) and any(a.name == "numpy" for a in node.names))
        app_line = min(line for line, node in imports if isinstance(node, ast.ImportFrom) and node.module == "isaacsim")
        self.assertLess(np_line, app_line)

    def test_no_rejected_ros_flag_or_core_imports(self):
        for name in ("generate_dataset.py", "isaac_scene.py"):
            text = (ROOT/name).read_text()
            self.assertNotIn("--no-ros-env", text)
            self.assertNotIn("robot_skill_stack.", text)
            self.assertNotIn(".Save(", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
