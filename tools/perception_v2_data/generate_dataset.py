"""Run with the EXISTING project run_isaac_python.sh. No pip installs in Isaac."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scene", type=Path, default=Path("scenes/playground.usd"))
    p.add_argument("--camera", default="/World/PerceptionCamera")
    p.add_argument("--objects-root", default="/World/Objects")
    p.add_argument("--blank-scene", action="store_true", help="Explicit fallback; no saved robot or scene")
    p.add_argument("--frames", type=int, default=1200)
    p.add_argument("--seed", type=int, default=34017)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--support-z", type=float, default=0.)
    p.add_argument("--bounds", type=float, nargs=4, default=[.30, -.18, .62, .18], metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    p.add_argument("--max-objects", type=int, default=5)
    p.add_argument("--subframes", type=int, default=4)
    p.add_argument("--save-depth", action="store_true", help="Optional float32 depth NPZ per frame; training itself uses RGB")
    p.add_argument("--gui", action="store_true", help="Show this separate capture process; not the live project UI")
    p.add_argument("--resume", action="store_true", help="Same settings and seed only; skip committed frames")
    p.add_argument("--zip", action="store_true", help="Package the completed output directory next to itself")
    a = p.parse_args()
    if a.frames < 12 or min(a.width, a.height) < 128 or not 1 <= a.max_objects <= 20 or a.subframes < 1:
        p.error("frames >= 12, resolution >= 128, 1 <= max-objects <= 20, subframes >= 1 required")
    if a.bounds[0] >= a.bounds[2] or a.bounds[1] >= a.bounds[3]:
        p.error("Invalid XY bounds")
    a.scene, a.out = a.scene.resolve(), a.out.resolve()
    if not a.blank_scene and not a.scene.is_file():
        p.error(f"Scene missing: {a.scene}; run from the repo root or pass its path")
    return a


def main():
    args = arguments()
    # Deliberately lock the working repo NumPy BEFORE SimulationApp/Replicator.
    import numpy as np
    if sys.version_info[:2] != (3, 11) or np.__version__ != "1.26.4":
        raise RuntimeError(f"Use the existing project wrapper: expected Python 3.11 + NumPy 1.26.4; got {sys.version.split()[0]} / {np.__version__} at {np.__file__}. Do not reinstall Isaac packages.")
    from common import CLASSES, FORMAT, SPLITS, decode_instances, dump_json, preview, sample_objects, sha256, split_schedule, write_png
    print(f"[SDG] Python {sys.version.split()[0]}; NumPy {np.__version__}: {np.__file__}", flush=True)
    scene_hash = None if args.blank_scene else sha256(args.scene)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("resume", "zip", "gui", "out")}
    config = {"format": FORMAT, "classes": list(CLASSES), "settings": settings, "scene_sha256": scene_hash}
    manifest = args.out / "manifest.json"
    if args.out.exists() and any(args.out.iterdir()):
        if not args.resume or not manifest.exists() or json.loads(manifest.read_text()) != config:
            raise RuntimeError("Output exists. Choose a new --out, or --resume with exactly the original settings. Nothing overwritten.")
    dump_json(manifest, config)
    for split in SPLITS:
        for kind in ("images", "masks", "meta") + (("depth",) if args.save_depth else ()):
            (args.out / kind / split).mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0]]  # Do not forward this tool's CLI flags into Kit.
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not args.gui, "renderer": "RaytracedLighting", "extra_args": [
        "--/isaac/startup/ros_bridge_extension=", "--/isaac/startup/ros_sim_control_extension=false",
        "--enable", "omni.replicator.core"]})
    scene, started, completed, newly_written = None, time.perf_counter(), 0, 0
    per_split, class_pixels = dict.fromkeys(SPLITS, 0), dict.fromkeys(CLASSES, 0)
    try:
        from isaac_scene import CaptureScene
        scene = CaptureScene(app, args)
        schedule = split_schedule(args.frames, args.seed)
        ordinal = dict.fromkeys(SPLITS, 0)
        for index, split in enumerate(schedule):
            if not app.is_running():
                raise KeyboardInterrupt("Capture window closed; rerun the same command with --resume")
            sequence = ordinal[split]
            ordinal[split] += 1
            stem = f"{index:06d}"
            meta_path = args.out / "meta" / split / f"{stem}.json"
            rgb_path = args.out / "images" / split / f"{stem}.png"
            mask_path = args.out / "masks" / split / f"{stem}.png"
            depth_path = args.out / "depth" / split / f"{stem}.npz"
            if args.resume and meta_path.exists() and rgb_path.exists() and mask_path.exists() and (not args.save_depth or depth_path.exists()):
                meta = json.loads(meta_path.read_text())
            else:
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, index, SPLITS.index(split)]))
                T = scene.choose_camera(rng)
                # First three examples in each split include each category; ~10% negatives afterward.
                negative = sequence >= 3 and sequence % 10 == 9
                count = 0 if negative else int(rng.integers(1, args.max_objects+1))
                for attempt in range(6):
                    objects = sample_objects(rng, count, CLASSES[sequence % 3], args.bounds, args.support_z, T, scene.K, (args.width, args.height))
                    scene.apply(objects, rng, T)
                    data = scene.capture()
                    rgb = np.asarray(data["rgb"])
                    if rgb.shape != (args.height, args.width, 4) and rgb.shape != (args.height, args.width, 3):
                        raise RuntimeError(f"Unexpected RGB shape {rgb.shape}")
                    if rgb.dtype != np.uint8:
                        raise RuntimeError(f"Expected uint8 RGB annotator, got {rgb.dtype}")
                    rgb = rgb[..., :3].copy()
                    mask, entries = decode_instances(data["instances"], objects)
                    if mask.shape != rgb.shape[:2]:
                        raise RuntimeError("RGB/instance-mask shape mismatch")
                    if negative or (entries and entries[0]["visible_pixels"] >= 45):
                        break
                    if attempt == 5:
                        mapping = data["instances"].get("info", {})
                        dump_json(args.out / "annotator_failure.json", {"info": str(mapping), "objects": objects})
                        raise RuntimeError("Targets repeatedly invisible or unmapped; inspect annotator_failure.json and camera. Stopping rather than saving false negatives.")
                if args.save_depth:
                    depth = np.squeeze(np.asarray(data["depth"], dtype=np.float32))
                    if depth.shape != mask.shape:
                        raise RuntimeError("Depth/image shape mismatch")
                    # Float32 is preserved; invalid depth stays inf/NaN, not a false 0 m surface.
                    with depth_path.with_suffix(".tmp").open("wb") as f:
                        np.savez_compressed(f, depth=depth)
                    depth_path.with_suffix(".tmp").replace(depth_path)
                write_png(rgb_path, rgb)
                write_png(mask_path, mask)
                cv_T = T @ np.diag([1., -1., -1., 1.])
                meta = {"frame_id": index, "scene_seed": [args.seed, index, SPLITS.index(split)], "split": split,
                        "resolution": [args.width, args.height], "K": scene.K.tolist(),
                        "T_world_camera_opencv": cv_T.tolist(), "camera_convention": "column vectors; X right, Y down, Z forward",
                        "depth_type": "distance_to_image_plane_m" if args.save_depth else None,
                        "support_z": args.support_z, "negative": negative, "objects": entries,
                        "rgb_sha256": sha256(rgb_path), "mask_sha256": sha256(mask_path)}
                dump_json(meta_path, meta)  # Metadata commits a complete RGB + mask (+depth) frame.
                if index < 24 or index % 100 == 0:
                    write_png(args.out / "preview" / f"{stem}.png", preview(rgb, mask, entries))
                newly_written += 1
            completed += 1
            per_split[split] += 1
            for obj in meta["objects"]:
                class_pixels[obj["shape"]] += obj["visible_pixels"]
            if completed % 10 == 0 or completed == 1 or completed == args.frames:
                elapsed = time.perf_counter()-started
                eta = (args.frames-completed)*elapsed/max(newly_written, 1)
                dump_json(args.out / "progress.json", {"completed": completed, "requested": args.frames, "splits": per_split,
                          "class_pixels": class_pixels, "elapsed_s": elapsed, "complete": completed == args.frames})
                print(f"[SDG] {completed}/{args.frames}; {elapsed:.0f}s elapsed; approx {eta/60:.1f} min left; pixels {class_pixels}", flush=True)
        if not all(class_pixels.values()):
            raise RuntimeError("A category has no visible annotations. Inspect the pilot before training.")
    finally:
        try:
            if scene is not None:
                scene.close()
        finally:
            app.close()
        if scene_hash is not None and sha256(args.scene) != scene_hash:
            raise RuntimeError("Source scene hash changed during capture; investigate before continuing")
    if args.zip:
        archive = shutil.make_archive(str(args.out), "zip", args.out.parent, args.out.name)
        print(f"[SDG] Transfer this archive to your PC: {archive}", flush=True)
    print(f"[SDG] Complete. Inspect previews in {args.out / 'preview'}. Red=cube, blue=sphere, green=cylinder.")


if __name__ == "__main__":
    main()
