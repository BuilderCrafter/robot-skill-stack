"""HOME PC only: fine-tune one pretrained three-class YOLOv8 instance-segmentation model."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo-config"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("WANDB_MODE", "disabled")


def resolve_data(path, smoke=False):
    import yaml
    from common import CLASSES
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names")
    if names != dict(enumerate(CLASSES)) and names != list(CLASSES):
        raise ValueError("Expected class order cube, sphere, cylinder")
    # The dataset is movable between PCs; do not trust the old absolute YAML root.
    data["path"] = path.parent.as_posix()
    for split in ("train", "val", "test"):
        folder = path.parent / "images" / split
        if not folder.is_dir() or not any(folder.glob("*.png")):
            raise ValueError(f"Missing split: {folder}")
        data[split] = f"images/{split}"
    suffix = "smoke" if smoke else "resolved"
    if smoke:
        for split, limit in (("train", 128), ("val", 32)):
            files = sorted((path.parent / "images" / split).glob("*.png"))[:limit]
            listing = path.parent / f"smoke_{split}.txt"
            listing.write_text("\n".join(p.as_posix() for p in files) + "\n", encoding="utf-8")
            data[split] = listing.as_posix()
    target = path.parent / f"data_{suffix}.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(target)


def training_options(args, data):
    return dict(data=data, epochs=2 if args.smoke else args.epochs, batch=args.batch, imgsz=640,
                device=0, workers=0, amp=False, cache=False, mask_ratio=2, overlap_mask=True,
                optimizer="AdamW", lr0=.001, nbs=16, patience=15, seed=42, deterministic=True,
                mosaic=0., mixup=0., copy_paste=0., degrees=0., translate=.05, scale=.15,
                fliplr=.5, flipud=0., hsv_h=.01, hsv_s=.35, hsv_v=.25, close_mosaic=0,
                plots=True, val=True, save=True, save_period=5, project=str(ROOT / "runs"),
                name="smoke" if args.smoke else args.name, exist_ok=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", choices=["yolov8n-seg.pt", "yolov8s-seg.pt"], default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--name", default="primitives_nano")
    parser.add_argument("--smoke", action="store_true", help="2 epochs, at most 128 training/32 validation images; plumbing check only")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted last.pt, not a finished smoke run")
    parser.add_argument("--evaluate", type=Path, help="Evaluate these weights on the untouched test split; no training")
    args = parser.parse_args()
    if args.batch < 1 or args.epochs < 1 or (not args.resume and args.data is None):
        parser.error("Positive batch/epochs and --data (unless --resume) required")
    if args.resume and (args.smoke or args.evaluate):
        parser.error("--resume cannot be combined with --smoke/--evaluate")
    from check_environment import check
    check()
    from ultralytics import YOLO
    if args.resume:
        if not args.resume.is_file():
            parser.error("Resume checkpoint not found")
        model = YOLO(str(args.resume.resolve()))
        model.train(resume=True, device=0, workers=0, amp=False)
        return
    data = resolve_data(args.data, args.smoke)
    if args.evaluate:
        model = YOLO(str(args.evaluate.resolve()))
        model.val(data=data, split="test", imgsz=640, batch=args.batch, device=0, workers=0, half=False, plots=True,
                  project=str(ROOT / "runs"), name="test_evaluation")
        return
    path = ROOT / "weights" / args.model
    path.parent.mkdir(exist_ok=True)
    model = YOLO(str(path))
    if model.task != "segment":
        raise ValueError("Expected segmentation weights")
    samples, started = [], [0.]

    def begin(trainer):
        started[0] = time.perf_counter()

    def end(trainer):
        dt = time.perf_counter() - started[0]
        samples.append(dt)
        if len(samples) > 1:
            normal = statistics.median(samples[1:][-5:])
            remaining = max(0, trainer.epochs-trainer.epoch-1)*normal/60
            print(f"[Timing] epoch {trainer.epoch+1}: {dt:.1f}s; estimated remaining {remaining:.1f} min", flush=True)

    model.add_callback("on_train_epoch_start", begin)
    model.add_callback("on_fit_epoch_end", end)
    options = training_options(args, data)
    model.train(**options)
    save_dir = Path(model.trainer.save_dir)
    (save_dir / "epoch_times.json").write_text(json.dumps({"epoch_seconds": samples, "smoke_only": args.smoke}, indent=2), encoding="utf-8")
    print(f"Weights: {save_dir / 'weights' / 'best.pt'}")
    print("Smoke run validates execution only; start the full run from PRETRAINED weights, not smoke weights." if args.smoke else
          "Review training results; use --evaluate best.pt --data ... once for the held-out test set. V1 remains untouched.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
