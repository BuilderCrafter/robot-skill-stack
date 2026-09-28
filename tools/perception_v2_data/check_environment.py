"""Validate the HOME-PC environment and actual CUDA kernels, not just GPU visibility."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo-config"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def check(download_model=False, model="yolov8n-seg.pt"):
    import numpy as np
    import torch
    import torchvision
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run with this toolkit's PC .venv. Do not install these packages in Isaac.")
    expected = {"torch": "2.7.1", "torchvision": "0.22.1", "ultralytics": "8.3.161", "numpy": "1.26.4"}
    versions = {k: importlib.metadata.version(k) for k in expected}
    if any(versions[k].split("+")[0] != v for k, v in expected.items()):
        raise RuntimeError(f"Unexpected package versions: {versions}. Run the provided setup script.")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("This pinned training setup expects Python 3.11")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check NVIDIA driver and the cu126 PyTorch install; no CPU fallback will be used.")
    torch.cuda.set_device(0)
    report = {"python": sys.version, "platform": platform.platform(), "packages": versions,
              "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "compute_capability": list(torch.cuda.get_device_capability(0)), "compiled_architectures": torch.cuda.get_arch_list()}
    try:
        report["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["nvidia_smi"] = "not available via PATH"
    layer = torch.nn.Conv2d(3, 8, 3).cuda()
    x = torch.randn(2, 3, 128, 128, device="cuda", requires_grad=True)
    layer(x).square().mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(layer.weight.grad).all():
        raise RuntimeError("Nonfinite CUDA backward gradients")
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda")
    kept = torchvision.ops.nms(boxes, torch.tensor([.9, .8], device="cuda"), .5)
    if kept.cpu().tolist() != [0]:
        raise RuntimeError("CUDA torchvision NMS failed")
    report["conv_forward_backward"] = report["torchvision_cuda_nms"] = "PASS"
    del x, layer
    torch.cuda.empty_cache()
    if download_model:
        from ultralytics import YOLO
        weights = ROOT / "weights" / model
        weights.parent.mkdir(exist_ok=True)
        network = YOLO(str(weights))
        network.predict(np.zeros((320, 320, 3), dtype=np.uint8), imgsz=320, device=0, half=False, verbose=False, save=False)
        report["pretrained_model_cuda_inference"] = "PASS"
        report["weights"] = str(weights)
    (ROOT / "environment_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Environment check passed. The separate --smoke run tests the real segmentation training path.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--model", choices=["yolov8n-seg.pt", "yolov8s-seg.pt"], default="yolov8n-seg.pt")
    args = parser.parse_args()
    check(args.download_model, args.model)
