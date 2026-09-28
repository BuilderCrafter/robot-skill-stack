# Primitive synthetic data + GTX 1060 training starter

An **additive offline toolkit**, not a V2 runtime replacement. Extract the archive
at your repository root: it adds only `tools/perception_v2_data/`. It does not
modify launchers, `.deps`, NumPy, V1, UI, skills, or gripper state. No rejected ROS
command-line option is used. Do **not** install the PC requirements in Isaac.

## 1. University workstation: generate a pilot

Close the running Isaac GUI first to free GPU memory. Use the existing project
wrapper, from the repository root:

```bash
cd /home/etfrobotics/Projects/ga253315m/robot-skill-stack
bash ./run_isaac_python.sh tools/perception_v2_data/generate_dataset.py \
  --scene scenes/playground.usd --frames 24 \
  --out outputs/primitive_pilot --save-depth
```

Inspect `outputs/primitive_pilot/preview/` (PNG overlays). Overlay colors:
**red=cube, blue=sphere, green=cylinder**. The object materials themselves are
random, not color-coded. Check that EVERY visible target has a correct outline,
that there are two outlines for two same-class objects, and that the overlays
match the RGB. Do not train through obviously incorrect labels.

The generator uses the saved scene's `/World/PerceptionCamera`, with 640x480
capture. It hides `/World/Objects` only in an unsaved session layer, retains the
saved robot/environment, and creates its own temporary target pool. The original
scene is never saved and its root-file hash is checked at exit. Other target
objects outside `/World/Objects` must not remain in the source scene unlabelled.
The scene must use meters, Z-up, a perspective camera, and a support plane at
Z=0 (the current project convention). Pass `--support-z` for another table.
Aperture offsets must be zero; this initial exporter assumes no lens distortion.

No OpenCV/Pillow/PyTorch is installed or imported in Isaac. A tiny stdlib PNG
writer saves images; label conversion runs on the PC. NumPy **1.26.4** is imported
before `SimulationApp`, using the existing wrapper's dependency path. This is a
separate standalone capture process, not a GUI callback or live-runtime command.

## 2. Generate the first training batch

```bash
bash ./run_isaac_python.sh tools/perception_v2_data/generate_dataset.py \
  --scene scenes/playground.usd --frames 1200 \
  --out outputs/primitive_1200 --save-depth --zip
```

Transfer `outputs/primitive_1200.zip` to the home PC. Generation prints measured
progress and an approximate remaining time. Do not assume 1200 images finish in
any fixed number of minutes: renderer startup, scene assets, disk, and GPU matter.
Use `--frames 600` with a *new output name* for a shorter first job. Changing the
requested frame count invalidates resumption; do not reuse an old output folder.

If interrupted, rerun the **same** command/settings with `--resume`. Completed
RGB/mask/depth/metadata bundles are reused; the scene random seed is per frame.
Outputs are never silently deleted. Full output is zipped only after completion.

Optional switches:
- `--gui`: show the separate capture process rather than headless capture.
- `--bounds .30 -.18 .62 .18`: target-center XY sampling area, meters.
- `--camera /World/PerceptionCamera`: override the saved camera path.
- `--subframes 4`: default; helps rendering after repositioning objects. Do not
  reduce it before checking the resulting masks and motion ghosting.
- `--blank-scene`: explicit fallback with a generated floor/camera and **no real
  robot background**. Never silently substituted for the saved scene.
- Omit `--save-depth` to reduce transfer size; RGB training does not need depth.

### What the initial dataset covers (and does not)

Cubes: 28–66 mm sides, yaw randomized. Spheres: 28–66 mm diameter. Cylinders:
28–66 mm diameter and 40–120 mm length, upright or lying. Each scene contains
1–5 targets, except about 10% intentionally empty/negative frames. Multiple
objects of the same class are independent instances. Cone distractors appear in
about 40% of scenes. Target placement avoids interpenetration via conservative
footprint separation. Colors, roughness, light intensity, and small camera
translations vary; the actual saved camera pose is retained in 75% of scenes.

Approximately 20% of target placements are elevated. **These are analytically
placed static renderings, not physically simulated grasps or manipulation
videos.** The real saved robot is included but its joints are not randomized.
Edge occlusion can arise from the robot/objects/cones; moving-finger/true grasp
sequences and broad background/material diversity need a later dataset expansion.
This starter is a three-class segmentation baseline, not evidence that tracking
or gripper verification has been solved. Test against real project captures before
considering replacing V1.

Frames are independently randomized using per-frame seeds; 1200 gives 960 train,
120 validation, and 120 test frames before conversion rejection. This avoids an
adjacent-video-frame split, but all still share the same source scene family.
Keep separate actual manipulation episodes for a stronger final evaluation.

### Raw output layout

```
primitive_1200/
  manifest.json          # generation settings, source scene hash, class order
  progress.json          # must say complete before conversion/training
  images/{train,val,test}/*.png
  masks/{train,val,test}/*.png   # uint16 visible INSTANCE IDs, 0 = background
  meta/{train,val,test}/*.json   # per-ID class/pose/size, K, camera transform, hashes
  depth/{train,val,test}/*.npz   # optional aligned float32 depth
  preview/*.png                 # inspection only, never training images
```

Instance numeric values from Replicator are resolved from that frame's prim-path
mapping. They are not mistaken for semantic class IDs and are not assumed stable
between frames. Cubes=class 0, spheres=1, cylinders=2. Ground-truth metadata belongs
to training/evaluation, never the operational perception provider.

## 3. Home PC (Windows, GTX 1060 6 GB): isolated environment

Extract this same toolkit on the PC, and open PowerShell **in the directory
containing this README** (`tools\perception_v2_data`). You do not need the complete
robot repository or Isaac on the PC. Python 3.11 64-bit with the `py` launcher is
required. Check `py -3.11 --version`. If missing, install Python 3.11, including the
launcher; for example `winget install -e --id Python.Python.3.11`, then reopen the
terminal. If winget is unavailable, use the official Python installer.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

The execution-policy override applies to that PowerShell process only; the script
does not change the machine policy. It creates `.venv` in this toolkit folder,
installs **torch 2.7.1 / torchvision 0.22.1 from the CUDA 12.6 index**, and installs
pinned Ultralytics 8.3.161 / NumPy 1.26.4 / OpenCV 4.11.0.86 plus dependencies.
It runs a CUDA convolution forward/backward, torchvision CUDA NMS, and downloads
`yolov8n-seg.pt` and runs one CUDA prediction. Results go to
`environment_check.json`; resolved packages go to `training-environment-lock.txt`.
No successful GTX 1060 execution is claimed before *your* check passes.

Do not replace the wheel index with an unqualified latest PyTorch/CUDA install.
GTX 1060 is Pascal, compute capability 6.1. CUDA architecture coverage differs
between wheel builds. This configuration is chosen for Pascal support. You need
a compatible NVIDIA driver; a separate CUDA toolkit/compiler install is not part
of this workflow. A driver or missing-kernel error must be resolved before training.

Optional home-PC Linux: with Python 3.11 + venv support available, use
`bash setup_linux.sh`, then `.venv/bin/python` instead of the Windows Python path.
Do not run this installer in the university Isaac environment.

## 4. Convert and review the data on the PC

Copy the generated ZIP to Downloads, then from the toolkit directory:

```powershell
Expand-Archive "$HOME\Downloads\primitive_1200.zip" -DestinationPath .\data
.\.venv\Scripts\python.exe prepare_dataset.py --raw .\data\primitive_1200 --out .\data\primitive_1200_yolo
Invoke-Item .\data\primitive_1200_yolo\review.jpg
```

The converter verifies hashes, instance identities, image/mask alignment, labels,
and class coverage. It outputs one YOLO polygon row **per instance**, not bounding
boxes or one row per fragment. Disconnected contours and holes use zero-width
bridges; rasterized polygons are checked against the exact masks, including a
resampling check approximating the pinned training loader. Minimum fidelity is
0.97 before / 0.90 after resampling. Noisy, tiny, or unrepresentable targets reject
the **whole frame**, not just that target, avoiding unlabeled positive objects.
Exact raw masks remain available. The format is still an approximation, not an
exact arbitrary-mask representation. Check `conversion_report.json` and review.jpg.

Only successful datasets receive `data.yaml`. A missing class in a split or an
excessive rejection rate stops preparation. Use a new output directory after
fixing bad inputs; the converter will not delete previous results automatically.

## 5. Run a short smoke test, then real training

```powershell
.\.venv\Scripts\python.exe train_model.py --data .\data\primitive_1200_yolo\data.yaml --smoke
.\.venv\Scripts\python.exe train_model.py --data .\data\primitive_1200_yolo\data.yaml --epochs 60 --batch 2 --name primitives_nano
```

Smoke uses 2 epochs with at most 128 training / 32 validation images. It validates
real segmentation training but is **not** a quality evaluation or full-dataset
speed benchmark. The full run starts again from pretrained weights, not smoke
weights. Defaults: YOLOv8n-seg, 640 input, batch 2, workers 0, FP32/AMP off,
mask_ratio=2, AdamW, patience 15. Epoch timing and ETA are printed during the full
run. It may stop early on validation stagnation. Training completion within an
hour is not promised. Your goal is to start useful work in that setup window.

For CUDA out-of-memory, close GPU-heavy programs and retry with `--batch 1` and a
new `--name`. There is no automatic CPU fallback. Later, compare the larger
`--model yolov8s-seg.pt --batch 1 --name primitives_small`; nano quality is not
assumed sufficient merely because it trains faster. No DINO/SAM training is done.

The script prints the actual run directory; if a name already exists, Ultralytics
may increment it. Preserve `best.pt`, `last.pt`, `args.yaml`, `results.csv`, and
`epoch_times.json`. Evaluate the untouched test split only after selecting the
model on validation:

```powershell
.\.venv\Scripts\python.exe train_model.py --data .\data\primitive_1200_yolo\data.yaml --evaluate .\runs\primitives_nano\weights\best.pt
```

Resume an *interrupted* full run (not a completed smoke run):

```powershell
.\.venv\Scripts\python.exe train_model.py --resume .\runs\primitives_nano\weights\last.pt
```

Do not relocate/delete the dataset during an interrupted run: the checkpoint
contains its dataset configuration. Model package/weight licenses remain those
of their publishers. No model weights or third-party packages are redistributed
in this ZIP.

## Verification and boundaries

See `validation/`. Local tests cover PNG round-trips, annotation mapping,
class/split checks, seeded sampling, contour fidelity including occlusion,
conversion, negative labels, failure handling, and training settings/CLI imports.
They do **not** execute Isaac/Replicator, PowerShell package installation, CUDA,
or Ultralytics training on a GTX 1060. The workstation pilot and PC smoke run are
mandatory validation steps. This is not the perception-V2 runtime integration.

## Primary API references

- Isaac 5.1 Replicator capture/annotators/RT subframes:
  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_getting_started.html
- Isaac 5.1 automatic labeled/randomized SDG:
  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_object_based_sdg.html
- Isaac 5.1 semantic LabelsAPI helper:
  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.utils/docs/index.html
- Official PyTorch 2.7.1 / torchvision 0.22.1 CUDA-wheel commands:
  https://pytorch.org/get-started/previous-versions/
- PyTorch CUDA architecture support matrix:
  https://github.com/pytorch/pytorch/blob/main/RELEASE.md
- NVIDIA legacy GPU compute capabilities:
  https://developer.nvidia.com/cuda/gpus/legacy
- YOLO segmentation polygon format and training:
  https://docs.ultralytics.com/datasets/segment/
  https://docs.ultralytics.com/modes/train/
- Pinned Ultralytics package:
  https://pypi.org/project/ultralytics/8.3.161/
