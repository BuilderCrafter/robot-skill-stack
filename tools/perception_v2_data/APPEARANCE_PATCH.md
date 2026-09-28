# Synthetic-data appearance and mask-preview correction

Apply over `robot-skill-stack-sdg-training-starter.zip` at the repository root.
Only `tools/perception_v2_data/` is affected. The patch replaces the generator,
its scene adapter, the PC converter and toolkit README, and adds small appearance
helpers, tests and this report. No Python launcher, `.deps`, dependency pin,
saved USD/TOML, V1/V2 runtime, GUI, robot controller or training recipe is changed.

## Run the new pilot

Close the normal Isaac GUI to free GPU memory. From the project root:

```bash
bash ./run_isaac_python.sh tools/perception_v2_data/generate_dataset.py \
  --scene scenes/playground.usd \
  --frames 24 \
  --out outputs/primitive_pilot_v2 \
  --save-depth \
  --zip
```

Use a new output directory. Old captures cannot be resumed with this revision;
this prevents mixing appearance policies into an existing dataset by accident.
A new-revision interrupted capture can be resumed with exactly the same arguments
plus `--resume`. The source scene is never saved. All material/light bindings are
authored in its unsaved USD session layer and its root file hash is checked.

Inspect these outputs:

| Folder / file | Meaning |
|---|---|
| `images/<split>/` | Actual rendered RGB used for training; no overlays |
| `preview/` | RGB with class-colored outlines: red cube, blue sphere, green cylinder |
| `mask_preview/` | Bright, unique colors for instance IDs; black means background |
| `review/` | Left: RGB. Center: class overlay. Right: instance-color label map |
| `masks/<split>/` | Original lossless uint16 IDs, intentionally near-black in normal viewers |
| `quality_report.json` | Per-object visible area and measured local-contrast/clipping warnings |
| `appearance_setup.json` | Actual support prims selected, source lights disabled, policy settings |
| `meta/<split>/` | Existing labels/calibration plus sampled appearance and per-frame diagnostics |

The complete pilot archive will be `outputs/primitive_pilot_v2.zip`.
Do not feed the colorful preview images into training or overwrite the raw masks.
The same-class objects intentionally receive different instance-preview colors.
An intentionally empty/negative scene still has an all-black mask preview.

## Changes

### Appearance

- Neutral gray, blue-gray, slate and muted taupe support colors replace the
  uniformly pale background in most frames. The source support geometry is
  **not** moved, resized or replaced. Automatic discovery looks for a large thin
  visible surface at `--support-z` that spans the target sampling bounds.
- Object and cone-distractor colors come from the **same class-independent**
  distribution. Clear examples prefer saturated/darker colors separated from the
  sampled support color. This is a material-color sampling heuristic, not a
  guarantee that the rendered pixels will have a particular contrast.
- Display-space sRGB colors are converted to scene-linear values for the USD
  material. Roughness is 0.75–0.95 and metallic is zero; the ground is matte too.
- Scene lights no longer receive an arbitrary multiplier of their inherited
  intensity. The standalone capture uses a controlled dome fill plus directional
  key with varied direction, intensity and slight color tint. Auto exposure is
  disabled in that standalone process so it does not compensate for the lighting.
- Sampling probabilities are 70% clear, 20% varied and 10% deliberately hard,
  including pale-on-pale scenes. A 24-image pilot will not have exact percentages.
  We retain difficult examples instead of training only on colorful dark blobs.
- **Robot materials remain authored**, including the black joints and fingers.
  The new background/lighting provides separation without repainting the robot.
- Camera calibration, FOV, resolution, actual camera-pose distribution and target
  size ranges stay unchanged. This patch is not a hidden zoom/camera relocation.
  The quality report flags small instances for subsequent review.

The auto-detected ground paths are printed at startup. If no suitable surface
can be found, generation stops rather than silently leaving it white. Use
`--ground-root` with the actual support prim path to override detection. A
material-binding ancestor can be supplied when a stronger ancestor binding
prevents the leaf override; do not pass the complete scene or robot root.

### Annotation display

The raw masks were valid. Your uploaded pilot has **24 frames: 23 positive and
one intentional negative**, with **69 visible annotated instances** (22 cubes,
22 spheres, 25 cylinders). Mask IDs, metadata pixel counts and recorded RGB/mask
hashes agree. Instance pixel counts range from 75 to 763 (median 337).

The new colored views make those low-valued IDs visible without changing any
training data. Quality diagnostics use the rendered RGB, not the requested
material colors. Their local-background ring can include a neighboring surface,
shadow or robot: these are review hints, **not certified visibility measurements**.
Hard examples are not silently dropped. Review flags do not mean labels are missing.

`validation/appearance_fix/original_pilot_review.png` is an inspection of your
**original uploaded pilot**, not a rendered prediction of the new appearance.
The original pilot audit measured low local color contrast on 45 of 69 instances
using the new heuristic; that is baseline evidence, not a model-accuracy result.

### Small-mask converter correction

During checks, the previous converter rejected 12/24 of the uploaded pilot frames
at its resampled-polygon fidelity gate, despite raw polygon IoU = 1.0. Its
resampling simulation omitted original vertices and did not follow the float32
processing order used by the pinned Ultralytics 8.3.161 loader.

The corrected `prepare_dataset.py` retains those vertices, resamples normalized
coordinates, and denormalizes in float32. The thresholds are **not reduced**.
All **24/24** original pilot frames now convert with zero rejections. The minimum
resampled IoU is **0.943942**, and minimum raw IoU is **1.0**, at native resolution.
This checks the exporter/loader geometry, not trained mask quality or augmentation.

Apply the patch to your Windows toolkit too, or replace its
`tools/perception_v2_data/prepare_dataset.py` with this version before converting.
No PC reinstall or training-parameter change is required. Use a fresh conversion
output directory to avoid mixing labels. A typical PC command is:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py `
  --raw .\data\primitive_pilot_v2 `
  --out .\data\primitive_pilot_v2_yolo
```

## Tests and limits

Verified locally:

- 29 existing toolkit tests.
- 25 new appearance/output/resume tests, including a synthetic capture adapter.
- 11 recorded-USD-call tests for support selection, material/light setup, and
  session-layer guards. These are **not native USD or renderer tests**.
- 6 converter regressions, including small masks and retained contour vertices.
- Total: **71 unittest methods passed**.
- Compilation and Python 3.11 grammar parsing for all changed/new Python files.
- Existing repository architecture/syntax checker after reconstructing the current
  repository + previous overlays + the corrected toolkit.
- Original uploaded pilot: all 24 RGB/mask bundles audited; converter succeeds on
  all 24 without editing the input; files and counts recorded in validation logs.
- Fresh overlay extraction and repeat of the tests / converter check.

Host execution: Python 3.13.5, NumPy 2.3.5, OpenCV 4.13.0. **No Isaac rendering,
physical simulation, native USD binding execution, GPU training or Windows run
was available here.** The generator still enforces your established Python 3.11 /
NumPy 1.26.4 startup and uses the existing wrapper without the rejected ROS flag.
Keep that environment unchanged; do not install test-host packages in Isaac.

Start with the pilot at `--light-scale 1.0`. Only if the real render is still too
bright, repeat in a different directory with `--light-scale 0.7`. Do not start the
large batch until the RGB and mask previews align and look useful. A successful
export does not demonstrate that this training distribution beats V1 perception.

## API references used

- NVIDIA Isaac Sim 5.1 scripted scene/light randomization:
  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_object_based_sdg.html
- USD binding strengths and material purposes:
  https://openusd.org/dev/api/class_usd_shade_material_binding_a_p_i.html
- Documented RTX histogram/auto-exposure setting:
  https://docs.omniverse.nvidia.com/avp/latest/performance.html
- Pinned Ultralytics 8.3.161 `resample_segments`, source blob
  `3ca26a8ff213a24a78a08cd04a3cae3c6bda84e9`:
  https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/utils/ops.py
