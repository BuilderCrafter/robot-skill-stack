"""Isaac-only scene adapter. Import AFTER SimulationApp, never from the live GUI."""
from __future__ import annotations

import time
import numpy as np
import carb.settings
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import is_stage_loading
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from common import CLASSES, camera_matrix, random_color, rotation

ROOT = "/PrimitiveTrainingSDG"


def gf_matrix(T):
    return Gf.Matrix4d(*map(float, np.asarray(T).T.ravel()))


def pose_matrix(position, dims, R):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R) @ np.diag(dims)
    T[:3, 3] = position
    return gf_matrix(T)


def material(stage, path):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    color = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    rough = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    metal = shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
    color.Set(Gf.Vec3f(.5, .5, .5)); rough.Set(.65); metal.Set(0.)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat, (color, rough, metal)


class CaptureScene:
    def __init__(self, app, args):
        self.app, self.args = app, args
        context = omni.usd.get_context()
        if args.blank_scene:
            context.new_stage()
        elif not context.open_stage(str(args.scene)):
            raise RuntimeError(f"Could not open scene {args.scene}")
        deadline = time.monotonic() + 300.
        for _ in range(15):
            app.update()
        while is_stage_loading():
            if time.monotonic() > deadline or not app.is_running():
                raise RuntimeError("Scene assets did not finish loading. Open the saved scene normally to check asset availability.")
            app.update()
        self.stage = context.get_stage()
        if self.stage is None:
            raise RuntimeError("No USD stage")
        # All edits are ephemeral. The original stage and its references are never saved.
        self.stage.SetEditTarget(Usd.EditTarget(self.stage.GetSessionLayer()))
        omni.timeline.get_timeline_interface().stop()
        if args.blank_scene:
            UsdGeom.SetStageMetersPerUnit(self.stage, 1.)
            UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        if abs(UsdGeom.GetStageMetersPerUnit(self.stage) - 1.) > 1e-8:
            raise ValueError("This generator expects a meter-unit stage; it will not rescale your scene")
        if UsdGeom.GetStageUpAxis(self.stage) != UsdGeom.Tokens.z:
            raise ValueError("This generator expects a Z-up stage")
        if self.stage.GetPrimAtPath(ROOT):
            raise ValueError(f"Reserved generation root already exists: {ROOT}")
        old = self.stage.GetPrimAtPath(args.objects_root)
        if old:
            old.SetActive(False)
        UsdGeom.Xform.Define(self.stage, ROOT)
        self._camera()
        self.pool, self.materials = {}, {}
        for slot in range(args.max_objects):
            mat, inputs = material(self.stage, f"{ROOT}/Materials/target_{slot}")
            self.materials[slot] = inputs
            for shape in CLASSES:
                path = f"{ROOT}/Targets/item_{slot}_{shape}"
                schema = {"cube": UsdGeom.Cube, "sphere": UsdGeom.Sphere, "cylinder": UsdGeom.Cylinder}[shape].Define(self.stage, path)
                if shape == "cube":
                    schema.CreateSizeAttr(1.)
                else:
                    schema.CreateRadiusAttr(.5)
                    if shape == "cylinder":
                        schema.CreateHeightAttr(1.)
                        schema.CreateAxisAttr(UsdGeom.Tokens.z)
                prim = schema.GetPrim()
                add_labels(prim, labels=[shape], instance_name="class")
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
                self.pool[slot, shape] = (prim, UsdGeom.Xformable(prim).AddTransformOp(), schema.CreateVisibilityAttr())
        self.clutter = []
        for index in range(3):
            prim = UsdGeom.Cone.Define(self.stage, f"{ROOT}/Clutter/cone_{index}")
            prim.CreateRadiusAttr(.5); prim.CreateHeightAttr(1.); prim.CreateAxisAttr(UsdGeom.Tokens.z)
            mat, inputs = material(self.stage, f"{ROOT}/Materials/clutter_{index}")
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)
            self.clutter.append((UsdGeom.Xformable(prim).AddTransformOp(), prim.CreateVisibilityAttr(), inputs))
        self.lights = []
        for prim in self.stage.Traverse():
            attr = prim.GetAttribute("inputs:intensity")
            if attr and prim.GetTypeName().endswith("Light"):
                value = attr.Get()
                if value is not None and value > 0:
                    self.lights.append((attr, float(value)))
        if not self.lights:
            light = UsdLux.DomeLight.Define(self.stage, f"{ROOT}/Light")
            light.CreateIntensityAttr(600.)
            self.lights = [(light.GetIntensityAttr(), 600.)]
        if args.blank_scene:
            floor = UsdGeom.Cube.Define(self.stage, f"{ROOT}/Floor")
            floor.CreateSizeAttr(1.)
            UsdGeom.Xformable(floor).AddTransformOp().Set(pose_matrix([.45, 0., args.support_z-.025], [2., 2., .05], np.eye(3)))
            self.floor_mat, self.floor_inputs = material(self.stage, f"{ROOT}/Materials/floor")
            UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(self.floor_mat)
        rep.orchestrator.set_capture_on_play(False)
        carb.settings.get_settings().set("/rtx/post/dlss/execMode", 2)
        self.product = rep.create.render_product(str(self.camera.GetPath()), (args.width, args.height))
        self.annotators = {}
        for key, name in (("rgb", "rgb"), ("instances", "instance_id_segmentation")):
            ann = rep.AnnotatorRegistry.get_annotator(name, init_params={"colorize": False} if key == "instances" else {})
            ann.attach(self.product)
            self.annotators[key] = ann
        if args.save_depth:
            ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            ann.attach(self.product)
            self.annotators["depth"] = ann
        for _ in range(12):
            app.update()

    def _camera(self):
        args = self.args
        self.camera = UsdGeom.Camera.Define(self.stage, f"{ROOT}/Camera")
        src = self.stage.GetPrimAtPath(args.camera)
        if not args.blank_scene and (not src or not src.IsA(UsdGeom.Camera)):
            raise ValueError(f"Expected a Camera at {args.camera}; use --camera or --blank-scene explicitly")
        if src and not args.blank_scene:
            source = UsdGeom.Camera(src)
            if source.GetProjectionAttr().Get() != UsdGeom.Tokens.perspective:
                raise ValueError("Only an undistorted perspective camera is supported")
            for attr_name in ("focalLength", "horizontalAperture", "horizontalApertureOffset", "verticalApertureOffset", "clippingRange"):
                value = src.GetAttribute(attr_name).Get()
                if value is not None:
                    self.camera.GetPrim().GetAttribute(attr_name).Set(value)
            offsets = [source.GetHorizontalApertureOffsetAttr().Get(), source.GetVerticalApertureOffsetAttr().Get()]
            if any(abs(v or 0.) > 1e-8 for v in offsets):
                raise ValueError("Nonzero aperture offsets are not supported by this initial exporter")
            self.base_camera = np.asarray(UsdGeom.Xformable(src).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T.copy()
        else:
            self.camera.CreateFocalLengthAttr(24.)
            self.camera.CreateHorizontalApertureAttr(32.)
            self.base_camera = camera_matrix([.92, -.66, .75], [.45, 0., .04])
        self.camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
        self.camera.CreateFocusDistanceAttr(0.)
        self.camera.CreateFStopAttr(0.)
        ha = float(self.camera.GetHorizontalApertureAttr().Get())
        self.camera.CreateVerticalApertureAttr(ha * args.height / args.width)
        f = float(self.camera.GetFocalLengthAttr().Get())
        self.K = np.array([[args.width*f/ha, 0., args.width/2], [0., args.width*f/ha, args.height/2], [0., 0., 1.]])
        self.camera_op = UsdGeom.Xformable(self.camera).AddTransformOp()
        self.camera_op.Set(gf_matrix(self.base_camera))

    def choose_camera(self, rng):
        T = self.base_camera.copy()
        # Keep the actual camera pose in 75% of samples; small translations otherwise.
        # Orientation stays calibrated, rather than accidentally pointing at a different workspace.
        if rng.random() < .25:
            T[:3, 3] += rng.uniform([-.015, -.015, -.015], [.015, .015, .025])
        self.camera_op.Set(gf_matrix(T))
        return T

    def apply(self, objects, rng, camera):
        for _, _, vis in self.pool.values():
            vis.Set(UsdGeom.Tokens.invisible)
        for obj in objects:
            prim, op, vis = self.pool[obj["slot"], obj["shape"]]
            op.Set(pose_matrix(obj["position"], obj["dimensions_local"], obj["rotation_world"]))
            vis.Set(UsdGeom.Tokens.inherited)
            color, rough, metal = self.materials[obj["slot"]]
            color.Set(Gf.Vec3f(*obj["color"])); rough.Set(obj["roughness"]); metal.Set(obj["metallic"])
            obj["prim_path"] = str(prim.GetPath())
        nclutter = int(rng.integers(1, 4)) if rng.random() < .40 else 0
        for i, (op, vis, inputs) in enumerate(self.clutter):
            vis.Set(UsdGeom.Tokens.inherited if i < nclutter else UsdGeom.Tokens.invisible)
            if i >= nclutter:
                continue
            d, h = float(rng.uniform(.025, .07)), float(rng.uniform(.035, .13))
            xy = rng.uniform(self.args.bounds[:2], self.args.bounds[2:])
            if i == 0 and objects and rng.random() < .7:
                target = objects[0]
                direction = camera[:2, 3] - target["position"][:2]
                direction /= max(np.linalg.norm(direction), 1e-6)
                xy = np.asarray(target["position"][:2]) + direction * (target["footprint"] + d/2 + .01)
            # Reposition intersecting distractors rather than generating impossible interpenetrations.
            for _ in range(100):
                if all(np.linalg.norm(xy-np.asarray(o["position"][:2])) >= d/2+o["footprint"]+.004 for o in objects):
                    break
                xy = rng.uniform(self.args.bounds[:2], self.args.bounds[2:])
            else:
                vis.Set(UsdGeom.Tokens.invisible)
                continue
            op.Set(pose_matrix([*xy, self.args.support_z+h/2], [d, d, h], rotation(float(rng.uniform(-np.pi, np.pi)))))
            inputs[0].Set(Gf.Vec3f(*random_color(rng)))
        for attr, base in self.lights:
            attr.Set(float(base * rng.uniform(.65, 1.4)))
        if self.args.blank_scene:
            self.floor_inputs[0].Set(Gf.Vec3f(*random_color(rng)))

    def capture(self):
        # This standalone process is not an OmniUI callback and never runs project skills.
        rep.orchestrator.step(rt_subframes=self.args.subframes, delta_time=0., pause_timeline=True)
        return {name: ann.get_data() for name, ann in self.annotators.items()}

    def close(self):
        for ann in self.annotators.values():
            ann.detach()
        self.product.destroy()
        rep.orchestrator.wait_until_complete()
