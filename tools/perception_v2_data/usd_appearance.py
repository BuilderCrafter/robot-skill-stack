"""Session-only visual overrides for the standalone capture process."""
from __future__ import annotations

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade

from appearance import REVISION, sample_look, sample_surface


def support_candidate(lo, hi, bounds, support_z):
    """Large, thin, horizontal surface spanning the sampling area."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all():
        return False
    span = hi - lo
    return bool(abs(hi[2] - support_z) <= .015 and 0 <= span[2] <= .15
                and lo[0] <= bounds[0] and hi[0] >= bounds[2]
                and lo[1] <= bounds[1] and hi[1] >= bounds[3]
                and min(span[:2]) >= .25)


class CaptureAppearance:
    def __init__(self, stage, args, root, material_factory):
        if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
            raise RuntimeError("SDG visual changes must target the unsaved session layer")
        self.stage, self.args, self.root = stage, args, root
        self.material, self.floor_inputs = material_factory(stage, f"{root}/Materials/support")
        self.grounds = self._ground_prims()
        for prim in self.grounds:
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            for purpose in (UsdShade.Tokens.allPurpose, UsdShade.Tokens.preview, UsdShade.Tokens.full):
                binding.Bind(self.material, UsdShade.Tokens.strongerThanDescendants, purpose)
                resolved, _ = binding.ComputeBoundMaterial(purpose)
                if not resolved or resolved.GetPath() != self.material.GetPath():
                    raise RuntimeError(f"Ground material override did not resolve at {prim.GetPath()}; pass --ground-root at its material-binding ancestor")
        self.disabled_lights = []
        for prim in stage.Traverse():
            attr = prim.GetAttribute("inputs:intensity")
            if prim.GetTypeName().endswith("Light") and attr:
                self.disabled_lights.append(str(prim.GetPath()))
                attr.Set(0.)
        self.fill = UsdLux.DomeLight.Define(stage, f"{root}/CaptureFill")
        self.fill.CreateIntensityAttr(180.)
        self.fill.CreateExposureAttr(0.)
        self.fill.CreateColorAttr(Gf.Vec3f(1., 1., 1.))
        self.key = UsdLux.DistantLight.Define(stage, f"{root}/CaptureKey")
        self.key.CreateIntensityAttr(950.)
        self.key.CreateExposureAttr(0.)
        self.key.CreateAngleAttr(6.)
        self.key.CreateColorAttr(Gf.Vec3f(1., 1., 1.))
        self.key_rotation = UsdGeom.Xformable(self.key.GetPrim()).AddRotateXYZOp()
        self.current = None

    def _ground_prims(self):
        if self.args.blank_scene:
            return [self.stage.GetPrimAtPath(f"{self.root}/Floor")]
        explicit = getattr(self.args, "ground_root", None)
        if explicit:
            prim = self.stage.GetPrimAtPath(explicit)
            if not prim or prim.IsInstanceProxy():
                raise ValueError(f"Invalid/non-editable --ground-root: {explicit}")
            # Do not accidentally bind a ground material over the complete scene/robot.
            if str(prim.GetPath()) in ("/", "/World", "/Franka", "/World/Franka", self.root):
                raise ValueError("--ground-root must identify the support surface, not the complete scene or robot")
            return [prim]
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
        result = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(self.root + "/") or path.startswith(("/Franka/", "/World/Franka/")):
                continue
            if prim.IsInstanceProxy() or not prim.IsA(UsdGeom.Gprim):
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible or imageable.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
                continue
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if not box.IsEmpty() and support_candidate(box.GetMin(), box.GetMax(), self.args.bounds, self.args.support_z):
                result.append(prim)
        if not result:
            raise ValueError("Could not find a visible support surface near --support-z spanning --bounds. Pass --ground-root with its USD path (not /World). Stopping rather than silently leaving the white background unchanged.")
        return result

    def apply(self, objects, rng):
        look = sample_look(rng, self.args.light_scale)
        color, rough, metal = self.floor_inputs
        color.Set(Gf.Vec3f(*look["floor_linear"]))
        rough.Set(.95)
        metal.Set(0.)
        self.fill.GetIntensityAttr().Set(look["fill_intensity"])
        self.key.GetIntensityAttr().Set(look["key_intensity"])
        self.key.GetColorAttr().Set(Gf.Vec3f(*look["light_color"]))
        self.key.GetAngleAttr().Set(look["key_angle_deg"])
        self.key_rotation.Set(Gf.Vec3f(*look["key_rotation_deg"]))
        for obj in objects:
            obj.update(sample_surface(rng, look))
        self.current = look
        return look

    def setup_info(self):
        return {"revision": REVISION, "ground_prims": [str(p.GetPath()) for p in self.grounds],
                "disabled_source_lights": self.disabled_lights,
                "robot_materials": "unchanged", "camera": "unchanged",
                "auto_exposure": False, "light_scale": self.args.light_scale,
                "mode_probabilities": {"clear": .7, "varied": .2, "hard": .1},
                "color_space": "sample sRGB, convert to linear for USD materials"}
