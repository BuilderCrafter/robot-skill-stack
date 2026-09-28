"""Recorded USD calls, NOT native USD material resolution or rendering tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Attr:
    def __init__(self, value=None): self.value = value
    def Set(self, value): self.value = value
    def Get(self): return self.value


class Prim:
    def __init__(self, path, kind='Mesh', lo=(-2, -2, -.02), hi=(2, 2, 0), visible='inherited', purpose='default', proxy=False):
        self.path, self.kind, self.lo, self.hi = path, kind, lo, hi
        self.visible, self.purpose, self.proxy = visible, purpose, proxy
        self.attrs, self.bindings = {}, {}
    def GetPath(self): return self.path
    def GetTypeName(self): return self.kind
    def GetAttribute(self, name): return self.attrs.get(name)
    def IsInstanceProxy(self): return self.proxy
    def IsA(self, cls): return cls == 'Gprim' and self.kind in ('Mesh', 'Cube')


class Material:
    def __init__(self, path): self.path = path
    def GetPath(self): return self.path


class Binding:
    force_failure = False
    def __init__(self, prim): self.prim = prim
    @classmethod
    def Apply(cls, prim): return cls(prim)
    def Bind(self, mat, strength, purpose):
        assert strength == 'strongerThanDescendants'
        self.prim.bindings[purpose] = (mat, strength)
    def ComputeBoundMaterial(self, purpose):
        return (None if self.force_failure else self.prim.bindings[purpose][0]), None


class Light:
    @classmethod
    def Define(cls, stage, path):
        light = cls(); light.prim = Prim(path, 'CaptureLight'); stage.prims[path] = light.prim
        return light
    def GetPrim(self): return self.prim
    def __getattr__(self, name):
        if name.startswith('Create') and name.endswith('Attr'):
            def create(value):
                attr = self.prim.attrs.setdefault(name[6:-4], Attr()); attr.Set(value); return attr
            return create
        if name.startswith('Get') and name.endswith('Attr'):
            return lambda: self.prim.attrs[name[3:-4]]
        raise AttributeError(name)


class Imageable:
    def __init__(self, prim): self.prim = prim
    def ComputeVisibility(self): return self.prim.visible
    def ComputePurpose(self): return self.prim.purpose


class BBoxCache:
    def __init__(self, *args, **kwargs): pass
    def ComputeWorldBound(self, prim):
        return NS(ComputeAlignedRange=lambda: NS(IsEmpty=lambda: False, GetMin=lambda: prim.lo, GetMax=lambda: prim.hi))


class Stage:
    def __init__(self, prims):
        self.session = object(); self.edit_layer = self.session
        self.prims = {p.path: p for p in prims}
    def GetEditTarget(self): return NS(GetLayer=lambda: self.edit_layer)
    def GetSessionLayer(self): return self.session
    def GetPrimAtPath(self, path): return self.prims.get(path)
    def Traverse(self): return list(self.prims.values())


def factory(stage, path): return Material(path), (Attr(), Attr(), Attr())


def load_module():
    pxr = NS(Gf=NS(Vec3f=lambda *v: tuple(v)), Usd=NS(TimeCode=NS(Default=lambda: 0)),
             UsdGeom=NS(Gprim='Gprim', Tokens=NS(default_='default', render='render', guide='guide', proxy='proxy', invisible='invisible'),
                        Imageable=Imageable, BBoxCache=BBoxCache, Xformable=lambda prim: NS(AddRotateXYZOp=lambda: Attr())),
             UsdShade=NS(MaterialBindingAPI=Binding, Tokens=NS(allPurpose='', preview='preview', full='full', strongerThanDescendants='strongerThanDescendants')),
             UsdLux=NS(DomeLight=Light, DistantLight=Light))
    spec = importlib.util.spec_from_file_location('usd_appearance_recording_test', ROOT/'usd_appearance.py')
    mod = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'pxr': pxr}): spec.loader.exec_module(mod)
    return mod


M = load_module()


def args(**overrides):
    return NS(**dict({'blank_scene': False, 'ground_root': None, 'bounds': [.3, -.18, .62, .18],
                      'support_z': 0., 'light_scale': 1.}, **overrides))


class UsdAppearanceTests(unittest.TestCase):
    def test_only_session_layer_allowed(self):
        stage = Stage([Prim('/Ground')]); stage.edit_layer = object()
        with self.assertRaisesRegex(RuntimeError, 'session'):
            M.CaptureAppearance(stage, args(), '/SDG', factory)

    def test_auto_ground_excludes_robot_targets_and_hidden(self):
        ground, robot, target = Prim('/Ground'), Prim('/Franka/mesh'), Prim('/SDG/Targets/cube')
        hidden, guide = Prim('/Hidden', visible='invisible'), Prim('/Guide', purpose='guide')
        app = M.CaptureAppearance(Stage([ground, robot, target, hidden, guide]), args(), '/SDG', factory)
        self.assertEqual(app.grounds, [ground])
        for p in [robot, target, hidden, guide]: self.assertEqual(p.bindings, {})

    def test_missing_support_is_explicit_failure(self):
        with self.assertRaisesRegex(ValueError, '--ground-root'):
            M.CaptureAppearance(Stage([]), args(), '/SDG', factory)

    def test_support_geometry_rule(self):
        rule = M.support_candidate; b = [.3, -.18, .62, .18]
        self.assertTrue(rule([-1, -1, -.02], [1, 1, 0], b, 0.))
        for lo, hi in [([.4, 0, 0], [.5, .1, .1]), ([-1, -1, -.4], [1, 1, 0]), ([-1, -1, 1], [1, 1, 1.1]), ([float('nan'), 0, 0], [1, 1, 0])]:
            self.assertFalse(rule(lo, hi, b, 0.))

    def test_blank_scene_floor(self):
        floor = Prim('/SDG/Floor')
        app = M.CaptureAppearance(Stage([floor]), args(blank_scene=True), '/SDG', factory)
        self.assertEqual(app.grounds, [floor])

    def test_explicit_ground_path(self):
        ground = Prim('/Table', 'Xform')
        app = M.CaptureAppearance(Stage([ground]), args(ground_root='/Table'), '/SDG', factory)
        self.assertEqual(app.grounds, [ground])

    def test_invalid_or_overbroad_path_rejected(self):
        for path in ['/World', '/Franka', '/World/Franka', '/SDG', '/']:
            with self.assertRaises(ValueError):
                M.CaptureAppearance(Stage([Prim(path)]), args(ground_root=path), '/SDG', factory)
        with self.assertRaises(ValueError):
            M.CaptureAppearance(Stage([]), args(ground_root='/Missing'), '/SDG', factory)

    def test_full_preview_and_all_purpose_bindings(self):
        floor = Prim('/Ground')
        app = M.CaptureAppearance(Stage([floor]), args(), '/SDG', factory)
        self.assertEqual(set(floor.bindings), {'', 'preview', 'full'})
        self.assertTrue(all(mat is app.material and strength == 'strongerThanDescendants' for mat, strength in floor.bindings.values()))

    def test_material_resolution_failure_stops(self):
        with patch.object(Binding, 'force_failure', True), self.assertRaisesRegex(RuntimeError, 'did not resolve'):
            M.CaptureAppearance(Stage([Prim('/Ground')]), args(), '/SDG', factory)

    def test_old_lights_disabled_not_other_prims(self):
        old, other = Prim('/OldSun', 'DistantLight'), Prim('/Other')
        old.attrs['inputs:intensity'] = Attr(10000.); other.attrs['inputs:intensity'] = Attr(42.)
        app = M.CaptureAppearance(Stage([Prim('/Ground'), old, other]), args(), '/SDG', factory)
        self.assertEqual(old.attrs['inputs:intensity'].Get(), 0.)
        self.assertEqual(other.attrs['inputs:intensity'].Get(), 42.)
        self.assertEqual(app.disabled_lights, ['/OldSun'])

    def test_apply_records_linear_colors_matte_and_no_pose_changes(self):
        app = M.CaptureAppearance(Stage([Prim('/Ground')]), args(), '/SDG', factory)
        objects = [{'shape': s, 'position': [1, 2, 3], 'slot': i} for i, s in enumerate(CLASSES)]
        look = app.apply(objects, np.random.default_rng(77))
        self.assertEqual(app.floor_inputs[0].Get(), tuple(look['floor_linear']))
        self.assertEqual(app.floor_inputs[1].Get(), .95)
        for o in objects:
            self.assertEqual(o['position'], [1, 2, 3])
            self.assertEqual(o['metallic'], 0.)
        json.dumps(app.setup_info(), allow_nan=False)


from common import CLASSES
if __name__ == '__main__':
    unittest.main(verbosity=2)
