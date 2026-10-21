"""Native research fixtures; import only after SimulationApp starts."""
from pathlib import Path

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from pxr import Gf, UsdGeom

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ('/World/PerceptionCamera', '/World/PerceptionCameraSide')


def open_experiment(app):
    if not open_stage(str(ROOT/'scenes/multiview_playground.usda')):
        raise RuntimeError('Could not open multi-view scene')
    for _ in range(600):
        if not is_stage_loading():
            break
        app.update()
    else:
        raise RuntimeError('Multi-view stage loading timed out')
    world = World(stage_units_in_meters=1.)
    world.reset()
    world.play()
    return world


def set_viewpoint(camera, eye, target=(.45, 0., .04)):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # USD uses row-vector matrices and a -Z optical axis.
    r = np.column_stack((right, up, -forward))
    quat = Gf.Matrix3d(*r.T.ravel().tolist()).ExtractRotation().GetQuat()
    orientation = np.r_[quat.GetReal(), np.asarray(quat.GetImaginary())]
    camera.camera.set_world_pose(position=eye, orientation=orientation, camera_axes='usd')


def populate(objects):
    stage = omni.usd.get_context().get_stage()
    stage.RemovePrim('/World/MultiViewObjects')
    UsdGeom.Xform.Define(stage, '/World/MultiViewObjects')
    for index, obj in enumerate(objects):
        path = '/World/MultiViewObjects/'+obj.name
        if obj.shape == 'cube':
            shape = UsdGeom.Cube.Define(stage, path)
            shape.CreateSizeAttr(1.)
        elif obj.shape == 'sphere':
            shape = UsdGeom.Sphere.Define(stage, path)
            shape.CreateRadiusAttr(obj.radius)
        elif obj.shape == 'cylinder':
            shape = UsdGeom.Cylinder.Define(stage, path)
            shape.CreateRadiusAttr(obj.radius)
            shape.CreateHeightAttr(obj.length)
            shape.CreateAxisAttr('Z' if abs(obj.axis[2]) > .99 else 'X')
        else:
            raise ValueError(obj.shape)
        xform = UsdGeom.Xformable(shape.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*obj.center.tolist()))
        yaw = obj.yaw if obj.shape == 'cube' else np.arctan2(obj.axis[1], obj.axis[0])
        if obj.shape != 'sphere':
            xform.AddRotateZOp().Set(float(np.rad2deg(yaw)))
        if obj.shape == 'cube':
            xform.AddScaleOp().Set(Gf.Vec3f(*obj.size.tolist()))
        UsdGeom.Gprim(shape.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(.25+.1*(index % 5), .5, .8)])


def research_objects():
    from tests.support.multiview_research import ResearchObject
    objects = []
    for index, (size, yaw) in enumerate((((.05, .05, .05), 0.), ((.075, .035, .045), .61))):
        objects.append(ResearchObject(f'box_{index}', 'cube', np.array([.45, 0, size[2]/2]), np.array(size), yaw))
    for index, radius in enumerate((.020, .032)):
        objects.append(ResearchObject(f'sphere_{index}', 'sphere', np.array([.45, 0, radius]),
                                       np.full(3, radius*2), radius=radius))
    for index, (radius, length) in enumerate(((.018, .09), (.027, .055))):
        objects.append(ResearchObject(f'upright_{index}', 'cylinder', np.array([.45, 0, length/2]),
                                       np.array([2*radius, 2*radius, length]), radius=radius, length=length))
    for index, yaw in enumerate((.0, .61)):
        radius, length = .018+index*.004, .10-index*.015
        axis = (np.cos(yaw), np.sin(yaw), 0.)
        objects.append(ResearchObject(f'lying_{index}', 'cylinder', np.array([.45, 0, radius]),
                                       np.array([length, 2*radius, 2*radius]), radius=radius, length=length, axis=axis))
    return objects
