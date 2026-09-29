"""Read a fixed allow-list of robot visual meshes; never label/query scene targets."""
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from robot_skill_stack.world.perception.v2.mesh_rays import triangulate
from robot_skill_stack.world.perception.v2.mesh_shapes import box_triangles, round_triangles
from robot_skill_stack.world.perception.v2.self_filter import RobotSurfaceModel

FRANKA_LINKS = tuple(f'panda_link{i}' for i in range(8)) + ('panda_hand', 'panda_leftfinger', 'panda_rightfinger')


def _geometry(prim):
    kind = prim.GetTypeName()
    if kind == 'Mesh':
        mesh = UsdGeom.Mesh(prim)
        attrs = (mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(), mesh.GetFaceVertexIndicesAttr())
        if any(a.ValueMightBeTimeVarying() for a in attrs):
            raise ValueError(f'Deforming robot mesh is unsupported: {prim.GetPath()}')
        return triangulate(*(a.Get() for a in attrs), holes=mesh.GetHoleIndicesAttr().Get() or ())
    if kind == 'Cube':
        return box_triangles(float(UsdGeom.Cube(prim).GetSizeAttr().Get()))
    if kind in {'Sphere', 'Cylinder', 'Capsule', 'Cone'}:
        shape = getattr(UsdGeom, kind)(prim)
        radius = float(shape.GetRadiusAttr().Get())
        height = 0. if kind == 'Sphere' else float(shape.GetHeightAttr().Get())
        axis = 'Z' if kind == 'Sphere' else str(shape.GetAxisAttr().Get())
        return round_triangles(kind, radius, height, axis)
    raise ValueError(f'Unsupported visible robot geometry {kind}: {prim.GetPath()}')


def load_robot_meshes(stage, robot_root, excluded_roots=(), link_names=FRANKA_LINKS):
    if (UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or
            not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.)):
        raise ValueError('Self-filter expects the existing meters/Z-up scene')
    root = stage.GetPrimAtPath(robot_root)
    if not root.IsValid() or str(root.GetPath()) == '/':
        raise ValueError('Self-filter needs the configured robot root, not the complete stage')
    def excluded(path):
        return any(path == p.rstrip('/') or path.startswith(p.rstrip('/')+'/') for p in excluded_roots)
    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    links = {}
    for name in link_names:
        matches = [p for p in prims if p.GetName() == name and p.IsA(UsdGeom.Xformable) and not excluded(str(p.GetPath()))]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one robot link {name} below {robot_root}; found {len(matches)}')
        links[name] = matches[0]
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meshes, inventory = [], []
    for name, link in links.items():
        pieces, paths = [], []
        for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if excluded(path) or not prim.IsA(UsdGeom.Gprim):
                continue
            # Never absorb a nested body/another link (e.g. a reparented carried object).
            parent, allowed = prim, True
            while parent and parent != link:
                if parent.HasAPI(UsdPhysics.RigidBodyAPI) or parent.GetName() in links:
                    allowed = False; break
                if UsdGeom.Xformable(parent).TransformMightBeTimeVarying():
                    raise ValueError(f'Animated local robot geometry unsupported: {parent.GetPath()}')
                parent = parent.GetParent()
            imageable = UsdGeom.Imageable(prim)
            if (not allowed or imageable.ComputeVisibility() == UsdGeom.Tokens.invisible or
                    imageable.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)):
                continue
            relative, resets = cache.ComputeRelativeTransform(prim, link)
            if resets:
                raise ValueError(f'Reset transform inside robot link: {path}')
            matrix = np.asarray(relative, float).T
            t = _geometry(prim)
            pieces.append(t @ matrix[:3, :3].T+matrix[:3, 3])
            paths.append(path)
        if not pieces:
            raise ValueError(f'No visible triangle geometry for {link.GetPath()}; self-filter not silently disabled')
        meshes.append((str(link.GetPath()), np.concatenate(pieces)))
        inventory.append(dict(link=str(link.GetPath()), geometry_paths=paths, triangles=sum(len(p) for p in pieces)))
    return RobotSurfaceModel(meshes), inventory
