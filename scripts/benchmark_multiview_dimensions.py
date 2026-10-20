"""Research-only matched native curved meshes; never commands robot motion."""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4 before SimulationApp, got {np.__version__}')
from isaacsim import SimulationApp
app = SimulationApp({'headless': False})

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import omni.usd
from pxr import Gf, UsdGeom

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.world.multiview import create_cameras
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, fuse_clouds, project_depth, save_ply
from tests.support.multiview_native import CAMERAS, ROOT, open_experiment, populate, set_viewpoint
from tests.support.multiview_research import ResearchObject, oracle_mask


from tests.support.multiview_dimension_research import mesh_surface


def author_mesh(obj, segments):
    stage = omni.usd.get_context().get_stage()
    path = '/World/MultiViewObjects/'+obj.name
    mesh = UsdGeom.Mesh.Define(stage, path)
    vertices, faces = mesh_surface(obj, segments)
    mesh.CreatePointsAttr([Gf.Vec3f(*point.tolist()) for point in vertices])
    mesh.CreateExtentAttr([Gf.Vec3f(*point.tolist()) for point in (vertices.min(0), vertices.max(0))])
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces).ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(.4, .6, .8)])


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(numpy_version=np.__version__, source_sha256={
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'scripts/benchmark_multiview_dimensions.py',
            'tests/support/multiview_dimension_research.py',
            'tests/support/multiview_native.py', 'tests/support/multiview_research.py',
            'robot_skill_stack/world/perception_multiview/clouds.py')})
    (output/'provenance_start.json').write_text(json.dumps(provenance, indent=2))
    world = open_experiment(app)
    scene = load_scene_config(ROOT/'config/scenes/multiview_playground.toml').scene
    rig = create_cameras(CAMERAS)
    manifest = []
    specs = [('sphere40', 'sphere', .020, 0),
             ('sphere71', 'sphere', .0355, 0), ('sphere71', 'sphere', .0355, 32),
             ('sphere71', 'sphere', .0355, 128),
             ('sphere76', 'sphere', .038, 0), ('sphere76', 'sphere', .038, 128),
             ('upright71', 'cylinder', .0355, 0), ('upright71', 'cylinder', .0355, 128),
             ('lying71', 'cylinder', .0355, 0), ('lying71', 'cylinder', .0355, 128)]
    try:
        for _ in range(60):
            world.step(render=True)
        for arrangement in range(2):
            if arrangement:
                set_viewpoint(rig.cameras[CAMERAS[1]], (.75, -.70, .55))
            for index, (label, shape, radius, segments) in enumerate(specs):
                name = f'view{arrangement}_{index}_{label}_{"usd" if not segments else f"mesh{segments}"}'
                axis = (np.cos(.43), np.sin(.43), 0.) if label.startswith('lying') else (0., 0., 1.)
                z = radius if shape == 'sphere' or label.startswith('lying') else .045
                center = np.array([.43 if not arrangement else .49, -.08 if not arrangement else .09, z])
                extent = (np.full(3, 2*radius) if shape == 'sphere' else
                          abs(np.asarray(axis))*.09+2*radius*np.sqrt(1-np.asarray(axis)**2))
                obj = ResearchObject(name, shape, center, extent, radius=radius, length=.09, axis=axis)
                populate([] if segments else [obj])
                if segments:
                    author_mesh(obj, segments)
                for _ in range(24):
                    world.step(render=True)
                frames = rig.capture()
                if frames is None:
                    raise RuntimeError('Controlled capture did not produce fresh frames')
                destination = output/name
                destination.mkdir(parents=True, exist_ok=True)
                clouds = []
                for camera_index, frame in enumerate(frames):
                    np.savez_compressed(destination/f'camera_{camera_index}.npz', depth=frame.depth,
                        intrinsics=frame.intrinsics, world_from_camera=frame.world_from_camera,
                        acquired_at=frame.acquired_at, token=np.asarray(frame.token))
                    points = crop_workspace(project_depth(frame, stride=1)[0], scene)
                    cloud = points[oracle_mask(points, obj)]
                    clouds.append(cloud)
                    save_ply(destination/f'camera_{camera_index}.ply', cloud)
                fused = fuse_clouds(clouds)
                save_ply(destination/'fused.ply', fused)
                # Empty or invisible fixtures fail visibly rather than silently
                # being omitted from a dimension comparison.
                record = dict(scenario=name, object=asdict(obj), segments=segments,
                              view_points=[len(cloud) for cloud in clouds], fused_points=len(fused),
                              directory=name, fixture_valid=len(fused) >= 30)
                manifest.append(record)
                (output/'manifest.json').write_text(json.dumps(manifest, indent=2, default=json_default))
                print(name, record['view_points'], record['fixture_valid'], flush=True)
        if not all(row['fixture_valid'] for row in manifest):
            raise AssertionError('Controlled geometry corpus contains invisible fixtures')
        (output/'result.json').write_text(json.dumps(dict(status='PASS', scenes=len(manifest),
            purpose='research-only curved surface capture; no motion or runtime oracle', robot_commands=0), indent=2))
        (output/'provenance_end.json').write_text(json.dumps(dict(numpy_version=np.__version__, source_sha256={
            name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in provenance['source_sha256']}), indent=2))
    finally:
        rig.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        main(args)
    finally:
        app.close()
