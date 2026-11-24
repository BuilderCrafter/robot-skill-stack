"""SDK-free A/B replay of unchanged native depth; invoke from each detached checkout.

Historical exposure times remain metadata. No live freshness or WorldModel
publication is exercised. Prepare once, then run versions sequentially.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import ctypes
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from unittest.mock import patch

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.common import box_fitting, components, depth_edges
from robot_skill_stack.world.common import camera as acquisition, point_clouds as clouds
from robot_skill_stack.world.perception.multiview import fitting, provider, segmentation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def owned(array):
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def native_frames(case):
    frames = []
    for row in case['frames']:
        assert digest(row['path'])==row['sha256'], 'Original native capture changed'
        with np.load(row['path'], allow_pickle=False) as data:
            assert data['depth'].shape==(480,640)
            frames.append(acquisition.DepthFrame(row['camera_id'], tuple(data['token']), owned(data['depth']),
                owned(data['intrinsics']), owned(data['world_from_camera']), float(data['acquired_at'])))
    return tuple(frames)


def prepare(captures, output):
    assert not output.exists(), 'Do not overwrite frozen input provenance'
    cameras = json.loads((captures/'result.json').read_text())['metrics']['cameras']
    manifest = dict(capture_result_sha256=digest(captures/'result.json'), cases={})
    for label in ('static','pregrasp'):
        rows = []
        for index in range(2):
            path = (captures/f'{label}-{index}.npz').resolve()
            with np.load(path, allow_pickle=False) as data:
                # Recover identities from recorded calibration, not filename assumptions.
                ids = [key for key, camera in cameras.items()
                       if np.array_equal(data['intrinsics'], camera['intrinsics']) and
                       np.array_equal(data['world_from_camera'], camera['transform'])]
                assert len(ids)==1, 'Ambiguous native camera provenance'
                rows.append(dict(path=str(path), sha256=digest(path), camera_id=ids[0],
                    acquired_at=float(data['acquired_at']), token=data['token'].tolist(),
                    depth_sha256=array_digest(data['depth'])))
        manifest['cases'][label] = dict(frames=rows)
    config = load_scene_config('config/scenes/multiview_perception_final.toml')
    scene, settings = config.scene, config.multi_view
    fs = native_frames(manifest['cases']['static'])
    per_view = [clouds.voxelize(clouds.crop_workspace(clouds.project_depth(f, stride=settings.stride,
        bounds=(scene.workspace_min,scene.workspace_max))[0],scene),settings.voxel_size,12000) for f in fs]
    fused = clouds.voxelize(np.concatenate(per_view),settings.voxel_size,12000)
    clusters = components.cluster_points(fused)
    cloud_path = output.with_name(output.stem+'-object-clouds.npz')
    assert not cloud_path.exists()
    np.savez_compressed(cloud_path, **{f'cloud_{i}': cloud for i,cloud in enumerate(clusters)})
    manifest.update(cloud_path=str(cloud_path.resolve()), cloud_sha256=digest(cloud_path),
        clouds={f'cloud_{i}':dict(points=len(c), center=c.mean(0).tolist(),sha256=array_digest(c))
                for i,c in enumerate(clusters)},
        config_sha256=digest('config/scenes/multiview_perception_final.toml'),
        settings=dict(stride=settings.stride,voxel_size=settings.voxel_size,max_points=12000,
                      fit_budget=24,workspace_min=scene.workspace_min.tolist(),workspace_max=scene.workspace_max.tolist(),
                      support_plane_z=scene.support_plane_z))
    output.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['clouds']))


def environment():
    cgroups = {}
    relative = Path('/proc/self/cgroup').read_text().strip().split('::')[-1].lstrip('/')
    directory = Path('/sys/fs/cgroup')/relative
    for parent in (directory,*directory.parents):
        if not parent.is_relative_to('/sys/fs/cgroup'):
            continue
        cgroups[str(parent)] = {name:(parent/name).read_text() for name in
            ('cpu.max','cpu.stat','cpuset.cpus.effective','memory.max') if (parent/name).exists()}
    blas = []
    for path in (Path(np.__file__).resolve().parent.parent/'numpy.libs').glob('*openblas*'):
        library = ctypes.CDLL(str(path))
        get_threads = library.openblas_get_num_threads64_
        get_config = library.openblas_get_config64_
        get_config.restype = ctypes.c_char_p
        blas.append(dict(path=str(path),threads=get_threads(),config=get_config().decode()))
    return dict(python=platform.python_version(),executable=os.sys.executable,numpy=np.__version__,
        numpy_path=np.__file__,affinity=sorted(os.sched_getaffinity(0)),blas=blas,
        thread_environment={key:os.environ.get(key) for key in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        cgroups=cgroups,loadavg=list(os.getloadavg()),
        cpuinfo=Path('/proc/cpuinfo').read_text(),proc_stat=Path('/proc/stat').read_text())


def geometry(fit, *, finalized):
    g = fit.geometry
    dimensions = g.box_size if g.shape.value=='cube' else np.full(3,2*g.radius) if g.shape.value=='sphere' else fit.size
    return dict(shape=g.shape.value,center=fit.center.tolist(),dimensions=dimensions.tolist(),
        yaw=g.yaw,axis=None if g.axis is None else g.axis.tolist(),radius=g.radius,length=g.length,
        confidence=float(g.confidence),graspable=(g.shape.value!='unknown' and bool(g.metadata.get('parameter_observable',False)))
        if finalized else None,reason=g.metadata.get('reason'))


def profiled_compute(processor, prepared):
    seconds, fit_clouds, coarse = defaultdict(float), [], []
    def measured(function, name):
        def call(*args, **kwargs):
            started = time.perf_counter()
            try:
                result = function(*args,**kwargs)
            finally:
                seconds[name] += time.perf_counter()-started
            if name=='fit':
                fit_clouds.append(args[0])
            elif name=='coarse':
                coarse.extend(result)
            return result
        return call
    targets = [(provider,name,'projection_voxel') for name in ('project_depth','crop_workspace','voxelize')]
    targets += [(provider,'cluster_points','coarse'),(provider,'refine_cluster','refinement'),
                (segmentation,'radius_components','metric_graph'),(processor.fitter,'fit','fit'),
                (processor.fitter,'finalize','parameters'),(provider,'assess','observability')]
    with ExitStack() as stack:
        for owner,attribute,name in targets:
            stack.enter_context(patch.object(owner,attribute,measured(getattr(owner,attribute),name)))
        started = time.perf_counter()
        result = processor.compute(prepared)
        total = time.perf_counter()-started
    return dict(seconds=total,stages=dict(seconds),fitting_calls=len(fit_clouds),
        fit_inputs=[dict(points=len(c),sha256=array_digest(c)) for c in fit_clouds],
        coarse_inputs=[dict(points=len(c),sha256=array_digest(c)) for c in coarse],
        detections=[dict(geometry(d.fit,finalized=True),views=list(d.views)) for d in result.detections],
        input_timestamp=prepared.timestamp,result_timestamp=result.timestamp)


def statistics(values):
    return dict(p50=float(np.median(values)),p95=float(np.percentile(values,95)),minimum=min(values),maximum=max(values))


def benchmark(args):
    root = Path.cwd().resolve()
    for name,module in list(os.sys.modules.items()):
        if name.startswith('robot_skill_stack') and getattr(module,'__file__',None):
            assert Path(module.__file__).resolve().is_relative_to(root), f'Wrong checkout: {module.__file__}'
    head = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert head==args.expected_head
    assert not args.output.exists(), 'Keep each benchmark result separate'
    manifest = json.loads(args.data.read_text())
    config = load_scene_config('config/scenes/multiview_perception_final.toml')
    assert digest('config/scenes/multiview_perception_final.toml')==manifest['config_sha256']
    s = config.multi_view
    processor = provider.MultiViewProcessor(config.scene,voxel_size=s.voxel_size,stride=s.stride,max_points=12000)
    before = environment()
    if args.mode=='compute':
        fs = native_frames(manifest['cases'][args.scenario])
        values = dict(frames=fs,timestamp=min(f.acquired_at for f in fs))
        if 'acquisition_ms' in {f.name for f in fields(provider.MultiViewInput)}:
            values.update(acquisition_ms=0.,synchronization={'state':'historical_offline_replay'})
        prepared = provider.MultiViewInput(**values)
        execute = lambda: profiled_compute(processor,prepared)
    else:
        assert digest(manifest['cloud_path'])==manifest['cloud_sha256']
        with np.load(manifest['cloud_path'],allow_pickle=False) as data:
            cloud = owned(data[args.scenario])
        assert array_digest(cloud)==manifest['clouds'][args.scenario]['sha256']
        def execute():
            started = time.perf_counter()
            fit = processor.fitter.fit(cloud,support_plane_z=config.scene.support_plane_z)
            return dict(seconds=time.perf_counter()-started,fit=geometry(fit,finalized=False),
                        cloud_sha256=array_digest(cloud),points=len(cloud))
    warmup = [execute()['seconds'] for _ in range(args.warmup)]
    runs = [execute() for _ in range(args.repeats)]
    sources = {name:dict(path=str(Path(module.__file__).resolve()),sha256=digest(module.__file__))
               for name,module in [('provider',provider),('segmentation',segmentation),('fitting',fitting),
                                   ('clouds',clouds),('acquisition',acquisition),('components',components),
                                   ('box_fitting',box_fitting),('depth_edges',depth_edges)]}
    result = dict(head=head,root=str(root),mode=args.mode,scenario=args.scenario,
        harness_sha256=digest(__file__),input_manifest_sha256=digest(args.data),sources=sources,
        environment_before=before,environment_after=environment(),warmup_seconds=warmup,runs=runs,
        summary_seconds=statistics([r['seconds'] for r in runs]))
    if args.mode=='compute':
        result['stages_seconds'] = {name:statistics([r['stages'].get(name,0) for r in runs])
            for name in ('projection_voxel','coarse','refinement','metric_graph','fit','parameters','observability')}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(head=head,mode=args.mode,scenario=args.scenario,seconds=result['summary_seconds'],
                         fitting_calls=[r.get('fitting_calls') for r in runs])))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','compute','fit'))
    parser.add_argument('--captures',type=Path)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--scenario')
    parser.add_argument('--expected-head')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=5)
    args = parser.parse_args()
    if args.mode=='prepare':
        prepare(args.captures,args.output)
    else:
        benchmark(args)
