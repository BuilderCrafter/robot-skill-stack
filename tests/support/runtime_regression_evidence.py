"""Manual diagnostic timing fixture, never an Isaac/Franka benchmark."""
import json
from pathlib import Path
import time
import numpy as np
from robot_skill_stack.world.perception.v2.mesh_shapes import round_triangles
from robot_skill_stack.world.perception.v2.self_filter import RobotSurfaceModel
from robot_skill_stack.world.perception.frame import PerceptionFrame


def measure():
    model=RobotSurfaceModel([(str(i),round_triangles('Sphere',.06,segments=96)) for i in range(11)])
    poses=np.tile(np.eye(4),(11,1,1))
    poses[:,0,3]=np.linspace(-.10,.10,11); poses[:,1,3]=np.linspace(-.20,.20,11); poses[:,2,3]=.6
    frame=PerceptionFrame(1,np.zeros((480,640,3),np.uint8),np.ones((480,640)),
                          np.array([[554.,0,320],[0,554,240],[0,0,1]]),np.eye(4),100.)
    times,outputs={},{}
    for name,method in [('previous_bvh',model.depth_reference),('batched_raster',model.depth)]:
        times[name]=[]
        for _ in range(3):
            start=time.perf_counter(); outputs[name]=method(frame,poses); times[name].append(time.perf_counter()-start)
    old,new=outputs.values(); good=np.isfinite(old)&np.isfinite(new)
    return dict(fixture='11 overlapping tessellated spheres, NOT the installed Franka',triangles=model.triangle_count,
                resolution=[640,480],python=__import__('sys').version.split()[0],numpy=np.__version__,
                times_seconds=times,median_seconds={k:float(np.median(v)) for k,v in times.items()},
                finite_pixel_masks_identical=bool(np.array_equal(np.isfinite(old),np.isfinite(new))),
                finite_pixels=int(good.sum()),maximum_depth_difference_m=float(np.max(abs(old[good]-new[good]))),
                tolerance_or_mesh_changed=False)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    report=measure(); args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
