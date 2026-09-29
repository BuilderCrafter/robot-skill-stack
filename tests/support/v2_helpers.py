from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from robot_skill_stack.world.perception.v2.types import CLASSES, Instance, Segmentation
from tests.support.primitive_scene import camera, intersect, render, FrameCamera, Solid

INFO = dict(ready=True, classes=list(CLASSES), sha256='0'*64, test_double=True)


def scene(solids=(Solid('sphere'),), **kwargs):
    frame = render(solids, **kwargs)
    _, T, rays = camera(eye=kwargs.get('eye', (1.05, .55, .9)), resolution=kwargs.get('resolution', (320, 240)))
    depth = frame.depth.ravel()
    masks = [np.isclose(intersect(s, T[:3, 3], rays), depth, atol=1e-7, rtol=0).reshape(frame.depth.shape)
             & np.isfinite(frame.depth) for s in solids]
    instances = [Instance(m, s.shape, .95) for s, m in zip(solids, masks)]
    return frame, instances


class FixtureClient:
    def __init__(self, instances):
        self.instances, self.calls = instances, []

    def health(self):
        return INFO

    def predict(self, frame_id, rgb):
        self.calls.append((frame_id, rgb.copy()))
        return Segmentation(frame_id, self.instances, INFO, .001)


class ManualExecutor:
    def __init__(self):
        self.jobs, self.stopped = [], False

    def submit(self, fn, *args):
        f = Future()
        self.jobs.append((f, fn, args))
        return f

    def finish(self):
        f, fn, args = self.jobs.pop(0)
        if f.cancelled():
            return
        try:
            f.set_result(fn(*args))
        except Exception as exc:
            f.set_exception(exc)

    def shutdown(self, **kwargs):
        self.stopped = True
        for f, _, _ in self.jobs:
            f.cancel()
