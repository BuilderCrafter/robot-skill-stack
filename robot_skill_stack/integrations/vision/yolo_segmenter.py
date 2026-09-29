"""Run only in the ML environment, never import Ultralytics into Isaac."""
from pathlib import Path
import hashlib
import time
import numpy as np
from robot_skill_stack.world.perception.v2.types import CLASSES, Instance, Segmentation


class YoloSegmenter:
    def __init__(self, weights, device='0', image_size=640):
        weights = Path(weights).expanduser().resolve()
        if not weights.is_file() or weights.suffix != '.pt':
            raise ValueError('Supply your existing, trusted trained best.pt; no weights are downloaded')
        if image_size < 64 or image_size > 1920 or image_size % 32:
            raise ValueError('image_size must be a multiple of 32 in [64, 1920]')
        import torch
        import ultralytics
        from ultralytics import YOLO
        if str(device) != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; select --device cpu explicitly for CPU testing')
        torch.set_num_threads(2)
        self.model = YOLO(str(weights))
        if self.model.task != 'segment':
            raise ValueError('Weights must be for INSTANCE SEGMENTATION, not detection')
        names = self.model.names
        self.names = {int(k): str(v).strip().lower() for k, v in
                      (names.items() if isinstance(names, dict) else enumerate(names))}
        if len(self.names) != 3 or set(self.names.values()) != set(CLASSES):
            raise ValueError(f'Expected exactly cube, sphere, cylinder; checkpoint names are {self.names}')
        self.device, self.image_size = device, image_size
        self.info = dict(classes=list(CLASSES), names=self.names, weights=str(weights),
                         sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                         ultralytics=ultralytics.__version__, torch=torch.__version__, numpy=np.__version__,
                         device=str(device), image_size=image_size, task='segment')

    def predict(self, frame_id, rgb, confidence=.35, nms_iou=.5, max_detections=32):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('RGB must be HxWx3 uint8')
        start = time.perf_counter()
        # Ultralytics treats NumPy images as BGR; retina_masks restores native resolution.
        result = self.model.predict(np.ascontiguousarray(rgb[..., ::-1]), imgsz=self.image_size,
                                    conf=confidence, iou=nms_iou, max_det=max_detections,
                                    device=self.device, half=False, retina_masks=True,
                                    agnostic_nms=True, verbose=False, save=False)[0]
        instances = []
        if result.boxes is not None and len(result.boxes):
            if result.masks is None:
                raise RuntimeError('The segmentation model returned boxes without masks')
            masks = result.masks.data.detach().cpu().numpy() > .5
            labels = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()
            if masks.shape != (len(labels), *rgb.shape[:2]) or len(scores) != len(labels):
                raise RuntimeError('Unexpected mask size/count; do not resize letterboxed masks blindly')
            instances = [Instance(m, self.names[int(c)], float(s)) for m, c, s in zip(masks, labels, scores) if m.any()]
        return Segmentation(frame_id, instances, self.info, time.perf_counter()-start)
