from dataclasses import dataclass, field
import numpy as np

CLASSES = ('cube', 'sphere', 'cylinder')


@dataclass(frozen=True)
class Instance:
    mask: np.ndarray
    label: str | None
    confidence: float

    def __post_init__(self):
        mask = np.asarray(self.mask)
        if mask.ndim != 2 or mask.dtype != np.bool_:
            raise ValueError('An instance mask must be HxW bool, in original-image coordinates')
        if self.label not in (*CLASSES, None) or not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError('Invalid instance class / score')
        object.__setattr__(self, 'mask', mask.copy())


@dataclass
class Segmentation:
    frame_id: int
    instances: list[Instance]
    model: dict = field(default_factory=dict)
    inference_s: float = 0.


def validate_frame(frame):
    if frame.rgb.dtype != np.uint8 or frame.rgb.shape[2] != 3:
        raise ValueError('V2 needs RGB uint8 with exactly three channels')
    K, T = frame.intrinsics, frame.world_from_camera
    if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError('Invalid intrinsics')
    if not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1])+abs(K[1, 0]) > 1e-8:
        raise ValueError('V2 expects zero-skew pinhole intrinsics')
    if (not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-3)
            or np.linalg.det(T[:3, :3]) < .99 or not np.isfinite(frame.timestamp)):
        raise ValueError('Invalid camera-to-world rigid transform / timestamp')
