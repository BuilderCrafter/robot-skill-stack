from dataclasses import dataclass
import math
from urllib.parse import urlsplit


@dataclass(frozen=True)
class V2Config:
    endpoint: str = 'http://127.0.0.1:8765'
    confidence: float = .35
    nms_iou: float = .50
    max_detections: int = 32
    request_hz: float = 5.0
    timeout_s: float = 10.0
    max_result_age_s: float = 2.0
    stale_frame_s: float = 2.5
    min_mask_pixels: int = 30
    max_observation_z: float = .55
    mask_erode_pixels: int = 0
    unknown_depth_fallback: bool = True

    def __post_init__(self):
        url = urlsplit(self.endpoint)
        if (url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost', '[::1]', '::1'}
                or url.path not in ('', '/') or url.query or url.fragment or url.username or url.password):
            raise ValueError('V2 endpoint must be a loopback HTTP URL, e.g. http://127.0.0.1:8765')
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError('Invalid V2 port')
        for name in ('confidence', 'nms_iou'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{name} must be in (0, 1]')
        for name in ('request_hz', 'timeout_s', 'max_result_age_s', 'stale_frame_s'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.max_result_age_s > self.stale_frame_s:
            raise ValueError('max_result_age_s must not exceed stale_frame_s')
        for name, lo, hi in (('max_detections', 1, 64), ('min_mask_pixels', 1, 100000), ('mask_erode_pixels', 0, 3)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f'{name} must be an integer in [{lo}, {hi}]')
        if not math.isfinite(self.max_observation_z) or not isinstance(self.unknown_depth_fallback, bool):
            raise ValueError('Invalid V2 height / fallback setting')

    def inference_options(self):
        return dict(confidence=self.confidence, nms_iou=self.nms_iou, max_detections=self.max_detections)
