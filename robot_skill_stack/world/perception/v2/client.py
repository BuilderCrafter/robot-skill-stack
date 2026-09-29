import json
from urllib import request, error
from robot_skill_stack.world.perception.v2 import wire
from robot_skill_stack.world.perception.v2.types import CLASSES


class VisionClient:
    def __init__(self, config):
        self.config = config
        self.url = config.endpoint.rstrip('/')
        # Never route localhost camera images through environment-configured proxies.
        self.opener = request.build_opener(request.ProxyHandler({}))

    def _call(self, route, body=None):
        req = request.Request(self.url+route, data=body,
                              headers={'Content-Type': 'application/octet-stream'})
        try:
            with self.opener.open(req, timeout=self.config.timeout_s) as response:
                data = response.read(wire.MAX_BYTES+1)
        except error.HTTPError as exc:
            raise RuntimeError(f'Vision worker HTTP {exc.code}: {exc.read(2048).decode(errors="replace")}') from exc
        except (OSError, error.URLError) as exc:
            raise RuntimeError(f'Vision worker unavailable at {self.url}: {exc}') from exc
        if len(data) > wire.MAX_BYTES:
            raise ValueError('Worker response too large')
        return data

    def health(self):
        result = json.loads(self._call('/health'))
        if not result.get('ready') or result.get('classes') != list(CLASSES):
            raise RuntimeError('Worker is not ready with the expected segmentation classes')
        return result

    def predict(self, frame_id, rgb):
        body = wire.encode_request(frame_id, rgb, self.config.inference_options())
        return wire.decode_response(self._call('/infer', body), frame_id, rgb.shape[:2])
