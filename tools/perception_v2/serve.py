#!/usr/bin/env python3
"""Local vision worker. Launch with the training/ML venv Python, NOT Isaac's wrapper."""
from pathlib import Path
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # Deliberately do not add .deps.
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path.home() / '.robot-skill-stack-yolo'))
from robot_skill_stack.world.perception.v2 import wire


def make_server(model, port):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, data, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != '/health':
                self.send(404, b'{"error":"not found"}')
                return
            self.send(200, json.dumps({**model.info, 'ready': True}).encode())

        def do_POST(self):
            if self.path != '/infer':
                self.send(404, b'{"error":"not found"}')
                return
            if not lock.acquire(blocking=False):
                self.send(503, b'{"error":"worker busy; no requests queued"}')
                return
            try:
                self.connection.settimeout(15)
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= wire.MAX_BYTES:
                    raise ValueError('Invalid Content-Length')
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError('Incomplete request')
                frame_id, rgb, options = wire.decode_request(body)
                result = model.predict(frame_id, rgb, **options)
                self.send(200, wire.encode_response(result, rgb.shape[:2]), 'application/octet-stream')
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                print(f'[vision] {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
                try:
                    self.send(500, json.dumps({'error': f'{type(exc).__name__}: {exc}'}).encode())
                except OSError:
                    pass
            finally:
                lock.release()

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--device', default='0')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--check', action='store_true', help='Load, warm up, print identity, and exit')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Invalid port')
    if any(Path(p).name == '.deps' for p in sys.path):
        parser.error('Isaac .deps is on sys.path. Use the isolated ML venv, with PYTHONPATH unset.')
    from robot_skill_stack.integrations.vision.yolo_segmenter import YoloSegmenter
    import numpy as np
    model = YoloSegmenter(args.weights, args.device, args.imgsz)
    model.predict(-1, np.zeros((480, 640, 3), np.uint8))
    print(json.dumps(model.info, indent=2), flush=True)
    if args.check:
        return 0
    server = make_server(model, args.port)
    print(f'Ready: http://127.0.0.1:{args.port} (loopback only; Ctrl+C stops the worker)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
