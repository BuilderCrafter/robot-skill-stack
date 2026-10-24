"""Diagnostic-only mapping evidence capture; no SDK imports or map changes.

The native diagnostic may substitute this processor before constructing its
normal ProcessWorker. Timings-only writes bounded scalar JSONL. Full capture
also writes the original MappingInput and exact post-update arrays in the
mapping subprocess; those runs are explicitly excluded from latency claims.
Never install this processor in a production profile.
"""
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from robot_skill_stack.runtime.process_worker import _TransportPickler
from robot_skill_stack.world.spatial_mapping import rgbd
from robot_skill_stack.world.spatial_mapping.runtime import MappingProcessor


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class AuditedMappingProcessor(MappingProcessor):
    """Run the unchanged processor and observe its worker-local boundaries.

    ``max_captures`` bounds full payload/array files; scalar timing rows continue
    afterward. The original frames, robot snapshots and per-call clock readings
    support deterministic offline replay without simulator geometry queries.
    """
    def __init__(self, settings, *, clock=time.monotonic, obstacle_settings=None,
                 audit_dir, capture_inputs=False, max_captures=64,
                 capture_format='uncompressed'):
        if type(capture_inputs) is not bool or type(max_captures) is not int or max_captures < 1:
            raise ValueError('Capture mode and a positive capture limit are required')
        if capture_format not in ('uncompressed', 'compressed'):
            raise ValueError('Capture format must be uncompressed or compressed')
        self._source_clock = clock
        self.audit_dir = str(audit_dir)
        self.capture_inputs = capture_inputs
        self.max_captures = max_captures
        self.capture_format = capture_format
        self._job = 0
        self._active_row = None
        self._phase = 'processor'
        self._hooks = []
        self._arrays = {}
        self._snapshot_for_capture = None
        self._helper_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        super().__init__(settings, clock=self._audit_clock, obstacle_settings=obstacle_settings)

    def _audit_clock(self):
        value = self._source_clock()
        if self._active_row is not None:
            self._active_row['clock_reads'].append(dict(phase=self._phase, value=float(value)))
        return value

    def _time(self, name, function, *args, **kwargs):
        before, previous = time.perf_counter(), self._phase
        self._phase = name
        try:
            return function(*args, **kwargs)
        finally:
            self._phase = previous
            if self._active_row is not None:
                self._active_row['timings_seconds'].setdefault(name, []).append(time.perf_counter()-before)

    def _hook(self, instance, name, replacement):
        original = getattr(instance, name)
        self._hooks.append((instance, name, original))
        setattr(instance, name, replacement(original))

    def _full_capture(self):
        return self.capture_inputs and self._job <= self.max_captures

    def _install_hooks(self):
        def measured(name):
            return lambda original: lambda *args, **kwargs: self._time(name, original, *args, **kwargs)

        def masks(original):
            def invoke(frames, snapshots, *, pixels_by_camera=None):
                result = self._time('robot_filter', original, frames, snapshots,
                                    pixels_by_camera=pixels_by_camera)
                if self._full_capture() and pixels_by_camera is not None:
                    for index, frame in enumerate(frames):
                        y, x = pixels_by_camera[frame.camera_id]
                        self._arrays[f'camera_{index}_selected_y'] = np.asarray(y).copy()
                        self._arrays[f'camera_{index}_selected_x'] = np.asarray(x).copy()
                        self._arrays[f'camera_{index}_automatic_mask'] = np.asarray(result[frame.camera_id]).copy()
                return result
            return invoke

        def integrate(original):
            def invoke(free_indices, occupied_indices, **kwargs):
                result = self._time('occupancy_integrate', original, free_indices, occupied_indices, **kwargs)
                if self._full_capture():
                    self._arrays['integrate_free_indices'] = np.asarray(free_indices).copy()
                    self._arrays['integrate_occupied_indices'] = np.asarray(occupied_indices).copy()
                    self._active_row['integration_timestamp'] = kwargs.get('timestamp')
                    self._active_row['integration_now'] = kwargs.get('now')
                return result
            return invoke

        def replace(original):
            def invoke(occupied_indices, unknown_indices=None, **kwargs):
                result = self._time('occupancy_replace', original, occupied_indices, unknown_indices, **kwargs)
                if self._full_capture():
                    self._arrays['replace_occupied_indices'] = np.asarray(occupied_indices).copy()
                    if unknown_indices is not None:
                        self._arrays['replace_unknown_indices'] = np.asarray(unknown_indices).copy()
                return result
            return invoke

        def snapshot(original):
            def invoke(*args, **kwargs):
                result = self._time('snapshot', original, *args, **kwargs)
                if self._full_capture():
                    self._snapshot_for_capture = result
                return result
            return invoke

        self._hook(self.mapper, 'update', measured('rgbd_update'))
        self._hook(self.filter, 'masks', masks)
        self._hook(self.map, 'integrate', integrate)
        self._hook(self.map, 'replace', replace)
        self._hook(self.map, 'snapshot', snapshot)
        if self._extractor is not None:
            self._hook(self._extractor, 'extract', measured('obstacle_extract'))

    def _reset(self, generation):
        self._time('processor_reset', super()._reset, generation)
        self._install_hooks()

    def _persist(self, payload, completed):
        row = self._active_row
        folder = Path(self.audit_dir)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f'job-{os.getpid()}-{self._job:05d}'
        if self._full_capture():
            started = time.perf_counter()
            path = folder/(stem+'-input.pkl')
            with path.open('wb') as stream:
                _TransportPickler(stream, protocol=5).dump(payload)
            row['input_pickle'] = path.name
            snapshot = self._snapshot_for_capture
            if snapshot is not None:
                arrays = dict(self._arrays)
                arrays.update(states=snapshot.states, probabilities=snapshot.probabilities,
                    observed=snapshot.observed, last_observed=snapshot.last_observed,
                    expired=snapshot.expired)
                path = folder/(stem+'-arrays.npz')
                writer = np.savez if self.capture_format == 'uncompressed' else np.savez_compressed
                writer(path, **arrays)
                row['arrays_npz'] = path.name
                row['capture_writer'] = 'numpy.savez' if self.capture_format == 'uncompressed' else 'numpy.savez_compressed'
                row['snapshot'] = _plain(snapshot.metadata)
            row['capture_write_seconds'] = time.perf_counter()-started
            row['capture_qualification'] = 'Full capture adds worker disk/serialization cost. Do not use this run to claim source latency or control performance.'
        if completed is not None:
            row['accepted'] = completed.accepted
            row['diagnostics'] = _plain(completed.diagnostics)
            row['input_timestamp'] = completed.input_timestamp
            packet = completed.obstacles
            if packet is not None:
                row['packet'] = dict(generation=packet.generation, revision=packet.revision,
                    timestamp=packet.timestamp, extracted_at=packet.extracted_at,
                    diagnostics=_plain(packet.diagnostics), proxy_count=len(packet.boxes))
                if self._full_capture():
                    row['packet']['boxes'] = [dict(id=box.id,
                        center=box.center, dimensions=box.dimensions,
                        expires_at=box.expires_at, voxel_count=box.voxel_count) for box in packet.boxes]
        row['metadata_json'] = stem+'.json'
        content = json.dumps(_plain(row), indent=2, allow_nan=False)+'\n'
        (folder/row['metadata_json']).write_text(content)
        # One worker owns this directory. Every line is independently readable,
        # including a final interrupted diagnostic job.
        with (folder/'jobs.jsonl').open('a') as stream:
            stream.write(json.dumps(_plain(row), separators=(',', ':'), allow_nan=False)+'\n')

    def compute(self, payload):
        self._job += 1
        self._arrays = {}
        self._snapshot_for_capture = None
        self._active_row = dict(schema_version=1, job=self._job, pid=os.getpid(),
            generation=payload.generation, source=self.settings.source,
            capture_inputs=self.capture_inputs, capture_selected=self._full_capture(),
            capture_format=self.capture_format,
            max_captures=self.max_captures, helper_sha256=self._helper_sha256,
            camera_ids=[frame.camera_id for frame in payload.frames],
            frames=[dict(camera_id=frame.camera_id, token=frame.token,
                acquired_at=frame.acquired_at, shape=frame.depth.shape) for frame in payload.frames],
            clock_reads=[], timings_seconds={}, processing_started_at=self._source_clock())
        if self.map is not None and not payload.reset and payload.generation == self.generation:
            self._install_hooks()
        original_traversal = rgbd._traverse_rays
        def traverse(*args, **kwargs):
            result = self._time('ray_traversal', original_traversal, *args, **kwargs)
            self._active_row.setdefault('traversal_visits', []).append(int(result[1]))
            return result
        rgbd._traverse_rays = traverse
        completed = None
        before = time.perf_counter()
        try:
            completed = super().compute(payload)
            return completed
        except BaseException as exc:
            self._active_row['processing_error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            self._active_row['baseline_compute_seconds'] = time.perf_counter()-before
            self._active_row['processing_finished_at'] = self._source_clock()
            rgbd._traverse_rays = original_traversal
            for instance, name, original in reversed(self._hooks):
                setattr(instance, name, original)
            self._hooks = []
            try:
                self._persist(payload, completed)
            except Exception as exc:
                # A diagnostic disk failure must not replace the underlying
                # mapping result or its original computation exception.
                import sys
                print(f'Mapping evidence audit write failed: {type(exc).__name__}: {exc}', file=sys.stderr)
            finally:
                self._active_row = None
                self._arrays = {}
                self._snapshot_for_capture = None
