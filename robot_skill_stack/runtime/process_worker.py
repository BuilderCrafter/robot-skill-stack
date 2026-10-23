"""Bounded observation work in a dedicated, SDK-free Python interpreter.

Only the inherited background thread starts the interpreter or serializes data.
The owning context keeps the same nonblocking submit/poll/invalidate/close API.
Pipes are trusted local transport, never an interface for untrusted pickle data.
"""
import io
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import threading
from types import MappingProxyType

import numpy as np

from .worker import BoundedWorker


def _restore_mapping(values):
    return MappingProxyType(values)


def _restore_readonly_array(data, dtype, shape):
    # Bytes backing, rather than a reversible writeable=False flag, is required
    # for published snapshots shared with owning-context readers.
    return np.frombuffer(data, dtype=dtype).reshape(shape)


class _TransportPickler(pickle.Pickler):
    def reducer_override(self, value):
        if isinstance(value, MappingProxyType):
            return _restore_mapping, (dict(value),)
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            if value.dtype.hasobject:
                raise TypeError('Observation transport requires numeric, non-object arrays')
            return _restore_readonly_array, (value.tobytes(), value.dtype, value.shape)
        return NotImplemented


def _send(stream, value):
    buffer = io.BytesIO()
    _TransportPickler(buffer, protocol=5).dump(value)
    data = buffer.getvalue()
    for part in (struct.pack('!Q', len(data)), data):
        view = memoryview(part)
        while view:
            written = stream.write(view)
            if not written:
                raise BrokenPipeError('Observation subprocess protocol write failed')
            view = view[written:]
    stream.flush()


def _read_exact(stream, count):
    chunks = []
    remaining = count
    while remaining:
        data = stream.read(remaining)
        if not data:
            raise EOFError('Observation subprocess closed its protocol pipe')
        chunks.append(data)
        remaining -= len(data)
    return b''.join(chunks)


def _receive(stream):
    size, = struct.unpack('!Q', _read_exact(stream, 8))
    return pickle.loads(_read_exact(stream, size))


class ProcessWorker(BoundedWorker):
    """One persistent child processor, one running job and one pending job.

    Pure computation exceptions retain the child and its processor state. A
    broken transport is terminal until the service is explicitly reinitialized;
    silently creating an empty replacement map would lose evidence history.
    """
    def __init__(self, process, *, name='observation-process'):
        if not Path(sys.executable).name.lower().startswith('python'):
            raise RuntimeError('Observation subprocesses require a Python interpreter; '
                               'launch Isaac with ./run_isaac_sim.sh instead of embedded kit/kit')
        self._processor = process
        self._child = None
        self._transport_error = None
        self._child_lock = threading.Lock()
        super().__init__(self._remote_compute, name=name+'-io')

    @property
    def health(self):
        values = super().health
        with self._child_lock:
            child = self._child
            values.update(execution='process', pid=None if child is None else child.pid,
                          transport_error=self._transport_error)
        return values

    def close(self):
        super().close()
        # Do not close buffered pipes here: the IO thread may hold their lock.
        # terminate() neither waits for computation nor for process exit.
        with self._child_lock:
            child = self._child
        if child is not None:
            try:
                child.terminate()
            except ProcessLookupError:
                pass

    def _start_child(self):
        child = subprocess.Popen([sys.executable, '-m',
            'robot_skill_stack.runtime.worker_process'], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=None, bufsize=0)
        with self._child_lock:
            self._child = child
        # close may have won while Popen was starting without its lock.
        if self._closed:
            child.terminate()
            raise RuntimeError('Observation worker closed during subprocess startup')
        _send(child.stdin, ('initialize', self._processor))
        status, value = _receive(child.stdout)
        if status != 'ready':
            raise RuntimeError(f'Observation subprocess initialization failed: {value}')
        return child

    def _remote_compute(self, payload):
        if self._transport_error is not None:
            raise RuntimeError('Observation subprocess requires reinitialization: '+self._transport_error)
        try:
            child = self._child if self._child is not None else self._start_child()
            _send(child.stdin, ('compute', payload))
            status, value = _receive(child.stdout)
        except Exception as exc:
            with self._child_lock:
                self._transport_error = f'{type(exc).__name__}: {exc}'
            self._dispose_child()
            raise RuntimeError('Observation subprocess transport failed: '+self._transport_error) from exc
        if status == 'error':
            raise RuntimeError(value)
        if status != 'result':
            with self._child_lock:
                self._transport_error = f'Unexpected protocol response: {status!r}'
            self._dispose_child()
            raise RuntimeError(self._transport_error)
        return value

    def _dispose_child(self):
        child = self._child
        if child is None:
            return
        try:
            child.terminate()
        except ProcessLookupError:
            pass
        # This method runs only on the background thread, after pipe IO ends.
        for stream in (child.stdin, child.stdout):
            if stream is not None:
                stream.close()
        child.wait()

    def _run(self):
        try:
            super()._run()
        finally:
            self._dispose_child()
