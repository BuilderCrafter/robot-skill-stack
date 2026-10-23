"""One SDK-free running computation and one replaceable pending observation."""
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class WorkResult:
    tag: object
    value: object
    error: str | None
    queued_at: float
    started_at: float
    completed_at: float

    @property
    def duration(self):
        return self.completed_at-self.started_at


class BoundedWorker:
    """Never runs caller code under the lock; close never waits for computation.

    ``process`` must own only SDK-independent data. Invalidating installs a
    publication barrier, rather than interrupting Python or resetting processor
    state concurrently. A processor that needs reset receives it in its input.
    """
    def __init__(self, process, *, name='observation-worker'):
        self._process = process
        self._condition = threading.Condition()
        self._pending = self._completed = None
        self._epoch, self._closed, self._running = 0, False, False
        self._counts = dict(submitted=0, started=0, completed=0, replaced=0,
                            discarded=0, errors=0, delivered=0)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, payload, tag=None):
        with self._condition:
            if self._closed:
                return False
            if self._pending is not None:
                self._counts['replaced'] += 1
            self._pending = (self._epoch, payload, tag, time.monotonic())
            self._counts['submitted'] += 1
            self._condition.notify()
            return True

    def poll(self):
        with self._condition:
            result, self._completed = self._completed, None
            if result is not None:
                self._counts['delivered'] += 1
            return result

    def invalidate(self):
        with self._condition:
            self._epoch += 1
            self._counts['discarded'] += int(self._pending is not None)+int(self._completed is not None)
            self._pending = self._completed = None
            return self._epoch

    @property
    def health(self):
        with self._condition:
            return dict(self._counts, running=self._running,
                        pending=self._pending is not None,
                        result_ready=self._completed is not None,
                        closed=self._closed, epoch=self._epoch)

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._epoch += 1
            self._counts['discarded'] += int(self._pending is not None)+int(self._completed is not None)
            self._pending = self._completed = None
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                epoch, payload, tag, queued_at = self._pending
                self._pending, self._running = None, True
                self._counts['started'] += 1
            started = time.monotonic()
            value, error = None, None
            try:
                value = self._process(payload)
            except Exception as exc:
                # Retain text, not traceback frames holding SDK-independent
                # input arrays (or arbitrary exception object references).
                error = f'{type(exc).__name__}: {exc}'
            result = WorkResult(tag, value, error, queued_at, started, time.monotonic())
            with self._condition:
                self._running = False
                self._counts['completed'] += 1
                self._counts['errors'] += int(error is not None)
                if self._closed or epoch != self._epoch:
                    self._counts['discarded'] += 1
                else:
                    self._counts['discarded'] += int(self._completed is not None)
                    self._completed = result
