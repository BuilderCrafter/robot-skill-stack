import threading
import time
import unittest

from robot_skill_stack.runtime.worker import BoundedWorker


def wait_for(predicate):
    deadline = time.monotonic()+2
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.001)
    raise AssertionError('Worker did not reach expected state')


class ObservationWorkerTests(unittest.TestCase):
    def test_bounded_replacement_and_serial_ownership(self):
        started, release = threading.Event(), threading.Event()
        calls, owner = [], threading.get_ident()
        def process(value):
            calls.append((value, threading.get_ident()))
            if value == 1:
                started.set()
                self.assertTrue(release.wait(2))
            return value*10
        worker = BoundedWorker(process)
        self.addCleanup(worker.close)
        worker.submit(1, tag='first')
        self.assertTrue(started.wait(2))
        worker.submit(2)
        worker.submit(3, tag='newest')
        self.assertEqual(worker.health['replaced'], 1)
        self.assertTrue(worker.health['running'])
        self.assertTrue(worker.health['pending'])
        release.set()
        result = wait_for(lambda: worker.poll())
        if result.tag == 'first':
            result = wait_for(lambda: worker.poll())
        self.assertEqual((result.tag, result.value), ('newest', 30))
        self.assertEqual([call[0] for call in calls], [1, 3])
        self.assertEqual(len({call[1] for call in calls}), 1)
        self.assertNotEqual(calls[0][1], owner)

    def test_failure_barrier_discards_old_success_and_allows_recovery(self):
        started, release = threading.Event(), threading.Event()
        def process(value):
            if value == 1:
                started.set()
                release.wait(2)
            if value == 3:
                raise ValueError('broken computation')
            return value
        worker = BoundedWorker(process)
        self.addCleanup(worker.close)
        worker.submit(1)
        self.assertTrue(started.wait(2))
        worker.submit(2)
        worker.invalidate()
        release.set()
        wait_for(lambda: not worker.health['running'])
        self.assertIsNone(worker.poll())
        self.assertEqual(worker.health['discarded'], 2)
        worker.submit(3)
        failure = wait_for(worker.poll)
        self.assertEqual(failure.error, 'ValueError: broken computation')
        worker.submit(4)
        self.assertEqual(wait_for(worker.poll).value, 4)

    def test_close_is_nonblocking_and_never_publishes_late_result(self):
        started, release = threading.Event(), threading.Event()
        def process(value):
            started.set()
            release.wait(2)
            return value
        worker = BoundedWorker(process)
        worker.submit(1)
        self.assertTrue(started.wait(2))
        before = time.monotonic()
        worker.close()
        self.assertLess(time.monotonic()-before, .05)
        self.assertFalse(worker.submit(2))
        release.set()
        wait_for(lambda: not worker.health['running'])
        self.assertIsNone(worker.poll())
        self.assertEqual(worker.health['discarded'], 1)


if __name__ == '__main__':
    unittest.main()
